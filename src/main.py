import functools
import logging
import os
import time
import warnings
from dataclasses import asdict
from logging import Logger
from pathlib import Path

import click
import torch
import wandb

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torio.*")
from config import config_to_yaml, load_config
from config.deepfake.baseline import DeepfakeBaselineConfig
from config.emotion.baseline import EmotionBaselineConfig
from controller import Tester, Trainer
from data import DeepfakeDataset, EmotionDataset, get_dataloader, load_dataset
from data.loader import get_trial_path, resolve_subset_list
from tools import LineBot
from utils import set_seed, setup_freeze, setup_logger, setup_tf32


def handle_exceptions(logger: Logger):
    """Create a decorator that logs exceptions raised by wrapped functions."""
    def decorator(func):
        """Decorate a function with exception logging."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Call the wrapped function and log any exception before re-raising it."""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f'An error occurred in {func.__name__}: {e}')
                raise
        return wrapper
    return decorator


@click.command()
@click.option('--config-name', help='Configuration name to load', required=True)
def main(config_name) -> None:
    """CLI entry point that loads a config and starts the pipeline."""
    cfg = load_config(config_name)

    # 在任何 CUDA 操作之前，依 device_id 限定本 process 可見的 GPU。
    # 被選到的卡會在程式內重新編號為 cuda:0，因此 device 維持 'cuda' 即可。
    if cfg.general.device == 'cuda' and cfg.general.device_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.general.device_id)

    os.makedirs(cfg.general.work_dir, exist_ok=True)

    logger = setup_logger(
        name='main',
        project_root=os.getcwd(),
        log_file=f'{cfg.general.work_dir}/{time.strftime("%Y%m%d_%H%M%S")}_main.log',
        level=logging.INFO
    )

    logger.info(f'Using config: \n{config_to_yaml(cfg)}')

    pipeline(logger, cfg)

def _create_dataset(cfg, tokenizer):
    """根據 config 類型建立對應的 Dataset 物件。"""
    if isinstance(cfg, DeepfakeBaselineConfig):
        return DeepfakeDataset(tokenizer=tokenizer, text_max_len=cfg.dataloader.text_max_len)
    else:
        return EmotionDataset(tokenizer=tokenizer, text_max_len=cfg.dataloader.text_max_len)

def pipeline(logger: logging.Logger, cfg):
    # set seed and deterministic mode
    """Run model loading, training, evaluation, logging, and notifications."""
    set_seed(cfg.general.seed, cfg.general.deterministic)
    model = None
    if isinstance(cfg, EmotionBaselineConfig):
        from model.emotion.loader import load_model
        model = load_model(logger, cfg)
    elif isinstance(cfg, DeepfakeBaselineConfig):
        from model.deepfake.loader import load_model
        model = load_model(logger, cfg)


    if model is None:
        raise ValueError('Model loading failed. Please check the configuration and model loader.')

    # parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model Total Parameters: {total_params:,}')
    logger.info(f'Model Trainable Parameters: {trainable_params:,}')

    setup_freeze(cfg, logger, model)

    setup_tf32(cfg, logger)


    # setup wandb
    wandb_run = None
    if cfg.wandb.enable:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.model.name,
            tags=[cfg.model.tag],
            notes=cfg.model.description,
            config=asdict(cfg),
        )
        logger.info(f'WandB initialized: Project - {cfg.wandb.project}, Run Name - {cfg.model.name}')

    training_time = None
    tokenizer = getattr(model, 'text_bundle', None)
    # torchtext bundles need .transform(), transformers tokenizers do not
    if tokenizer is not None and hasattr(tokenizer, 'transform'):
        tokenizer = tokenizer.transform()

    if cfg.general.train:
        logger.info('Loading train and dev datasets')

        train_dataset = _create_dataset(cfg, tokenizer)
        train_dataset = load_dataset(cfg_dataset=cfg.datasets.train_datasets[0], dataset=train_dataset, subset_list=['train'])

        dev_dataset = _create_dataset(cfg, tokenizer)
        dev_dataset = load_dataset(cfg_dataset=cfg.datasets.train_datasets[0], dataset=dev_dataset, subset_list=['dev'])

        train_dataloader = get_dataloader(cfg, train_dataset, 'train', True)
        dev_dataloader = get_dataloader(cfg, dev_dataset, 'dev', False)

        logger.info(f'Start train and dev')

        trial_path = get_trial_path(cfg.datasets.train_datasets[0], 'dev')
        trainer = Trainer(
            logger,
            cfg,
            wandb_run,
            model,
            [train_dataloader, dev_dataloader],
            str(trial_path) if trial_path else None,
            cfg.datasets.train_datasets[0].name
        )

        starttime = time.time()
        trainer.run()
        endtime = time.time()

        training_time = endtime - starttime

        logger.info(f'Train and dev completed in {training_time/60:.2f} minutes')

        # Clean up trainer to release memory before evaluation
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cfg.general.eval:
        logger.info('Load eval dataset')
        for cfg_dataset in cfg.datasets.test_datasets:
            eval_subsets = resolve_subset_list(cfg_dataset, ['eval'])

            # Load checkpoint to CPU first to save VRAM
            checkpoint = torch.load(cfg.general.testing_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            del checkpoint
            logger.info(f'Loaded model from checkpoint: {cfg.general.testing_ckpt}')

            for eval_subset in eval_subsets:
                eval_dataset = _create_dataset(cfg, tokenizer)
                eval_dataset = load_dataset(cfg_dataset=cfg_dataset, dataset=eval_dataset, subset_list=[eval_subset])
                eval_dataloader = get_dataloader(cfg, eval_dataset, eval_subset, False)

                logger.info(
                    f'Eval dataset {cfg_dataset.name}/{eval_subset} loaded: '
                    f'{len(eval_dataset)} samples, {len(eval_dataloader)} batches.'
                )
                if len(eval_dataset) == 0:
                    logger.warning(f'Eval dataset {cfg_dataset.name}/{eval_subset} is empty. Skipping evaluation for this split.')
                    continue

                logger.info(f'Start eval for {cfg_dataset.name}/{eval_subset}')

                trial_path = get_trial_path(cfg_dataset, eval_subset)
                tester = Tester(
                    logger,
                    cfg,
                    wandb_run,
                    model,
                    [eval_dataloader],
                    str(trial_path) if trial_path else None,
                    cfg_dataset.name,
                    eval_split=eval_subset
                )
                if cfg_dataset.name in ['ASVspoof5', 'ASVspoof2019_LA', 'ASVspoof2021_LA', 'ASVspoof2021_DF']:
                    eer, loss = tester.run()
                    tester._calculate_minDCF_EER_CLLR_actDCF(
                        cm_scores_file=os.path.join(cfg.general.work_dir, 'evaluation_scores.txt'),
                        output_file=os.path.join(cfg.general.work_dir, f'{cfg_dataset.name}_{eval_subset}_result.txt'),
                        printout=True
                    )
                    metrices = {'EER': eer}
                elif cfg_dataset.name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
                    war, uar, loss = tester.run()
                    metrices = {'WAR': war, 'UAR': uar}
                else:
                    continue

                if cfg.linebot.enable:
                    linebot = LineBot(cfg.linebot.channel_access_token, cfg.linebot.user_id, logger)
                    linebot.send(
                        cfg,
                        total_params,
                        loss,
                        metrices
                    )

    if cfg.wandb.enable:
        wandb_run.finish()


if __name__ == '__main__':
    '''
    ex:
    python main.py --config-name=deepfake/baseline
    '''
    main()
