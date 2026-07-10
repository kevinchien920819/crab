import time
from logging import Logger

import numpy as np
import torch
from config import BaseConfig
from tqdm import tqdm
from utils import update_parameter_requires_grad
from wandb import Run

from .base import Controller
from .eval import compute_eer_from_labels, compute_uar, compute_war


class Trainer(Controller):
    def __init__(
            self,
            logger: Logger,
            cfg: BaseConfig,
            wandb_run: Run,
            model: torch.nn.Module,
            dataloaders: list[torch.utils.data.DataLoader],
            trial_file: str = None,
            dataset_name: str = None,
        ):

        """Initialize the training controller and metric tracking state."""
        super(Trainer, self).__init__(
            logger,
            cfg,
            model,
            dataloaders,
            trial_file,
            dataset_name
        )

        if dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
            self.best_score = 0.0 # for UAR
        else:
            self.best_score = float('inf') # lower EER is better
        self.wandb_run = wandb_run
        self.wandb_log = {}


    def run(self):
        """Run training by optimizer update step and log metrics against steps."""
        scaler = torch.amp.GradScaler(self.device) if self.amp_dtype == 'fp16' else None
        cuda_memory_device = self._cuda_memory_device()
        if cuda_memory_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_memory_device)

        train_dataloader = self.dataloaders[0]
        val_dataloader = self.dataloaders[1]

        self.logger.info(f'[Trainer] max_steps: {self.max_steps}')
        self.logger.info(f'[Trainer] steps_per_epoch: {self.steps_per_epoch}')
        self.logger.info(f'[Trainer] iters_to_accumulate: {self.cfg.solver.iters_to_accumulate}')
        if self.cfg.wandb.enable and self.wandb_run is not None:
            self.wandb_run.config.update(
                {
                    'solver.max_steps': self.max_steps,
                    'solver.steps_per_train_pass': self.steps_per_epoch,
                    'solver.freeze_steps': self.freeze_steps,
                },
                allow_val_change=True,
            )

        bar = tqdm(total=self.max_steps, initial=self.current_step, unit='step', desc='Train step')
        try:
            while self.current_step < self.max_steps:
                self._update_freeze()
                step_before = self.current_step

                startime = time.time()
                train_true, train_pred, train_loss, train_loss_list = self.do_epoch(train_dataloader, scaler, backward=True)
                self.completed_train_passes += 1

                if self.current_step == step_before:
                    self.logger.warning('[Trainer] No optimizer steps were completed; stopping training loop.')
                    break

                val_true, val_pred, val_loss, val_loss_list = self.do_epoch(val_dataloader, backward=False)
                endtime = time.time()

                bar.update(self.current_step - step_before)
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f'[Trainer] Step: {self.current_step}/{self.max_steps}, '
                    f'Train pass: {self.completed_train_passes}, '
                    f'Time: {endtime-startime:.2f}s, lr: {current_lr:.8f}'
                )
                self.logger.info(f'[Trainer] Train Loss components: ' + ', '.join([f'loss_{i}: {train_loss_list[i]:.6f}' for i in range(len(train_loss_list))]))
                self.logger.info(f'[Trainer] Val Loss components: ' + ', '.join([f'loss_{i}: {val_loss_list[i]:.6f}' for i in range(len(val_loss_list))]))

                wandb_log = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': current_lr,
                    'train_pass': self.completed_train_passes,
                }
                wandb_log.update({f'train_loss_components_{i}': train_loss_list[i] for i in range(len(train_loss_list))})
                wandb_log.update({f'val_loss_components_{i}': val_loss_list[i] for i in range(len(val_loss_list))})

                checkpoint_metric = None
                if self.dataset_name in ['ASVspoof2019_LA', 'ASVspoof2021_LA', 'ASVspoof2021_DF', 'ASVspoof5']:
                    train_eer = self.cal_eer(train_true, train_pred)
                    val_eer = self.cal_eer(val_true, val_pred)
                    checkpoint_metric = val_eer
                    self.logger.info(f'[Trainer] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}')
                    self.logger.info(f'[Trainer] Training EER: {train_eer:.6f}, Validation EER: {val_eer:.6f}')
                    wandb_log.update({'train_eer': train_eer, 'val_eer': val_eer})

                elif self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
                    train_uar = self.cal_uar(train_true, train_pred)
                    val_uar = self.cal_uar(val_true, val_pred)
                    train_war = self.cal_war(train_true, train_pred)
                    val_war = self.cal_war(val_true, val_pred)
                    checkpoint_metric = val_uar
                    self.logger.info(f'[Trainer] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}')
                    self.logger.info(f'[Trainer] Training UAR: {train_uar:.6f}, Validation UAR: {val_uar:.6f}')
                    self.logger.info(f'[Trainer] Training WAR: {train_war:.6f}, Validation WAR: {val_war:.6f}')
                    wandb_log.update({'train_war': train_war, 'val_war': val_war})
                    wandb_log.update({'train_uar': train_uar, 'val_uar': val_uar})

                if self.cfg.wandb.enable and self.wandb_run is not None:
                    self.wandb_run.log(wandb_log, step=self.current_step)

                self._save_checkpoint(metric=checkpoint_metric)

                if cuda_memory_device is not None:
                    peak = torch.cuda.max_memory_allocated(cuda_memory_device) / 1024**3
                    reserved_peak = torch.cuda.max_memory_reserved(cuda_memory_device) / 1024**3
                    self.logger.info(f'Peak allocated VRAM: {peak:.2f} GB')
                    self.logger.info(f'Peak reserved VRAM: {reserved_peak:.2f} GB')
        finally:
            bar.close()

    def _log_train_step(self, train_loss: float, train_loss_list: list[float]) -> None:
        """Stream optimizer-step train loss to WandB."""
        if not self.cfg.wandb.enable or self.wandb_run is None:
            return

        wandb_log = {
            'train_loss': train_loss,
            'lr': self.scheduler.get_last_lr()[0],
        }
        wandb_log.update({f'train_loss_components_{i}': train_loss_list[i] for i in range(len(train_loss_list))})
        self.wandb_run.log(wandb_log, step=self.current_step)

    def _cuda_memory_device(self):
        """Resolve the CUDA device used for memory statistics, if available."""
        if not torch.cuda.is_available():
            return None

        device = torch.device(self.device)
        if device.type != 'cuda':
            return None

        return device

    def _update_freeze(self):
        """Unfreeze configured model parameters after the freeze period ends."""
        if self.is_model_freeze and self.current_step >= self.freeze_steps:
            update_parameter_requires_grad(self.logger, self.model, 'unfreezing', self.cfg.general.freeze, True)
            self.is_model_freeze = False
            self.logger.info(f'[Trainer] Unfreezing model parameters at step {self.current_step}')

    def _save_checkpoint(self, metric: float = None):
        """Save the best checkpoint according to the dataset-specific validation metric."""
        is_best = False
        if self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
            if metric is not None and metric > self.best_score:
                self.best_score = metric
                is_best = True
                self.logger.info(f'[Trainer] Saving model checkpoint with UAR: {self.best_score:.6f}')
        else:
            if metric is not None and metric < self.best_score:
                self.best_score = metric
                is_best = True
                self.logger.info(f'[Trainer] Saving model checkpoint with EER: {self.best_score:.6f}')

        if is_best:
            torch.save(
                {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },
                f'{self.cfg.general.work_dir}/checkpoint.pt',
            )
            if self.cfg.wandb.enable:
                if self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
                    self.wandb_run.summary['best_uar'] = self.best_score
                else:
                    self.wandb_run.summary['best_eer'] = self.best_score
