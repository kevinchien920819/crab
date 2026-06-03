from pathlib import Path
from typing import Union

from config.base import BaseConfig, DatasetConfig
from data import DeepfakeDataset,EmotionDataset
from torch.utils.data import DataLoader, Dataset


from .sampler import TokenBatchSampler, BalancedLengthSampler, PaddingBatchSampler

def resolve_subset_list(cfg_dataset: DatasetConfig, subset_list: list[str]) -> list[str]:
    """Resolve generic subset aliases into dataset-specific split names."""
    dataset_path = Path(cfg_dataset.dir)
    resolved_subsets: list[str] = []

    for subset in subset_list:
        if subset != 'eval':
            resolved_subsets.append(subset)
            continue

        if cfg_dataset.name == 'IEMOCAP':
            resolved_subsets.append('test')
        elif cfg_dataset.name == 'MELD':
            csv_dir = dataset_path / 'MELD' / 'csv'
            meld_eval_splits = sorted(
                path.name.replace('_sent_emo.csv', '')
                for path in csv_dir.glob('test*_sent_emo.csv')
            )
            resolved_subsets.extend(meld_eval_splits or ['test'])
        elif cfg_dataset.name == 'MSP_Podcast':
            resolved_subsets.extend(['test1', 'test2'])
        else:
            resolved_subsets.append('eval')

    return resolved_subsets

def load_dataset(cfg_dataset: DatasetConfig, dataset: Union[EmotionDataset, DeepfakeDataset], subset_list: list):
    """Dispatch dataset preloading based on the configured dataset name."""
    if cfg_dataset.name.startswith("ASVspoof") and not isinstance(dataset, DeepfakeDataset):
        raise TypeError(f"{cfg_dataset.name} need to use DeepfakeDataset, but got {type(dataset).__name__}")

    resolved_subset_list = resolve_subset_list(cfg_dataset, subset_list)

    if cfg_dataset.name == 'ASVspoof2019_LA':
        dataset.preload_asvspoof(Path(cfg_dataset.dir), subset_list=subset_list, use_duration=cfg_dataset.use_duration)
    elif cfg_dataset.name == 'ASVspoof2021_LA':
        dataset.preload_asvspoof(Path(cfg_dataset.dir), year = '2021_LA', subset_list=subset_list, use_duration=cfg_dataset.use_duration)
    elif cfg_dataset.name == 'ASVspoof2021_DF':
        dataset.preload_asvspoof(Path(cfg_dataset.dir), year = '2021_DF', subset_list=subset_list, use_duration=cfg_dataset.use_duration)
    elif cfg_dataset.name == 'ASVspoof5':
        dataset.preload_asvspoof(Path(cfg_dataset.dir), year = '5', subset_list=subset_list, use_duration=cfg_dataset.use_duration)
    if cfg_dataset.name == 'IEMOCAP':
        resolved_subset_list = resolve_subset_list(cfg_dataset, subset_list)
        dataset.preload_iemocap(Path(cfg_dataset.dir), subset_list=resolved_subset_list)
    elif cfg_dataset.name == 'MELD':
        resolved_subset_list = resolve_subset_list(cfg_dataset, subset_list)
        dataset.preload_meld(Path(cfg_dataset.dir), subset_list=resolved_subset_list)
    elif cfg_dataset.name == 'MSP_Podcast':
        resolved_subset_list = resolve_subset_list(cfg_dataset, subset_list)
        dataset.preload_msp_podcast(Path(cfg_dataset.dir), subset_list=resolved_subset_list)
    return dataset

def get_trial_path(cfg_dataset: DatasetConfig, subset: str) -> Path:
    """Return the protocol or trial metadata path for a dataset split."""
    dataset_path = Path(cfg_dataset.dir)
    if cfg_dataset.name == 'ASVspoof2019_LA':
        year = '2019_LA'
        year_prefix, track = year.split('_')
        asvspoof_str = f'ASVspoof{year_prefix}'
        protocol_suffix = 'trn' if subset == 'train' else 'trl'
        year_dot = year.replace('_', '.')
        return dataset_path / asvspoof_str / f'ASVspoof{year}_cm_protocols' / f'ASVspoof{year_dot}.cm.{subset}.{protocol_suffix}.txt'
    elif cfg_dataset.name == 'ASVspoof2021_LA':
        year = '2021_LA'
        year_prefix, track = year.split('_')
        asvspoof_str = f'ASVspoof{year_prefix}'
        return dataset_path / asvspoof_str / 'keys' / track / 'CM' / 'trial_metadata.txt'
    elif cfg_dataset.name == 'ASVspoof2021_DF':
        year = '2021_DF'
        year_prefix, track = year.split('_')
        asvspoof_str = f'ASVspoof{year_prefix}'
        return dataset_path / asvspoof_str / 'keys' / track / 'CM' / 'trial_metadata.txt'
    elif cfg_dataset.name == 'ASVspoof5':
        year = '5'
        year_dot = '5'
        asvspoof_str = 'ASVspoof5'
        return dataset_path / asvspoof_str / f'ASVspoof_cm_protocols' / f'ASVspoof{year_dot}.{subset}.txt'
    elif cfg_dataset.name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
        # Emotion datasets don't use trial files for scoring like ASVspoof
        return None
    else:
        raise NotImplementedError(f'Unsupported dataset name for trial path: {cfg_dataset.name}')


def get_dataloader(cfg: BaseConfig, dataset: Dataset, subset_name: str='', shuffle=True) -> DataLoader:
    """Create a DataLoader with the configured fixed or dynamic batching strategy."""
    max_tokens = cfg.dataloader.token_batch_size
    batch_size = cfg.dataloader.batch_size.get(
        subset_name,
        cfg.dataloader.batch_size.get('eval', cfg.dataloader.batch_size['train'])
    )
    lengths = dataset.get_lengths()
    batch_sampler = None

    # 1. PaddingBatchSampler
    if cfg.dataloader.name == 'PaddingBatchSampler' and max_tokens > 0:
        batch_sampler = PaddingBatchSampler(
            lengths=lengths,
            max_tokens=max_tokens,
            shuffle=shuffle,
            drop_last=False,
            seed=cfg.general.seed,
        )

    # 2. TokenBatchSampler
    if cfg.dataloader.name == 'TokenBatchSampler' and max_tokens > 0:
        batch_sampler = TokenBatchSampler(
            lengths=lengths,
            max_tokens=max_tokens,
            shuffle=shuffle,
            drop_last=False,
            seed=cfg.general.seed,
        )

    # 3. BalancedLengthSampler
    if cfg.dataloader.name == 'BalancedLengthSampler':
        batch_sampler = BalancedLengthSampler(
            lengths=lengths,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            seed=cfg.general.seed,
        )

    # 4. Default dynamic sampler fallback if token_batch_size is set but no specific sampler name is given
    if max_tokens > 0 and batch_sampler is None:
        batch_sampler = TokenBatchSampler(
            lengths=lengths,
            max_tokens=max_tokens,
            shuffle=shuffle,
            drop_last=False,
            seed=cfg.general.seed,
        )

    # If any batch_sampler is defined, use it
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            num_workers     = cfg.dataloader.num_workers,
            batch_sampler   = batch_sampler,
            collate_fn      = dataset.collate_fn_padded,
            # speed up
            pin_memory          = cfg.dataloader.pin_memory,
            persistent_workers  = cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
            prefetch_factor     = cfg.dataloader.prefetch_factor if cfg.dataloader.num_workers > 0 else None,
        )

    # 5. Default fixed batch size DataLoader
    return DataLoader(
        dataset,
        num_workers=cfg.dataloader.num_workers,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn_padded,
        shuffle=shuffle,
        # speed up
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
        prefetch_factor=cfg.dataloader.prefetch_factor if cfg.dataloader.num_workers > 0 else None,
    )
