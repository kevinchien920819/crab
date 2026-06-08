from dataclasses import dataclass, field


@dataclass
class GeneralConfig:
    device:     str = 'cuda'
    device_id:  str = '0'
    work_dir:   str = 'default'
    
    # torch, numpy, cuda
    seed: int = 39
    
    # cudnn.deterministic, cudnn.benchmark
    deterministic: bool = False
    
    # list of layer names to freeze or unfreeze, e.g., ['encoder', 'decoder']
    # ['all'] means all layers
    freeze:     list = field(default_factory=list)
    unfreeze:   list = field(default_factory=list)
    
    # model checkpoint
    ckpt: dict = field(default_factory=lambda: {
        'path': '',
        'modules': {
            'from': ['all'],
            'to': ['all'],
        }
    })
    
    # 'default' == '{work_dir}/checkpoints/checkpoint.pt'
    # 'same' == model.ckpt['path']
    testing_ckpt: str = 'default'
    
    train: bool = True
    eval: bool = True
    produce_evaluation_file: bool = False


@dataclass
class SolverConfig:
    optimizer: str = 'AdamW'
    weight_decay: float = 0.01
    
    scheduler: str = 'div'
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    min_lr: dict | float | None = None
    max_grad_norm: float = 1.0
    
    max_epochs: int = 100
    freeze_epochs: int = 0
    lr: dict = field(default_factory=lambda: {
        # 'ssl_model': 1e-6,
        # 'text_model': 1e-6,
        # 'ser_model': 1e-4,
    })
    iters_to_accumulate: int = 1
    amp_dtype: str = 'fp16'  # 'fp16', 'bf16', or 'none'
    
    criterions: dict = field(default_factory=lambda: {
        # 'ce_loss': {
        #     'name': 'CrossEntropyLoss',
        #     'label_smoothing': 0.1,
        #     'total_weight': 1.0,
        # }
    })

@dataclass
class LineBotConfig:
    enable: bool = True
    channel_access_token: str = ''
    user_id: str = ''


@dataclass
class DatasetConfig:
    name: str = 'LibriSpeech'
    dir: str = '/path/to/dataset'
    
    
    # for LibriSpeech
    #   train-clean-100, train-clean-360, train-other-500
    #   dev-clean, dev-other
    #   test-clean, test-other
    train_subset_list: list[str]    = field(default_factory=list) # lambda: ['train-clean-100', 'train-clean-360', 'train-other-500']
    dev_subset_list: list[str]      = field(default_factory=list)
    
    use_duration: bool = False  # for ASVspoof2019_LA


@dataclass
class DatasetsConfig:
    train_datasets: list[DatasetConfig] = field(default_factory=lambda: [DatasetConfig()])
    test_datasets: list[DatasetConfig] = field(default_factory=lambda: [DatasetConfig()])


@dataclass
class DataloaderConfig:
    name: str = 'Dataset'
    num_workers: int = 20
    pin_memory: bool = True             # 使用 pinned memory 加速 CPU->GPU 傳輸
    persistent_workers: bool = True     # DataLoader workers 常駐，減少每個 epoch 啟動開銷
    prefetch_factor: int = 4            # 每個 worker 預先準備的 batch 數量
    non_blocking_transfer: bool = True  # 搬移到 GPU 時使用 non_blocking，需搭配 pin_memory
    token_batch_size: int = 0           # 每個 batch 的總 token/frame 上限，0 表示停用
    text_max_len: int = 128             # 文字固定長度上限，與參考實作一致
    smoothed_class_batch_size: int = 500  # SmoothedClassBatchSampler 每個 mini-batch 的樣本數
    class_smoothing_power: float = 0.5    # 1.0 保留原分佈，0.0 接近類別平均
    
    batch_size: dict = field(default_factory=lambda: {
        'train': 128,
        'dev': 128,
        'eval': 128
    })


@dataclass
class WandbConfig:
    enable: bool = True
    entity: str = 'fcu-slp'
    project: str = 'ssl-mamba'


@dataclass
class BaseConfig:
    general:    GeneralConfig       = field(default_factory=GeneralConfig)
    solver:     SolverConfig        = field(default_factory=SolverConfig)
    datasets:    DatasetsConfig      = field(default_factory=DatasetsConfig)
    dataloader: DataloaderConfig    = field(default_factory=DataloaderConfig)
    wandb:      WandbConfig         = field(default_factory=WandbConfig)
