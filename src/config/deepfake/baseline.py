from dataclasses import dataclass, field

from config.base import (
    DataloaderConfig,
    DatasetConfig,
    GeneralConfig,
    LineBotConfig,
    SolverConfig,
    WandbConfig,
)


@dataclass
class DeepfakeBaselineModelConfig:
    name: str = "DeepfakeBaseline"
    tag: str = "v1"
    description: str = ""
    
    ssl_model_str: str = 'WAVLM_LARGE'
    text_model_str: str = 'ROBERTA_BASE_ENCODER'
    
    num_classes: int = 8
    d_model: int = 64
    d_contrastive: int = 16

    lr: float = 1e-6
    dropout: float = 0.3
    
@dataclass
class DeepfakeBaselineConfig:
    # general: GeneralConfig
    # model: DeepfakeBaselineModelConfig
    # dataset: DatasetsConfig
    # dataloader: DataloaderConfig
    # solver: SolverConfig
    # wandb: WandbConfig

    general: GeneralConfig              = field(default_factory=GeneralConfig)
    model: DeepfakeBaselineModelConfig  = field(default_factory=DeepfakeBaselineModelConfig)
    solver: SolverConfig                = field(default_factory=SolverConfig)
    dataset: DatasetConfig             = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig        = field(default_factory=DataloaderConfig)
    linebot: LineBotConfig              = field(default_factory=LineBotConfig)
    wandb: WandbConfig                  = field(default_factory=WandbConfig)