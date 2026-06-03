from dataclasses import dataclass, field

from config.base import (
    DataloaderConfig,
    DatasetsConfig,
    GeneralConfig,
    LineBotConfig,
    SolverConfig,
    WandbConfig,
)


@dataclass
class EmotionBaselineModelConfig:
    name: str = "EmotionBaseline"
    tag: str = "v1"
    description: str = ""
    
    ssl_model_str: str = 'WAVLM_LARGE'
    text_model_str: str = 'ROBERTA_BASE_ENCODER'
    
    num_classes: int = 8
    d_model: int = 512
    d_contrastive: int = 512

    dropout: float = 0.5
    
@dataclass
class EmotionBaselineConfig:
    # general: GeneralConfig
    # model: DeepfakeBaselineModelConfig
    # dataset: DatasetsConfig
    # dataloader: DataloaderConfig
    # solver: SolverConfig
    # wandb: WandbConfig

    general: GeneralConfig              = field(default_factory=GeneralConfig)
    model: EmotionBaselineModelConfig  = field(default_factory=EmotionBaselineModelConfig)
    solver: SolverConfig                = field(default_factory=SolverConfig)
    datasets: DatasetsConfig             = field(default_factory=DatasetsConfig)
    dataloader: DataloaderConfig        = field(default_factory=DataloaderConfig)
    linebot: LineBotConfig              = field(default_factory=LineBotConfig)
    wandb: WandbConfig                  = field(default_factory=WandbConfig)