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
class DeepfakeBaselineModelConfig:
    name: str = "DeepfakeAudioRhythmTransformerModel"
    tag: str = "v1"
    description: str = ""
    
    ssl_model_str: str = 'WAVLM_LARGE'
    text_model_str: str = 'ROBERTA_BASE_ENCODER'
    use_audio: bool = True
    use_rhythm: bool = False
    use_text: bool = False
    rhythm_sources: list[str] = field(default_factory=lambda: ['word', 'vowel', 'consonant'])
    
    num_classes: int = 2
    d_model: int = 64
    d_contrastive: int = 16
    n_heads: int = 4
    n_cls_encoder_layers: int = 1
    n_rhythm_encoder_layers: int = 2
    n_inter_encoder_layers: int = 1

    dropout: float = 0.3
    
@dataclass
class DeepfakeBaselineConfig:
    general: GeneralConfig              = field(default_factory=GeneralConfig)
    model: DeepfakeBaselineModelConfig  = field(default_factory=DeepfakeBaselineModelConfig)
    solver: SolverConfig                = field(default_factory=SolverConfig)
    datasets: DatasetsConfig            = field(default_factory=DatasetsConfig)
    dataloader: DataloaderConfig        = field(default_factory=DataloaderConfig)
    linebot: LineBotConfig              = field(default_factory=LineBotConfig)
    wandb: WandbConfig                  = field(default_factory=WandbConfig)
