from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from config.deepfake.baseline import DeepfakeBaselineModelConfig
from data.dataclass import Batch
from ..base import SSLModel
from .baseline import AudioRhythmFusionModel


@dataclass
class ModelOutput:
    logits: torch.Tensor
    feature: Optional[torch.Tensor] = None
    embeddings: Optional[dict[str, torch.Tensor]] = None


class DeepfakeAudioRhythmModel(nn.ModuleDict):
    """Audio deepfake detector with optional rhythm encoder fusion."""

    def __init__(self, cfg: DeepfakeBaselineModelConfig):
        self.cfg = cfg
        self.use_audio = cfg.use_audio
        self.use_rhythm = cfg.use_rhythm

        if not self.use_audio and not self.use_rhythm:
            raise ValueError('At least one deepfake input must be enabled.')

        ssl_model = SSLModel(cfg) if self.use_audio else nn.Identity()
        ssl_dim = ssl_model.ssl_bundle._params['encoder_embed_dim'] if self.use_audio else None

        super().__init__({
            'ssl_model': ssl_model,
            'fusion_model': AudioRhythmFusionModel(cfg, ssl_dim),
        })

        if self.use_audio:
            self._freeze_ssl_feature_extractor()

    def _freeze_ssl_feature_extractor(self) -> None:
        target_model = self['ssl_model'].model
        if hasattr(target_model, 'model'):
            target_model = target_model.model

        frozen = False
        if hasattr(target_model, 'freeze_feature_encoder'):
            target_model.freeze_feature_encoder()
            frozen = True

        for attr in ['feature_extractor', 'feature_encoder']:
            if not frozen and hasattr(target_model, attr):
                for param in getattr(target_model, attr).parameters():
                    param.requires_grad = False
                frozen = True

        if frozen:
            print(f"--- [DEBUG] SSL feature_extractor ({type(target_model).__name__}) has been frozen ---")
        else:
            print(f"--- [DEBUG] Warning: No trainable feature extractor found in {type(target_model).__name__} ---")

    def forward(self, b: Batch):
        raw_audio_feat = None
        audio_feat_length = None
        if self.use_audio:
            wavform = b.wavform.view(b.wavform.size(0), -1)
            length = b.length.view(-1)
            raw_audio_feat, audio_feat_length = self['ssl_model'](wavform, length)

        logits, feature, embeddings = self['fusion_model'](raw_audio_feat, audio_feat_length, b)
        return ModelOutput(logits=logits, feature=feature, embeddings=embeddings)


class DeepfakeBaseline(DeepfakeAudioRhythmModel):
    def __init__(self, cfg):
        super().__init__(cfg)


class DeepfakeBaselineModel(DeepfakeAudioRhythmModel):
    def __init__(self, cfg):
        super().__init__(cfg)


class DeepfakeCrabModel(DeepfakeAudioRhythmModel):
    def __init__(self, cfg):
        super().__init__(cfg)


def build_model(cfg):
    model_class = globals()[cfg.name]
    return model_class(cfg)
