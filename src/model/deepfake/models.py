from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from config.deepfake.baseline import DeepfakeBaselineModelConfig
from data.dataclass import Batch
from ..base import RhythmEncoder, SSLModel, TextModel
from .baseline import (
    DeepfakeAudioRhythmGRUModel as _DeepfakeAudioRhythmGRUModel,
    DeepfakeAudioRhythmTransformerModel as _DeepfakeAudioRhythmTransformerModel,
    DeepfakeAudioTextGRUModel as _DeepfakeAudioTextGRUModel,
    DeepfakeAudioTextTransformerModel as _DeepfakeAudioTextTransformerModel,
)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    feature: Optional[torch.Tensor] = None
    embeddings: Optional[dict[str, torch.Tensor]] = None


def _ssl_output_dim(ssl_model: SSLModel) -> int:
    if hasattr(ssl_model, 'output_dim'):
        return ssl_model.output_dim

    if hasattr(ssl_model, 'model') and hasattr(ssl_model.model, 'config'):
        hidden_size = getattr(ssl_model.model.config, 'hidden_size', None)
        if hidden_size is not None:
            return hidden_size

    if hasattr(ssl_model, 'ssl_bundle'):
        encoder_dim = ssl_model.ssl_bundle._params.get('encoder_embed_dim')
        if encoder_dim is not None:
            return encoder_dim

    raise ValueError('Unable to infer SSL output dimension.')


class _DeepfakeFrontendFusionModel(nn.ModuleDict):
    """Run enabled front-end encoders and pass their features to a fusion model."""

    fusion_model_cls: type[nn.Module]
    secondary_modality: str

    def __init__(self, cfg: DeepfakeBaselineModelConfig):
        use_audio = bool(cfg.use_audio)
        use_text = bool(getattr(cfg, 'use_text', False))
        use_rhythm = bool(cfg.use_rhythm)

        if self.secondary_modality == 'text' and use_rhythm:
            raise ValueError(f'{self.__class__.__name__} does not support use_rhythm=True.')
        if self.secondary_modality == 'rhythm' and use_text:
            raise ValueError(f'{self.__class__.__name__} does not support use_text=True.')

        ssl_model = SSLModel(cfg) if use_audio else nn.Identity()
        text_model = TextModel(cfg) if use_text else nn.Identity()
        rhythm_model = RhythmEncoder(cfg) if use_rhythm else nn.Identity()

        ssl_dim = _ssl_output_dim(ssl_model) if use_audio else None
        text_dim = text_model.model.config.hidden_size if use_text else None
        rhythm_dim = cfg.d_model if use_rhythm else None

        if self.secondary_modality == 'text':
            fusion_model = self.fusion_model_cls(cfg, ssl_dim, text_dim)
        else:
            fusion_model = self.fusion_model_cls(cfg, ssl_dim, rhythm_dim)

        super().__init__({
            'ssl_model': ssl_model,
            'text_model': text_model,
            'rhythm_model': rhythm_model,
            'fusion_model': fusion_model,
        })

        self.cfg = cfg
        self.use_audio = use_audio
        self.use_text = use_text
        self.use_rhythm = use_rhythm

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
        elif hasattr(target_model, 'freeze_feature_extractor'):
            target_model.freeze_feature_extractor()
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

    def _extract_audio_features(self, b: Batch) -> torch.Tensor | None:
        if not self.use_audio:
            return None
        if b.wavform is None:
            raise ValueError('use_audio requires b.wavform.')

        wavform = b.wavform.view(b.wavform.size(0), -1)
        length = b.length.view(-1) if b.length is not None else None
        raw_speech_feat, _ = self['ssl_model'](wavform, length)
        return raw_speech_feat

    def _extract_text_features(self, b: Batch) -> torch.Tensor | None:
        if not self.use_text:
            return None
        if b.tokens is None:
            raise ValueError('use_text requires b.tokens.')

        return self['text_model'](b.tokens, b.text_mask)

    def _extract_rhythm_features(self, b: Batch) -> torch.Tensor | None:
        if not self.use_rhythm:
            return None

        raw_rhythm_feat, _ = self['rhythm_model'](b)
        return raw_rhythm_feat

    def forward(self, b: Batch) -> ModelOutput:
        raw_speech_feat = self._extract_audio_features(b)

        if self.secondary_modality == 'text':
            raw_text_feat = self._extract_text_features(b)
            logits, feature, embeddings = self['fusion_model'](raw_speech_feat, raw_text_feat, b)
        else:
            raw_rhythm_feat = self._extract_rhythm_features(b)
            logits, feature, embeddings = self['fusion_model'](raw_speech_feat, raw_rhythm_feat, b)

        return ModelOutput(logits=logits, feature=feature, embeddings=embeddings)


class DeepfakeAudioTextTransformerModel(_DeepfakeFrontendFusionModel):
    fusion_model_cls = _DeepfakeAudioTextTransformerModel
    secondary_modality = 'text'


class DeepfakeAudioTextGRUModel(_DeepfakeFrontendFusionModel):
    fusion_model_cls = _DeepfakeAudioTextGRUModel
    secondary_modality = 'text'


class DeepfakeAudioRhythmTransformerModel(_DeepfakeFrontendFusionModel):
    fusion_model_cls = _DeepfakeAudioRhythmTransformerModel
    secondary_modality = 'rhythm'


class DeepfakeAudioRhythmGRUModel(_DeepfakeFrontendFusionModel):
    fusion_model_cls = _DeepfakeAudioRhythmGRUModel
    secondary_modality = 'rhythm'


def build_model(cfg):
    model_class = globals().get(cfg.name)
    if model_class is None:
        raise ValueError(f'Unknown deepfake model: {cfg.name}')
    return model_class(cfg)
