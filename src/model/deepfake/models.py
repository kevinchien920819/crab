from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from ..base import SSLModel, TextModel, SERModel
from config.emotion.baseline import EmotionBaselineModelConfig
from data import Batch

@dataclass
class ModelOutput:
    logits: torch.Tensor
    feature: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None

class DeepfakeCrabModel(nn.ModuleDict):
    def __init__(self, cfg: EmotionBaselineModelConfig):
        """Initialize the deepfake model modules and freeze the SSL feature extractor."""
        self.cfg = cfg
        ssl_model = SSLModel(cfg)
        text_model = TextModel(cfg)

        ssl_dim = ssl_model.output_dim
        text_dim = text_model.model.config.hidden_size

        super().__init__({
            'ssl_model': ssl_model,
            'text_model': text_model,
            'ser_model': SERModel(cfg, ssl_dim, text_dim)
        })

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

        self.text_bundle = AutoTokenizer.from_pretrained(
            {'ROBERTA_BASE_ENCODER': 'roberta-base', 'ROBERTA_LARGE_ENCODER': 'roberta-large'}.get(cfg.text_model_str, cfg.text_model_str)
        )

    def forward(self, b: Batch):
        """Run the deepfake model on a batch and return logits and embeddings."""
        wavform = b.wavform.view(b.wavform.size(0), -1)
        length = b.length.view(-1)

        raw_speech_feat, _feat_length = self['ssl_model'](wavform, length)
        raw_text_feat = self['text_model'](b.tokens, b.text_mask)
        logits, feature, embeddings = self['ser_model'](raw_speech_feat, raw_text_feat, b)

        return ModelOutput(
            logits=logits,
            feature=feature,
            embeddings=embeddings

        )

