from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from config.emotion.baseline import EmotionBaselineModelConfig
from data import Batch
from ..base import SSLModel, TextModel


class SERModel(nn.Module):
    def __init__(self, cfg: EmotionBaselineModelConfig, ssl_dim: int | None, text_dim: int | None):
        """Initialize unimodal or multimodal SER layers based on enabled inputs."""
        super().__init__()
        self.use_audio = cfg.use_audio
        self.use_text = cfg.use_text

        if not self.use_audio and not self.use_text:
            raise ValueError('At least one emotion input must be enabled.')

        self.dropout = nn.Dropout(cfg.dropout)

        if self.use_audio:
            if ssl_dim is None:
                raise ValueError('ssl_dim is required when use_audio is enabled.')
            self.speech_proj = nn.Linear(ssl_dim, cfg.d_model)
            self.speech_norm = nn.LayerNorm(cfg.d_model)
            self.speech_encoder = nn.GRU(
                input_size=cfg.d_model,
                hidden_size=cfg.d_model,
                batch_first=True,
                bidirectional=True,
            )
            self.speech_atten = nn.Linear(cfg.d_model * 2, 1)
            self.speech_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive)
            )
            self.speech_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive)
            )

        if self.use_text:
            if text_dim is None:
                raise ValueError('text_dim is required when use_text is enabled.')
            self.text_proj = nn.Linear(text_dim, cfg.d_model)
            self.text_norm = nn.LayerNorm(cfg.d_model)
            self.text_encoder = nn.GRU(
                input_size=cfg.d_model,
                hidden_size=cfg.d_model,
                batch_first=True,
                bidirectional=True,
            )
            self.text_atten = nn.Linear(cfg.d_model * 2, 1)
            self.text_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive)
            )
            self.text_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive)
            )

        if self.use_audio and self.use_text:
            self.speech_attention = nn.MultiheadAttention(cfg.d_model * 2, 1, dropout=cfg.dropout, batch_first=True)
            self.text_attention = nn.MultiheadAttention(cfg.d_model * 2, 1, dropout=cfg.dropout, batch_first=True)

        fusion_input_dim = cfg.d_model * 4 if self.use_audio and self.use_text else cfg.d_model * 2
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.classifier_proj = nn.Linear(fusion_input_dim, cfg.d_model)
        self.classifier_activation = nn.ReLU()
        self.classifier_dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)
        self.fusion_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive)
        )

    def attention_pool(self, features, attention_layer, mask=None):
        # features: [batch, seq_len, hidden]

        # Calculate attention scores
        """Pool sequence features with learned attention weights and an optional mask."""
        attn_weights = attention_layer(features)  # [batch, seq_len, 1]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply attention
        weighted_features = features * attn_weights
        pooled = weighted_features.sum(dim=1)  # [batch, hidden]

        return pooled

    def forward(self, raw_speech_feat, raw_text_feat, b):
        """Encode available modalities, optionally cross-attend, and classify."""
        speech_feat = None
        speech_pooled = None
        text_feat = None
        text_pooled = None
        pooled_features = []

        if self.use_audio:
            if raw_speech_feat is None:
                raise ValueError('use_audio requires speech features.')
            speech_feat = self.speech_norm(self.speech_proj(raw_speech_feat))
            speech_x, _ = self.speech_encoder(speech_feat)
        else:
            speech_x = None

        if self.use_text:
            if raw_text_feat is None:
                raise ValueError('use_text requires text features.')
            text_feat = self.text_norm(self.text_proj(raw_text_feat))
            text_x, _ = self.text_encoder(text_feat)
        else:
            text_x = None

        if self.use_audio and self.use_text:
            speech_attended, _ = self.speech_attention(query=speech_x, key=text_x, value=text_x)
            text_attended, _ = self.text_attention(query=text_x, key=speech_x, value=speech_x)
            speech_x = speech_x + speech_attended
            text_x = text_x + text_attended

        if self.use_audio:
            speech_pooled = self.attention_pool(speech_x, self.speech_atten)
            pooled_features.append(speech_pooled)

        if self.use_text:
            text_pooled = self.attention_pool(text_x, self.text_atten)
            pooled_features.append(text_pooled)

        fusion_emb = pooled_features[0] if len(pooled_features) == 1 else torch.cat(pooled_features, dim=-1)
        fusion_emb = self.fusion_norm(fusion_emb)
        fusion_emb = self.classifier_proj(fusion_emb)
        x = self.classifier_activation(fusion_emb)
        x = self.classifier_dropout(x)
        logits = self.classifier(x)

        if self.training:
            embeddings = {}
            if self.use_audio:
                embeddings['speech_frame_emb'] = self.speech_contrastive_mlp(speech_feat.mean(dim=1))
                embeddings['speech_pooled_emb'] = self.speech_pooled_contrastive_mlp(speech_pooled)
            if self.use_text:
                embeddings['text_frame_emb'] = self.text_contrastive_mlp(text_feat.mean(dim=1))
                embeddings['text_pooled_emb'] = self.text_pooled_contrastive_mlp(text_pooled)
            embeddings['fusion_emb'] = self.fusion_contrastive_mlp(fusion_emb)

            return logits, x, embeddings
        return logits, x, None


@dataclass
class ModelOutput:
    logits: torch.Tensor
    feature: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


class EmotionBaselineModel(nn.ModuleDict):
    def __init__(self, cfg: EmotionBaselineModelConfig):
        """Initialize the emotion model modules and freeze the SSL feature extractor."""
        self.cfg = cfg
        self.use_audio = cfg.use_audio
        self.use_text = cfg.use_text

        if not self.use_audio and not self.use_text:
            raise ValueError('At least one emotion input must be enabled.')

        ssl_model = SSLModel(cfg) if self.use_audio else nn.Identity()
        text_model = TextModel(cfg) if self.use_text else nn.Identity()

        ssl_dim = ssl_model.ssl_bundle._params['encoder_embed_dim'] if self.use_audio else None
        text_dim = text_model.model.config.hidden_size if self.use_text else None

        super().__init__({
            'ssl_model': ssl_model,
            'text_model': text_model,
            'ser_model': SERModel(cfg, ssl_dim, text_dim)
        })

        if self.use_audio:
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

        if self.use_text:
            self.text_bundle = AutoTokenizer.from_pretrained(
                {'ROBERTA_BASE_ENCODER': 'roberta-base', 'ROBERTA_LARGE_ENCODER': 'roberta-large'}.get(cfg.text_model_str, cfg.text_model_str)
            )

    def forward(self, b: Batch):
        """Run the emotion model on a batch and return logits and embeddings."""
        raw_speech_feat = None
        raw_text_feat = None
        if self.use_audio:
            wavform = b.wavform.view(b.wavform.size(0), -1)
            length = b.length.view(-1)
            raw_speech_feat, _feat_length = self['ssl_model'](wavform, length)
        if self.use_text:
            raw_text_feat = self['text_model'](b.tokens, b.text_mask)
        logits, feature, embeddings = self['ser_model'](raw_speech_feat, raw_text_feat, b)

        return ModelOutput(
            logits=logits,
            feature=feature,
            embeddings=embeddings

        )
