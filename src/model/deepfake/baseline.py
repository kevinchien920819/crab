from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.deepfake.baseline import DeepfakeBaselineModelConfig
from data import Batch
from ..base import RhythmEncoder
from ..embedding import PositionalEncoding


def _length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create a boolean mask where True marks valid sequence positions."""
    lengths = lengths.to(dtype=torch.long).clamp(min=0, max=max_len)
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


class AudioRhythmFusionModel(nn.Module):
    """Fuse SSL audio tokens and rhythm tokens before utterance-level pooling."""

    def __init__(self, cfg: DeepfakeBaselineModelConfig, ssl_dim: int | None):
        super().__init__()
        self.use_audio = cfg.use_audio
        self.use_rhythm = cfg.use_rhythm

        if not self.use_audio and not self.use_rhythm:
            raise ValueError('At least one deepfake input must be enabled.')

        if self.use_audio:
            if ssl_dim is None:
                raise ValueError('ssl_dim is required when use_audio is enabled.')
            self.audio_proj = nn.Linear(ssl_dim, cfg.d_model)
            self.audio_norm = nn.LayerNorm(cfg.d_model)
            self.audio_positional_encoding = PositionalEncoding(cfg.d_model)
            if cfg.n_cls_encoder_layers > 0:
                self.audio_encoder = nn.TransformerEncoder(
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model         = cfg.d_model,
                        nhead           = cfg.n_heads,
                        dim_feedforward = 4 * cfg.d_model,
                        dropout         = cfg.dropout,
                        activation      = 'gelu',
                        batch_first     = True,
                        norm_first      = True,
                    ),
                    num_layers=cfg.n_cls_encoder_layers,
                )
            else:
                self.audio_encoder = nn.Identity()
            self.audio_atten = nn.Linear(cfg.d_model, 1)

        if self.use_rhythm:
            self.rhythm_encoder = RhythmEncoder(cfg)
            self.rhythm_atten = nn.Linear(cfg.d_model, 1)

        if self.use_audio and self.use_rhythm:
            self.audio_to_rhythm = nn.MultiheadAttention(
                cfg.d_model,
                cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.rhythm_to_audio = nn.MultiheadAttention(
                cfg.d_model,
                cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            if cfg.n_inter_encoder_layers > 0:
                inter_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                self.inter_encoder = nn.TransformerEncoder(
                    inter_encoder_layer,
                    num_layers=cfg.n_inter_encoder_layers,
                )
            else:
                self.inter_encoder = nn.Identity()

        self.fusion_atten = nn.Linear(cfg.d_model, 1)
        self.fusion_norm = nn.LayerNorm(cfg.d_model)
        self.classifier_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.classifier_dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

        self.audio_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive),
        )
        self.rhythm_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive),
        )
        self.fusion_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive),
        )

    def attention_pool(
        self,
        features: torch.Tensor,
        attention_layer: nn.Linear,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_weights = attention_layer(features)
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        return (features * attn_weights).sum(dim=1)

    def forward(
        self,
        raw_audio_feat: torch.Tensor | None,
        audio_feat_length: torch.Tensor | None,
        b: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        embeddings = {}
        joint_tokens = []
        joint_masks = []

        if self.use_audio:
            if raw_audio_feat is None or audio_feat_length is None:
                raise ValueError('use_audio requires SSL audio features and feature lengths.')
            audio_mask = _length_to_mask(audio_feat_length, raw_audio_feat.size(1))
            audio_x = self.audio_norm(self.audio_proj(raw_audio_feat))
            audio_x = self.audio_positional_encoding(audio_x)
            if isinstance(self.audio_encoder, nn.Identity):
                audio_x = self.audio_encoder(audio_x)
            else:
                audio_x = self.audio_encoder(audio_x, src_key_padding_mask=~audio_mask)
        else:
            audio_x = None
            audio_mask = None

        if self.use_rhythm:
            rhythm_x, rhythm_mask = self.rhythm_encoder(b)
        else:
            rhythm_x = None
            rhythm_mask = None

        if self.use_audio and self.use_rhythm:
            audio_attended, _ = self.audio_to_rhythm(
                query=audio_x,
                key=rhythm_x,
                value=rhythm_x,
                key_padding_mask=~rhythm_mask,
            )
            rhythm_attended, _ = self.rhythm_to_audio(
                query=rhythm_x,
                key=audio_x,
                value=audio_x,
                key_padding_mask=~audio_mask,
            )
            audio_x = audio_x + audio_attended
            rhythm_x = rhythm_x + rhythm_attended
            joint_x = torch.cat([audio_x, rhythm_x], dim=1)
            joint_mask = torch.cat([audio_mask, rhythm_mask], dim=1)
            if isinstance(self.inter_encoder, nn.Identity):
                joint_x = self.inter_encoder(joint_x)
            else:
                joint_x = self.inter_encoder(joint_x, src_key_padding_mask=~joint_mask)
        elif self.use_audio:
            joint_x = audio_x
            joint_mask = audio_mask
        else:
            joint_x = rhythm_x
            joint_mask = rhythm_mask

        if self.use_audio:
            audio_pooled = self.attention_pool(audio_x, self.audio_atten, audio_mask)
            embeddings['audio_emb'] = self.audio_contrastive_mlp(audio_pooled)
            joint_tokens.append(audio_x)
            joint_masks.append(audio_mask)
        if self.use_rhythm:
            rhythm_pooled = self.attention_pool(rhythm_x, self.rhythm_atten, rhythm_mask)
            embeddings['rhythm_emb'] = self.rhythm_contrastive_mlp(rhythm_pooled)
            joint_tokens.append(rhythm_x)
            joint_masks.append(rhythm_mask)

        fusion_pooled = self.attention_pool(joint_x, self.fusion_atten, joint_mask)
        fusion_pooled = self.fusion_norm(fusion_pooled)
        feature = F.relu(self.classifier_proj(fusion_pooled))
        feature = self.classifier_dropout(feature)
        logits = self.classifier(feature)
        embeddings['fusion_emb'] = self.fusion_contrastive_mlp(fusion_pooled)

        return logits, feature, embeddings if self.training else None
