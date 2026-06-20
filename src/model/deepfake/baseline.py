from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.deepfake.baseline import DeepfakeBaselineModelConfig
from data import Batch
from ..base import RhythmEncoder
from ..embedding import PositionalEncoding
from ..pooling import attention_pool

class DeepfakeAudioTextTransformerModel(nn.Module):
    def __init__(self, cfg: DeepfakeBaselineModelConfig, ssl_dim: int | None, text_dim: int | None):
        super().__init__()
        self.use_audio = cfg.use_audio
        self.use_text = cfg.use_text
        if not self.use_audio and not self.use_text:
            raise ValueError('At least one deepfake input must be enabled.')

        self.dropout = nn.Dropout(cfg.dropout)

        n_heads = getattr(cfg, 'n_transformer_heads', 4)
        n_layers = getattr(cfg, 'n_transformer_encoder_layers', getattr(cfg, 'n_cls_encoder_layers', 2))
        ff_dim = getattr(cfg, 'transformer_ff_dim', cfg.d_model * 4)

        if cfg.d_model % n_heads != 0:
            raise ValueError(f'd_model ({cfg.d_model}) must be divisible by n_transformer_heads ({n_heads}).')

        if self.use_audio:
            if ssl_dim is None:
                raise ValueError('ssl_dim is required when use_audio is enabled.')

            self.speech_proj = nn.Linear(ssl_dim, cfg.d_model)
            self.speech_norm = nn.LayerNorm(cfg.d_model)
            self.speech_pos = PositionalEncoding(cfg.d_model)

            speech_encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.speech_encoder = nn.TransformerEncoder(
                speech_encoder_layer,
                num_layers=n_layers,
            )

            self.speech_atten = nn.Linear(cfg.d_model, 1)
            self.speech_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive),
            )
            self.speech_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive),
            )

        if self.use_text:
            if text_dim is None:
                raise ValueError('text_dim is required when use_text is enabled.')

            self.text_proj = nn.Linear(text_dim, cfg.d_model)
            self.text_norm = nn.LayerNorm(cfg.d_model)
            self.text_pos = PositionalEncoding(cfg.d_model)

            text_encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.text_encoder = nn.TransformerEncoder(
                text_encoder_layer,
                num_layers=n_layers,
            )

            self.text_atten = nn.Linear(cfg.d_model, 1)
            self.text_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive),
            )
            self.text_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive),
            )

        if self.use_audio and self.use_text:
            self.speech_attention = nn.MultiheadAttention(
                cfg.d_model,
                n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.text_attention = nn.MultiheadAttention(
                cfg.d_model,
                n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )

        fusion_input_dim = cfg.d_model * 2 if self.use_audio and self.use_text else cfg.d_model
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.classifier_proj = nn.Linear(fusion_input_dim, cfg.d_model)
        self.classifier_activation = nn.ReLU()
        self.classifier_dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

        self.fusion_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive),
        )

    def forward(self, raw_speech_feat, raw_text_feat, b):
        speech_feat = None
        speech_pooled = None
        text_feat = None
        text_pooled = None
        pooled_features = []

        if self.use_audio:
            if raw_speech_feat is None:
                raise ValueError('use_audio requires speech features.')
            speech_feat = self.speech_norm(self.speech_proj(raw_speech_feat))
            speech_x = self.speech_encoder(self.dropout(self.speech_pos(speech_feat)))
        else:
            speech_x = None

        if self.use_text:
            if raw_text_feat is None:
                raise ValueError('use_text requires text features.')
            text_feat = self.text_norm(self.text_proj(raw_text_feat))
            text_x = self.text_encoder(self.dropout(self.text_pos(text_feat)))
        else:
            text_x = None

        if self.use_audio and self.use_text:
            speech_attended, _ = self.speech_attention(query=speech_x, key=text_x, value=text_x)
            text_attended, _ = self.text_attention(query=text_x, key=speech_x, value=speech_x)
            speech_x = speech_x + speech_attended
            text_x = text_x + text_attended

        if self.use_audio:
            speech_pooled = attention_pool(speech_x, self.speech_atten)
            pooled_features.append(speech_pooled)

        if self.use_text:
            text_pooled = attention_pool(text_x, self.text_atten)
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

class DeepfakeAudioTextGRUModel(nn.Module):
    def __init__(self, cfg: DeepfakeBaselineModelConfig, ssl_dim: int | None, text_dim: int | None):
        super().__init__()
        self.use_audio = cfg.use_audio
        self.use_text = cfg.use_text
        if not self.use_audio and not self.use_text:
            raise ValueError('At least one deepfake input must be enabled.')
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
            speech_pooled = attention_pool(speech_x, self.speech_atten)
            pooled_features.append(speech_pooled)

        if self.use_text:
            text_pooled = attention_pool(text_x, self.text_atten)
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

class DeepfakeAudioRhythmTransformerModel(nn.Module):
    def __init__(self, cfg: DeepfakeBaselineModelConfig, ssl_dim: int | None, rhythm_dim: int | None):
        super().__init__()
        self.use_audio = cfg.use_audio
        self.use_rhythm = cfg.use_rhythm
        if not self.use_audio and not self.use_rhythm:
            raise ValueError('At least one deepfake input must be enabled.')

        self.dropout = nn.Dropout(cfg.dropout)

        n_heads = getattr(cfg, 'n_transformer_heads', 4)
        n_layers = getattr(cfg, 'n_transformer_encoder_layers', getattr(cfg, 'n_cls_encoder_layers', 2))
        ff_dim = getattr(cfg, 'transformer_ff_dim', cfg.d_model * 4)

        if cfg.d_model % n_heads != 0:
            raise ValueError(f'd_model ({cfg.d_model}) must be divisible by n_transformer_heads ({n_heads}).')

        if self.use_audio:
            if ssl_dim is None:
                raise ValueError('ssl_dim is required when use_audio is enabled.')

            self.speech_proj = nn.Linear(ssl_dim, cfg.d_model)
            self.speech_norm = nn.LayerNorm(cfg.d_model)
            self.speech_pos = PositionalEncoding(cfg.d_model)

            speech_encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.speech_encoder = nn.TransformerEncoder(
                speech_encoder_layer,
                num_layers=n_layers,
            )

            self.speech_atten = nn.Linear(cfg.d_model, 1)
            self.speech_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive),
            )
            self.speech_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive),
            )

        if self.use_rhythm:
            if rhythm_dim is None:
                raise ValueError('rhythm_dim is required when use_rhythm is enabled.')

            self.rhythm_proj = nn.Linear(rhythm_dim, cfg.d_model)
            self.rhythm_norm = nn.LayerNorm(cfg.d_model)
            self.rhythm_pos = PositionalEncoding(cfg.d_model)

            rhythm_encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.rhythm_encoder = nn.TransformerEncoder(
                rhythm_encoder_layer,
                num_layers=n_layers,
            )

            self.rhythm_atten = nn.Linear(cfg.d_model, 1)
            self.rhythm_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive),
            )
            self.rhythm_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive),
            )

        if self.use_audio and self.use_rhythm:
            self.speech_attention = nn.MultiheadAttention(
                cfg.d_model,
                n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.rhythm_attention = nn.MultiheadAttention(
                cfg.d_model,
                n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )

        fusion_input_dim = cfg.d_model * 2 if self.use_audio and self.use_rhythm else cfg.d_model
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.classifier_proj = nn.Linear(fusion_input_dim, cfg.d_model)
        self.classifier_activation = nn.ReLU()
        self.classifier_dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

        self.fusion_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive),
        )

    def forward(self, raw_speech_feat, raw_rhythm_feat, b):
        speech_feat = None
        speech_pooled = None
        rhythm_feat = None
        rhythm_pooled = None
        pooled_features = []

        if self.use_audio:
            if raw_speech_feat is None:
                raise ValueError('use_audio requires speech features.')
            speech_feat = self.speech_norm(self.speech_proj(raw_speech_feat))
            speech_x = self.speech_encoder(self.dropout(self.speech_pos(speech_feat)))
        else:
            speech_x = None

        if self.use_rhythm:
            if raw_rhythm_feat is None:
                raise ValueError('use_rhythm requires rhythm features.')
            rhythm_feat = self.rhythm_norm(self.rhythm_proj(raw_rhythm_feat))
            rhythm_x = self.rhythm_encoder(self.dropout(self.rhythm_pos(rhythm_feat)))
        else:
            rhythm_x = None

        if self.use_audio and self.use_rhythm:
            speech_attended, _ = self.speech_attention(query=speech_x, key=rhythm_x, value=rhythm_x)
            rhythm_attended, _ = self.rhythm_attention(query=rhythm_x, key=speech_x, value=speech_x)
            speech_x = speech_x + speech_attended
            rhythm_x = rhythm_x + rhythm_attended

        if self.use_audio:
            speech_pooled = attention_pool(speech_x, self.speech_atten)
            pooled_features.append(speech_pooled)

        if self.use_rhythm:
            rhythm_pooled = attention_pool(rhythm_x, self.rhythm_atten)
            pooled_features.append(rhythm_pooled)

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

            if self.use_rhythm:
                embeddings['rhythm_frame_emb'] = self.rhythm_contrastive_mlp(rhythm_feat.mean(dim=1))
                embeddings['rhythm_pooled_emb'] = self.rhythm_pooled_contrastive_mlp(rhythm_pooled)

            embeddings['fusion_emb'] = self.fusion_contrastive_mlp(fusion_emb)

            return logits, x, embeddings

        return logits, x, None
    
    
class DeepfakeAudioRhythmGRUModel(nn.Module):
    def __init__(self, cfg: DeepfakeBaselineModelConfig, ssl_dim: int | None, rhythm_dim: int | None):
        super().__init__()
        self.use_audio = cfg.use_audio
        self.use_rhythm = cfg.use_rhythm
        if not self.use_audio and not self.use_rhythm:
            raise ValueError('At least one deepfake input must be enabled.')
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
        if self.use_rhythm:
            if rhythm_dim is None:
                raise ValueError('rhythm_dim is required when use_rhythm is enabled.')
            self.rhythm_proj = nn.Linear(rhythm_dim, cfg.d_model)
            self.rhythm_norm = nn.LayerNorm(cfg.d_model)
            self.rhythm_encoder = nn.GRU(
                input_size=cfg.d_model,
                hidden_size=cfg.d_model,
                batch_first=True,
                bidirectional=True,
            )
            self.rhythm_atten = nn.Linear(cfg.d_model * 2, 1)
            self.rhythm_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(cfg.d_model // 2, cfg.d_contrastive)
            )
            self.rhythm_pooled_contrastive_mlp = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_contrastive)
            )
        if self.use_audio and self.use_rhythm:
            self.speech_attention = nn.MultiheadAttention(cfg.d_model * 2, 1, dropout=cfg.dropout, batch_first=True)
            self.rhythm_attention = nn.MultiheadAttention(cfg.d_model * 2, 1, dropout=cfg.dropout, batch_first=True)
        fusion_input_dim = cfg.d_model * 4 if self.use_audio and self.use_rhythm else cfg.d_model * 2
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
    def forward(self, raw_speech_feat, raw_rhythm_feat, b):
        """Encode available modalities, optionally cross-attend, and classify."""
        speech_feat = None
        speech_pooled = None
        rhythm_feat = None
        rhythm_pooled = None
        pooled_features = []

        if self.use_audio:
            if raw_speech_feat is None:
                raise ValueError('use_audio requires speech features.')
            speech_feat = self.speech_norm(self.speech_proj(raw_speech_feat))
            speech_x, _ = self.speech_encoder(speech_feat)
        else:
            speech_x = None

        if self.use_rhythm:
            if raw_rhythm_feat is None:
                raise ValueError('use_rhythm requires rhythm features.')
            rhythm_feat = self.rhythm_norm(self.rhythm_proj(raw_rhythm_feat))
            rhythm_x, _ = self.rhythm_encoder(rhythm_feat)
        else:
            rhythm_x = None

        if self.use_audio and self.use_rhythm:
            speech_attended, _ = self.speech_attention(query=speech_x, key=rhythm_x, value=rhythm_x)
            rhythm_attended, _ = self.rhythm_attention(query=rhythm_x, key=speech_x, value=speech_x)
            speech_x = speech_x + speech_attended
            rhythm_x = rhythm_x + rhythm_attended

        if self.use_audio:
            speech_pooled = attention_pool(speech_x, self.speech_atten)
            pooled_features.append(speech_pooled)

        if self.use_rhythm:
            rhythm_pooled = attention_pool(rhythm_x, self.rhythm_atten)
            pooled_features.append(rhythm_pooled)

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
            if self.use_rhythm:
                embeddings['rhythm_frame_emb'] = self.rhythm_contrastive_mlp(rhythm_feat.mean(dim=1))
                embeddings['rhythm_pooled_emb'] = self.rhythm_pooled_contrastive_mlp(rhythm_pooled)
            embeddings['fusion_emb'] = self.fusion_contrastive_mlp(fusion_emb)

            return logits, x, embeddings
        return logits, x, None
    