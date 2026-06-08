import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchaudio import pipelines
from transformers import AutoModel, AutoTokenizer

from dataclasses import dataclass
from typing import Optional
from config.emotion.baseline import EmotionBaselineModelConfig
from data.dataclass import Batch



class SSLModel(nn.Module):
    HF_MODEL_MAP = {
        "WAVLM_BASE": "microsoft/wavlm-base",
        "WAVLM_BASE_PLUS": "microsoft/wavlm-base-plus",
        "WAVLM_LARGE": "microsoft/wavlm-large",
        "WAV2VEC2_BASE": "facebook/wav2vec2-base",
        "WAV2VEC2_LARGE": "facebook/wav2vec2-large",
        "WAV2VEC2_LARGE_LV60K": "facebook/wav2vec2-large-lv60",
        "WAV2VEC2_XLSR53": "facebook/wav2vec2-large-xlsr-53",
        "WAV2VEC2_XLSR_300M": "facebook/wav2vec2-xls-r-300m",
        "WAV2VEC2_XLSR_1B": "facebook/wav2vec2-xls-r-1b",
        "WAV2VEC2_XLSR_2B": "facebook/wav2vec2-xls-r-2b",
        "HUBERT_BASE": "facebook/hubert-base-ls960",
        "HUBERT_LARGE": "facebook/hubert-large-ll60k",
        "HUBERT_XLARGE": "facebook/hubert-xlarge-ll60k",
        "HUBERT_ASR_LARGE": "facebook/hubert-large-ls960-ft",
        "HUBERT_ASR_XLARGE": "facebook/hubert-xlarge-ls960-ft",
    }

    def __init__(self, cfg: EmotionBaselineModelConfig):
        """Initialize the speech SSL encoder from torchaudio or Hugging Face backends."""
        super().__init__()
        self.ssl_model_str = cfg.ssl_model_str
        self.ssl_model_key = cfg.ssl_model_str.upper()
        self.ssl_bundle = getattr(pipelines, self.ssl_model_key, None)
        self.use_hf_backend = self.ssl_bundle is None or 'WAVLM' in self.ssl_model_key

        if self.use_hf_backend:
            hf_model_name = self.HF_MODEL_MAP.get(self.ssl_model_key, cfg.ssl_model_str)
            self.model = AutoModel.from_pretrained(hf_model_name)
            self.output_dim = self.model.config.hidden_size
        else:
            self.model = self.ssl_bundle.get_model()
            self.output_dim = self.ssl_bundle._params['encoder_embed_dim']

    def _length_to_attention_mask(self, waveform, length):
        """
        waveform: Tensor, shape [B, T]
        length: Tensor, shape [B], raw waveform valid lengths before padding
        """
        if length is None:
            return None

        batch_size, max_len = waveform.shape
        device = waveform.device

        attention_mask = (
            torch.arange(max_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, max_len)
            < length.unsqueeze(1)
        )

        return attention_mask.long()

    def _hf_feat_lengths(self, length, fallback_size, device):
        if length is None:
            return torch.full(
                size=(fallback_size[0],),
                fill_value=fallback_size[1],
                dtype=torch.long,
                device=device,
            )

        if hasattr(self.model, "_get_feat_extract_output_lengths"):
            return self.model._get_feat_extract_output_lengths(length)

        return torch.full(
            size=(length.size(0),),
            fill_value=fallback_size[1],
            dtype=torch.long,
            device=device,
        )

    def forward(self, waveform, length=None):
        """
        waveform: [B, T]
        length: [B], waveform-level valid length, not feature-level length
        return:
            ssl_feat: [B, T_feat, hidden_dim]
            feat_length: [B]
        """
        if self.use_hf_backend:
            attention_mask = self._length_to_attention_mask(waveform, length)

            outputs = self.model(
                input_values=waveform,
                attention_mask=attention_mask,
                return_dict=True,
            )

            ssl_feat = outputs.last_hidden_state
            feat_length = self._hf_feat_lengths(
                length,
                fallback_size=(waveform.size(0), ssl_feat.size(1)),
                device=waveform.device,
            )
            return ssl_feat, feat_length

        ssl_feat, feat_length = self.model(waveform, length)
        if feat_length is None:
            feat_length = torch.full(
                size=(waveform.size(0),),
                fill_value=ssl_feat.size(1),
                dtype=torch.long,
                device=waveform.device,
            )
        return ssl_feat, feat_length

class TextModel(nn.Module):
    def __init__(self, cfg: EmotionBaselineModelConfig):
        """Initialize the transformer text encoder from a configured model name."""
        super().__init__()
        mapping = {
            'ROBERTA_BASE_ENCODER': 'roberta-base',
            'ROBERTA_LARGE_ENCODER': 'roberta-large',
        }
        model_id = mapping.get(cfg.text_model_str, cfg.text_model_str)
        self.model = AutoModel.from_pretrained(model_id)

    def forward(self, tokens, text_mask):
        """Encode token IDs and attention masks into contextual text features."""
        text_outputs = self.model(input_ids=tokens, attention_mask=text_mask.long() if text_mask is not None else None)
        feat = text_outputs.last_hidden_state
        return feat

class SERModel(nn.Module):
    def __init__(self, cfg: EmotionBaselineModelConfig, ssl_dim: int, text_dim: int):
        """Initialize multimodal projection, recurrent encoding, attention, and classifier layers."""
        super().__init__()
        # Projections and Norms moved here from sub-models
        self.speech_proj = nn.Linear(ssl_dim, cfg.d_model)
        self.text_proj = nn.Linear(text_dim, cfg.d_model)
        self.speech_norm = nn.LayerNorm(cfg.d_model)
        self.text_norm = nn.LayerNorm(cfg.d_model)

        self.dropout = nn.Dropout(cfg.dropout)

        self.speech_encoder = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            batch_first=True,
            bidirectional=True,
        )
        self.text_encoder = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            batch_first=True,
            bidirectional=True,
        )
        self.speech_attention = nn.MultiheadAttention(cfg.d_model * 2, 1, dropout=cfg.dropout, batch_first=True)
        self.text_attention = nn.MultiheadAttention(cfg.d_model * 2, 1, dropout=cfg.dropout, batch_first=True)

        self.speech_atten = nn.Linear(cfg.d_model * 2, 1)
        self.text_atten = nn.Linear(cfg.d_model * 2, 1)

        # Contrastive embedding MLPs for frame-level features
        self.speech_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, cfg.d_contrastive)
        )

        self.text_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, cfg.d_contrastive)
        )

        # Contrastive embedding MLPs for pooled features
        self.speech_pooled_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive)
        )

        self.text_pooled_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive)
        )

        # Fusion contrastive embedding MLP
        self.fusion_contrastive_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_contrastive)
        )


        self.fusion_norm = nn.LayerNorm(cfg.d_model * 4)
        self.classifier_proj = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.classifier_activation = nn.ReLU()
        self.classifier_dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

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
        # Apply projections and norms first
        """Fuse speech and text features and return logits, features, and optional embeddings."""
        speech_feat = self.speech_norm(self.speech_proj(raw_speech_feat)) # [batch, seq_len, d_model]
        text_feat = self.text_norm(self.text_proj(raw_text_feat)) # [batch, seq_len, d_model]

        # Calculate valid lengths for GRU
        # speech_key_padding_mask = ~b.audio_mask if b.audio_mask is not None else None
        # if speech_key_padding_mask is not None and speech_key_padding_mask.size(1) != speech_feat.size(1):
        #     speech_key_padding_mask = F.interpolate(speech_key_padding_mask.float().unsqueeze(1), size=speech_feat.size(1), mode='nearest').squeeze(1).bool()

        # speech_lens = (~speech_key_padding_mask).sum(dim=1).clamp(min=1).cpu() if speech_key_padding_mask is not None else torch.full((speech_feat.size(0),), speech_feat.size(1))
        # text_lens = b.text_mask.sum(dim=1).clamp(min=1).cpu() if b.text_mask is not None else torch.full((text_feat.size(0),), text_feat.size(1))

        # Pack, GRU, then Unpack
        # packed_speech = pack_padded_sequence(speech_feat, speech_lens, batch_first=True, enforce_sorted=False)
        speech_x, _ = self.speech_encoder(speech_feat)
        # speech_x, _ = pad_packed_sequence(speech_x, batch_first=True, total_length=speech_feat.size(1))

        # packed_text = pack_padded_sequence(text_feat, text_lens, batch_first=True, enforce_sorted=False)
        text_x, _ = self.text_encoder(text_feat)
        # text_x, _ = pad_packed_sequence(text_x, batch_first=True, total_length=text_feat.size(1))

        # MultiheadAttention
        # text_key_padding_mask = ~b.text_mask if b.text_mask is not None else None
        speech_attended, _ = self.speech_attention(query=speech_x, key=text_x, value=text_x)
        text_attended, _ = self.text_attention(query=text_x, key=speech_x, value=speech_x)

        speech_final = speech_x + speech_attended
        text_final = text_x + text_attended

        # Attention Pooling
        # speech_pool_mask = b.audio_mask
        # if speech_pool_mask is not None and speech_pool_mask.size(1) != speech_final.size(1):
        #      speech_pool_mask = F.interpolate(speech_pool_mask.float().unsqueeze(1), size=speech_final.size(1), mode='nearest').squeeze(1).bool()

        speech_pooled = self.attention_pool(speech_final, self.speech_atten)
        text_pooled = self.attention_pool(text_final, self.text_atten)

        # Fusion and Classifier
        fusion_emb = torch.cat([speech_pooled, text_pooled], dim=-1)
        fusion_emb = self.fusion_norm(fusion_emb)
        fusion_emb = self.classifier_proj(fusion_emb)
        x = self.classifier_activation(fusion_emb)
        x = self.classifier_dropout(x)
        logits = self.classifier(x)

        if self.training:
            # Compute contrastive embeddings
            speech_frame_mean = speech_feat.mean(dim=1)
            text_frame_mean = text_feat.mean(dim=1)
            speech_contrastive_emb = self.speech_contrastive_mlp(speech_frame_mean)
            text_contrastive_emb = self.text_contrastive_mlp(text_frame_mean)
            speech_pooled_contrastive_emb = self.speech_pooled_contrastive_mlp(speech_pooled)
            text_pooled_contrastive_emb = self.text_pooled_contrastive_mlp(text_pooled)
            fusion_contrastive_emb = self.fusion_contrastive_mlp(fusion_emb)

            embeddings = {
                'speech_frame_emb': speech_contrastive_emb,
                'text_frame_emb': text_contrastive_emb,
                'speech_pooled_emb': speech_pooled_contrastive_emb,
                'text_pooled_emb': text_pooled_contrastive_emb,
                'fusion_emb': fusion_contrastive_emb
            }

            return logits, x, embeddings
        return logits, x, None
