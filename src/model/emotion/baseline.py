import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchaudio import pipelines
from torchaudio.models import Wav2Vec2Model
from torchaudio.pipelines import Wav2Vec2Bundle
from transformers import AutoModel, AutoTokenizer

from dataclasses import dataclass
from typing import Optional
from config.emotion.baseline import EmotionBaselineModelConfig
from data.dataclass import Batch



class SSLModel(nn.Module):
    def __init__(self, cfg: EmotionBaselineModelConfig):
        super().__init__()
        bundle: Wav2Vec2Bundle = getattr(pipelines, cfg.ssl_model_str)
        self.model = bundle.get_model()
        self.ssl_bundle = bundle
        self.is_wavlm = 'WAVLM' in cfg.ssl_model_str.upper()

    def forward(self, wavform, length):
        # WavLM in torchaudio does not support attention_mask (via length)
        # In batch_size=1, length is not needed.
        if self.is_wavlm:
            ssl_feat, _ = self.model(wavform, None)
        else:
            try:
                ssl_feat, _ = self.model(wavform, length)
            except AssertionError:
                ssl_feat, _ = self.model(wavform, None)
        return ssl_feat

class TextModel(nn.Module):
    def __init__(self, cfg: EmotionBaselineModelConfig):
        super().__init__()
        mapping = {
            'ROBERTA_BASE_ENCODER': 'roberta-base',
            'ROBERTA_LARGE_ENCODER': 'roberta-large',
        }
        model_id = mapping.get(cfg.text_model_str, cfg.text_model_str)
        self.model = AutoModel.from_pretrained(model_id)
        if cfg.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, tokens, text_mask):
        text_outputs = self.model(input_ids=tokens, attention_mask=text_mask.long() if text_mask is not None else None)
        feat = text_outputs.last_hidden_state
        return feat

class SERModel(nn.Module):
    def __init__(self, cfg: EmotionBaselineModelConfig, ssl_dim: int, text_dim: int):
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
        
        self.fusion_norm = nn.LayerNorm(cfg.d_model * 4)
        self.classifier_proj = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.classifier_activation = nn.ReLU()
        self.classifier_dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

    def attention_pool(self, features, attention_layer, mask=None):
        attn_weights = attention_layer(features)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (features * attn_weights).sum(dim=1)
        return pooled

    def forward(self, raw_speech_feat, raw_text_feat, b):
        # Apply projections and norms first
        speech_feat = self.speech_norm(self.speech_proj(raw_speech_feat))
        text_feat = self.text_norm(self.text_proj(raw_text_feat))

        # Calculate valid lengths for GRU
        speech_key_padding_mask = ~b.audio_mask if b.audio_mask is not None else None
        if speech_key_padding_mask is not None and speech_key_padding_mask.size(1) != speech_feat.size(1):
            speech_key_padding_mask = F.interpolate(speech_key_padding_mask.float().unsqueeze(1), size=speech_feat.size(1), mode='nearest').squeeze(1).bool()
        
        speech_lens = (~speech_key_padding_mask).sum(dim=1).clamp(min=1).cpu() if speech_key_padding_mask is not None else torch.full((speech_feat.size(0),), speech_feat.size(1))
        text_lens = b.text_mask.sum(dim=1).clamp(min=1).cpu() if b.text_mask is not None else torch.full((text_feat.size(0),), text_feat.size(1))

        # Pack, GRU, then Unpack
        packed_speech = pack_padded_sequence(speech_feat, speech_lens, batch_first=True, enforce_sorted=False)
        speech_x, _ = self.speech_encoder(packed_speech)
        speech_x, _ = pad_packed_sequence(speech_x, batch_first=True, total_length=speech_feat.size(1))

        packed_text = pack_padded_sequence(text_feat, text_lens, batch_first=True, enforce_sorted=False)
        text_x, _ = self.text_encoder(packed_text)
        text_x, _ = pad_packed_sequence(text_x, batch_first=True, total_length=text_feat.size(1))
        
        # MultiheadAttention
        text_key_padding_mask = ~b.text_mask if b.text_mask is not None else None
        speech_attended, _ = self.speech_attention(query=speech_x, key=text_x, value=text_x, key_padding_mask=text_key_padding_mask)
        text_attended, _ = self.text_attention(query=text_x, key=speech_x, value=speech_x, key_padding_mask=speech_key_padding_mask)

        speech_final = speech_x + speech_attended
        text_final = text_x + text_attended
        
        # Attention Pooling
        speech_pool_mask = b.audio_mask
        if speech_pool_mask is not None and speech_pool_mask.size(1) != speech_final.size(1):
             speech_pool_mask = F.interpolate(speech_pool_mask.float().unsqueeze(1), size=speech_final.size(1), mode='nearest').squeeze(1).bool()

        speech_pooled = self.attention_pool(speech_final, self.speech_atten, mask=speech_pool_mask)
        text_pooled = self.attention_pool(text_final, self.text_atten, mask=b.text_mask)
        
        # Fusion and Classifier
        final_feature = torch.cat([speech_pooled, text_pooled], dim=-1)
        final_feature_norm = self.fusion_norm(final_feature)
        x = self.classifier_proj(final_feature_norm)
        x = self.classifier_activation(x)
        x = self.classifier_dropout(x)
        logits = self.classifier(x)
        
        return logits, x

