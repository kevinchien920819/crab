import torch
import torch.nn as nn
from torchaudio import pipelines
from torchaudio.pipelines import Wav2Vec2Bundle
from transformers import AutoModel, WavLMModel

from data.dataclass import Batch
from .embedding import PositionalEncoding


class SSLModel(nn.Module):
    def __init__(self, cfg):
        """Initialize the speech SSL encoder from torchaudio or Hugging Face backends."""
        super().__init__()
        bundle: Wav2Vec2Bundle = getattr(pipelines, cfg.ssl_model_str)
        self.is_wavlm = 'WAVLM' in cfg.ssl_model_str.upper()
        model_map = {
            "WAVLM_BASE": "microsoft/wavlm-base",
            "WAVLM_LARGE": "microsoft/wavlm-large",
            "WAV2VEC2_BASE": "facebook/wav2vec2-base",
            "WAV2VEC2_LARGE": "facebook/wav2vec2-large",
        }

        hf_model_name = model_map.get(cfg.ssl_model_str, cfg.ssl_model_str)

        if self.is_wavlm:
            self.model = WavLMModel.from_pretrained(hf_model_name)
            self.ssl_bundle = bundle
            self.backend = 'huggingface'

        if self.use_hf_backend:
            hf_model_name = self.HF_MODEL_MAP.get(self.ssl_model_key, cfg.ssl_model_str)
            self.model = AutoModel.from_pretrained(hf_model_name)
            self.output_dim = self.model.config.hidden_size
        else:
            self.model = bundle.get_model()
            self.ssl_bundle = bundle
            self.backend = 'torchaudio'

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
        if self.backend == 'torchaudio':
            ssl_feat, feat_length = self.model(waveform, length)
            if feat_length is None:
                feat_length = torch.full(
                    size=(waveform.size(0),),
                    fill_value=ssl_feat.size(1),
                    dtype=torch.long,
                    device=waveform.device,
                )
            return ssl_feat, feat_length

        attention_mask = self._length_to_attention_mask(waveform, length)
        if self.backend == 'huggingface':
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
    def __init__(self, cfg):
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


class RhythmEncoder(nn.Module):
    """Encode duration, Devil, and nPVI interval features into rhythm tokens."""

    def __init__(self, cfg):
        super().__init__()
        self.sources = list(getattr(cfg, 'rhythm_sources', ['word', 'vowel', 'consonant']))
        if not self.sources:
            raise ValueError('rhythm_sources must contain at least one interval source.')

        self.feature_proj = nn.Linear(3, cfg.d_model)
        self.source_embedding = nn.Embedding(len(self.sources), cfg.d_model)
        self.positional_encoding = PositionalEncoding(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.n_rhythm_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.n_rhythm_encoder_layers,
            )
        else:
            self.encoder = nn.Identity()

    def _required_feature(
        self,
        value: torch.Tensor | None,
        source: str,
        field_name: str,
        reference: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(f'Rhythm source "{source}" requires precomputed {field_name}.')
        value = value.to(device=reference.device, dtype=reference.dtype)
        return value.masked_fill(~mask, 0.0)

    def _source_features(
        self,
        b: Batch,
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        duration = getattr(b, f'{source}_d', None)
        if duration is None or duration.numel() == 0:
            return None

        duration = duration.float()
        mask = duration >= 0
        if duration.size(1) == 0:
            return None

        clean_duration = duration.masked_fill(~mask, 0.0)
        devil = self._required_feature(
            getattr(b, f'{source}_devi', None),
            source,
            f'{source}_devi',
            clean_duration,
            mask,
        )
        npvi = self._required_feature(
            getattr(b, f'{source}_mu_diff', None),
            source,
            f'{source}_mu_diff',
            clean_duration,
            mask,
        )
        features = torch.stack([clean_duration, devil, npvi], dim=-1)
        return features, mask

    def forward(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        masks = []
        for source_idx, source in enumerate(self.sources):
            source_data = self._source_features(b, source)
            if source_data is None:
                continue
            features, mask = source_data
            x = self.feature_proj(features)
            source_ids = torch.full(
                (x.size(0), x.size(1)),
                source_idx,
                device=x.device,
                dtype=torch.long,
            )
            x = x + self.source_embedding(source_ids)
            x = self.positional_encoding(x)
            tokens.append(x)
            masks.append(mask)

        if not tokens:
            raise ValueError(
                "Rhythm modality requires duration features. "
                "Set use_duration: true for the ASVspoof dataset config and provide duration CSV files."
            )

        x = torch.cat(tokens, dim=1)
        mask = torch.cat(masks, dim=1)
        if not mask.any(dim=1).all():
            raise ValueError('Every sample needs at least one valid rhythm interval.')

        x = self.dropout(self.norm(x))
        if isinstance(self.encoder, nn.Identity):
            x = self.encoder(x)
        else:
            x = self.encoder(x, src_key_padding_mask=~mask)
        return x, mask
