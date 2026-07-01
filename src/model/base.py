import torch
import torch.nn as nn
from torch import Tensor
from torchaudio import pipelines
from torchaudio.pipelines import Wav2Vec2Bundle
from transformers import AutoModel, WavLMModel

from config.deepfake.baseline import DeepfakeBaselineModelConfig
from data.dataclass import Batch
from .embedding import RhythmEmbedding

class SSLModel(nn.Module):
    HF_MODEL_MAP = {
        "WAVLM_BASE": "microsoft/wavlm-base",
        "WAVLM_LARGE": "microsoft/wavlm-large",
        "WAV2VEC2_BASE": "facebook/wav2vec2-base",
        "WAV2VEC2_LARGE": "facebook/wav2vec2-large",
    }

    def __init__(self, cfg):
        """Initialize the speech SSL encoder from torchaudio or Hugging Face backends."""
        super().__init__()
        self.ssl_model_key = cfg.ssl_model_str
        self.ssl_bundle: Wav2Vec2Bundle | None = getattr(pipelines, self.ssl_model_key, None)
        self.is_wavlm = 'WAVLM' in self.ssl_model_key.upper()
        self.use_hf_backend = self.is_wavlm or self.ssl_bundle is None

        if self.use_hf_backend:
            hf_model_name = self.HF_MODEL_MAP.get(self.ssl_model_key, self.ssl_model_key)
            self.model = WavLMModel.from_pretrained(hf_model_name) if self.is_wavlm else AutoModel.from_pretrained(hf_model_name)
            self.output_dim = self.model.config.hidden_size
            self.backend = 'huggingface'
        else:
            self.model = self.ssl_bundle.get_model()
            self.output_dim = self.ssl_bundle._params['encoder_embed_dim']
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
    """Encode configured duration-level rhythm features into rhythm tokens."""

    FEATURE_SUFFIXES = ('d', 'devi', 'mu_diff')

    def __init__(self, cfg: DeepfakeBaselineModelConfig):
        super().__init__()
        self.sources = list(getattr(cfg, 'rhythm_sources', ['syllable', 'vowel', 'consonant']))
        if not self.sources:
            raise ValueError('rhythm_sources must contain at least one interval source.')

        rhythm_input_dim = len(self.sources) * len(self.FEATURE_SUFFIXES)
        self.rhythm_embedding = RhythmEmbedding(rhythm_input_dim, cfg.d_model, cfg.dropout)
        if cfg.n_rhythm_encoder_layers < 0:
            raise ValueError('n_rhythm_encoder_layers must be non-negative.')

        if cfg.n_rhythm_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.n_rhythm_encoder_layers,
            )
        else:
            self.encoder = nn.Identity()

    def _source_features(
        self,
        b: Batch,
        source: str,
    ) -> tuple[list[Tensor], Tensor] | None:
        field_names = [f'{source}_{suffix}' for suffix in self.FEATURE_SUFFIXES]
        duration = getattr(b, field_names[0], None)
        if duration is None or duration.numel() == 0:
            return None
        if duration.dim() != 2:
            raise ValueError(
                f'Rhythm source {source!r} expects {field_names[0]} to be a [B, T] tensor, '
                f'got shape {tuple(duration.shape)}.'
            )
        if duration.size(1) == 0:
            return None

        duration = duration.float()
        # Padding is marked on source_d only; devi/mu_diff can legitimately be 0.
        mask = duration != -1.0

        features: list[Tensor] = []
        for field_name in field_names:
            value = getattr(b, field_name, None)
            if value is None:
                raise ValueError(f'Rhythm source {source!r} requires precomputed {field_name}.')

            value = value.to(device=duration.device, dtype=duration.dtype)
            if value.shape != duration.shape:
                raise ValueError(
                    f'Rhythm source {source!r} has mismatched {field_name} shape: '
                    f'{tuple(value.shape)} != {tuple(duration.shape)}.'
                )
            features.append(value.masked_fill(torch.logical_not(mask), 0.0))

        return features, mask

    def forward(self, b: Batch, source: str | None = None) -> tuple[Tensor, Tensor]:
        features = []
        masks = []
        sources = [source] if source is not None else self.sources
        for rhythm_source in sources:
            source_data = self._source_features(b, rhythm_source)
            if source_data is None:
                continue
            source_features, mask = source_data
            if masks and mask.shape != masks[0].shape:
                raise ValueError(
                    f'Rhythm source {rhythm_source!r} mask shape {tuple(mask.shape)} '
                    f'does not match aligned rhythm mask shape {tuple(masks[0].shape)}.'
                )
            features.extend(source_features)
            masks.append(mask)

        if not features:
            raise ValueError(
                'Rhythm modality requires duration features. '
                'Set use_duration: true for the ASVspoof dataset config and provide duration CSV files.'
            )

        mask = torch.stack(masks, dim=0).all(dim=0)
        if not mask.any(dim=1).all():
            raise ValueError('Every sample needs at least one valid rhythm interval.')

        x = self.rhythm_embedding(*features)
        if isinstance(self.encoder, nn.Identity):
            x = self.encoder(x)
        else:
            x = self.encoder(x, src_key_padding_mask=torch.logical_not(mask))
        return x, mask
