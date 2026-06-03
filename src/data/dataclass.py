from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Duration:
    st: float   # start time in seconds
    et: float   # end time in seconds
    sid: int    # start frame index
    eid: int    # end frame index
    d: float    # duration in seconds


@dataclass
class Sample:
    filename: str
    path: str = None
    length: int = 0
    wavform: Optional[torch.Tensor] = None

    # for deepfake detection
    deepfake_label: Optional[int] = None  # 0: bonafide, 1: spoof
    utt_data : Optional[Duration] = None
    word_data: Optional[Duration] = None
    vowel_data : Optional[Duration] = None
    consonant_data: Optional[Duration] = None

    # for text model
    sentence: Optional[str] = None
    tokens: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    # for emotion detection
    emotion_label: Optional[int] = None
    sentiment_label: Optional[int] = None
    dataset_name: Optional[str] = None  # "IEMOCAP", "MELD", or "MSP-Podcast"

@dataclass
class Batch:
    filenames: list[str]
    path: list[str]
    length: torch.Tensor
    wavform: torch.Tensor  # [B, T]
    audio_mask: Optional[torch.Tensor] = None  # [B, T]
    # for text model
    sentences: Optional[list[str]] = None
    tokens: Optional[torch.Tensor] = None
    text_mask: Optional[torch.Tensor] = None  # [B, L]
    # for emotion detection
    emotion_labels: Optional[torch.Tensor] = None  # [B]
    sentiment_labels: Optional[torch.Tensor] = None  # [B]
    # for deepfake detection
    deepfake_labels: Optional[torch.Tensor] = None  # [B]

    # duration in seconds [B, N], (start - end)
    # utt_d: Optional[torch.Tensor] = None
    word_d: Optional[torch.Tensor] = None
    vowel_d: Optional[torch.Tensor] = None
    vowel_devi: Optional[torch.Tensor] = None
    vowel_mu_diff: Optional[torch.Tensor] = None

    consonant_d: Optional[torch.Tensor] = None
    consonant_devi: Optional[torch.Tensor] = None
    consonant_mu_diff: Optional[torch.Tensor] = None

    # utt_sid: Optional[torch.Tensor] = None
    word_sid: Optional[torch.Tensor] = None
    vowel_sid: Optional[torch.Tensor] = None
    consonant_sid: Optional[torch.Tensor] = None

    seq_idx: Optional[torch.Tensor] = None
    sid: Optional[torch.Tensor] = None
    eid: Optional[torch.Tensor] = None

    def to(self, device, non_blocking: bool = False):
        """Move tensor fields in the batch to a target device."""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value is not None:
                setattr(self, key, value.to(device, non_blocking=non_blocking))
        return self

    # n_sample
    def __len__(self):
        """Return the number of samples represented by the batch."""
        return len(self.path)
