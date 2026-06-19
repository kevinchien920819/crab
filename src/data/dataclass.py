from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class Duration:
    st: Any     # start time in seconds
    et: Any     # end time in seconds
    sid: Any    # start frame index
    eid: Any    # end frame index
    d: Any      # duration in seconds
    devi: Optional[Any] = None
    mu_diff: Optional[Any] = None


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
    syllable_data: Optional[Duration] = None
    vowel_data : Optional[Duration] = None
    consonant_data: Optional[Duration] = None
    asvspoof5_cache: Optional["ASVspoof5Cache"] = None

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
    wavform: torch.Tensor                               # [B, T]
    audio_mask: Optional[torch.Tensor] = None           # [B, T]
    # for text model
    sentences: Optional[list[str]] = None
    tokens: Optional[torch.Tensor] = None
    text_mask: Optional[torch.Tensor] = None            # [B, L]
    # for emotion detection
    emotion_labels: Optional[torch.Tensor] = None       # [B]
    sentiment_labels: Optional[torch.Tensor] = None     # [B]
    # for deepfake detection
    deepfake_labels: Optional[torch.Tensor] = None      # [B]

    # duration in seconds [B, N], (start - end)
    word_d: Optional[torch.Tensor] = None               # [B, N]
    word_devi: Optional[torch.Tensor] = None            # [B, N]
    word_mu_diff: Optional[torch.Tensor] = None         # [B, N]

    syllable_d: Optional[torch.Tensor] = None           # [B, N]
    syllable_devi: Optional[torch.Tensor] = None        # [B, N]
    syllable_mu_diff: Optional[torch.Tensor] = None     # [B, N]

    vowel_d: Optional[torch.Tensor] = None              # [B, N]
    vowel_devi: Optional[torch.Tensor] = None           # [B, N]
    vowel_mu_diff: Optional[torch.Tensor] = None        # [B, N]

    consonant_d: Optional[torch.Tensor] = None          # [B, N]
    consonant_devi: Optional[torch.Tensor] = None       # [B, N]
    consonant_mu_diff: Optional[torch.Tensor] = None    # [B, N]


    # reserve for Mamba interface
    utt_sid: Optional[torch.Tensor] = None
    word_sid: Optional[torch.Tensor] = None
    syllable_sid: Optional[torch.Tensor] = None
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


@dataclass
class ASVspoof5Cache:
    speaker_id: str
    flac_file_name: str
    speaker_gender: str
    codec: str
    codec_q: str
    codec_seed: str
    attack_tag: str
    attack_label: str
    label: int

    content_sentence: str
    starttime_sentence: float
    endtime_sentence: float
    duration_sentence: float

    content_word: str
    word_count: int
    starttime_word: list[float]
    endtime_word: list[float]
    duration_word: list[float]

    content_syllable: str
    syllable_count: int
    starttime_syllable: list[float]
    endtime_syllable: list[float]
    duration_syllable: list[float]

    content_phoneme: str
    phoneme_count: int
    starttime_phoneme: list[float]
    endtime_phoneme: list[float]
    duration_phoneme: list[float]

    content_vowel: str
    vowel_count: int
    starttime_vowel: list[float]
    endtime_vowel: list[float]
    duration_vowel: list[float]

    starttime_consonant: list[float]
    endtime_consonant: list[float]
    duration_consonant: list[float]

    devi_mu_syllable: list[float]
    mu_diff_syllable: list[float]
    nPVI_syllable: float

    devi_mu_vowel: list[float]
    mu_diff_vowel: list[float]
    nPVI_vowel: float

    devi_mu_consonant: list[float]
    mu_diff_consonant: list[float]
    nPVI_consonant: float

    filepath: str
