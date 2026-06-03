from dataclasses import dataclass, field
@dataclass(frozen=True)
class SplitConfig:
    protocols: str
    audio_dir: str
    output: str

# ASVspoof5 dataclass for model input
@dataclass
class DatasetItem:
    speaker_id: str
    flac_file_name: str
    speaker_gender: str
    codec: str
    codec_q: str
    codec_seed: str
    attack_tag: str
    attack_label: str
    label: int
    num_word: int

    content_sentence: str
    starttime_sentence: list[float]
    endtime_sentence: list[float]
    duration_sentence: float

    content_syllable: str
    starttime_syllable: list[float]
    endtime_syllable: list[float]

    starttime_word: list[float]
    endtime_word: list[float]
    duration_word: list[float]

    vowel_count: int
    vowel_content: str
    starttime_vowel: list[float]
    endtime_vowel: list[float]
    duration_vowel: list[float]

    constanant_count: int
    constanant_content: str
    starttime_constanant: list[float]
    endtime_constanant: list[float]
    duration_constanant: list[float]

    devi_mu_syllable: float
    mu_diff_syllable: float

    devi_mu_vowel: float
    mu_diff_vowel: float

    devi_mu_constanant: float
    mu_diff_constanant: float

    filepath: str
