from dataclasses import dataclass, field
@dataclass(frozen=True)
class SplitConfig:
    protocols: str
    audio_dir: str
    output: str

# ASVspoof5 dataclass for model input
@dataclass
class DatasetItem:
    # ASVspoof5 metadata 
    speaker_id: str
    flac_file_name: str
    speaker_gender: str
    codec: str
    codec_q: str
    codec_seed: str
    attack_tag: str
    attack_label: str
    label: int
    
    # content of data sentence EX: "Hello world"
    content_sentence: str
    # start time, end time, duration of sentence
    starttime_sentence: list[float]
    endtime_sentence: list[float]
    duration_sentence: float
    
    # content of word EX: "Hello,world"
    content_word: str
    # number of word in sentence
    word_count: int
    # start time, end time, duration of each word
    starttime_word: list[float]
    endtime_word: list[float]
    duration_word: list[float]
    
    # content of syllable EX: "HH AH0, L OW1 W, ER1 L D"
    content_syllable: str
    # number of syllable in sentence
    syllable_count: int
    # start time, end time, duration of each syllable
    starttime_syllable: list[float]
    endtime_syllable: list[float]
    duration_syllable: list[float]

    # content of phoneme EX: "HH,AH0,L,OW1,W,ER1,L,D"
    content_phoneme: str
    # number of phoneme in sentence
    phoneme_count: int
    # start time, end time, duration of each phoneme
    starttime_phoneme: list[float]
    endtime_phoneme: list[float]
    duration_phoneme: list[float]

    # content of vowel EX: "AH0, OW1, ER1"
    content_vowel: str
    vowel_count: int
    # start time, end time, duration of each vowel in one syllable
    starttime_vowel: list[float]
    endtime_vowel: list[float]
    # sum of duration of vowel in one syllable
    duration_vowel: list[float]
     
     # start time, end time, duration of each vowel in one syllable
    starttime_consonant: list[float]
    endtime_consonant: list[float]
    # duration time in each syllable - vowel = duration of consonant
    duration_consonant: list[float]

    # Rhythm features from duration intervals.
    # Paper notation:
    #   Int_i = duration of the i-th interval
    #   μ_Int = mean duration of the same interval type within this utterance/sentence

    # Deviation from utterance-level mean:
    #   Devil_i = Int_i - μ_Int
    #
    # For this DatasetItem:
    #   devi_mu_syllable:
    #       Int_i = duration_syllable[i]
    #   devi_mu_vowel:
    #       Int_i = duration_vowel[i]
    #       duration_vowel[i] should be the total vowel duration inside syllable i
    #   devi_mu_consonant:
    #       Int_i = duration_consonant[i]
    #       duration_consonant[i] should be:
    #           duration_syllable[i] - duration_vowel[i]

    # Normalized pairwise duration difference:
    #   μDiff_i = (Int_i - Int_{i+1}) / ((Int_i + Int_{i+1}) / 2)
    #
    # For the last interval:
    #   μDiff_last = nPVI-Int
    #
    # where:
    #   nPVI-Int = mean(
    #       abs((Int_k - Int_{k+1}) / ((Int_k + Int_{k+1}) / 2))
    #   )
    #   for k = 0 ... n-2
    #
    # For this DatasetItem:
    #   mu_diff_syllable:
    #       computed from adjacent duration_syllable values
    #   mu_diff_vowel:
    #       computed from adjacent duration_vowel values
    #       each duration_vowel[i] is the total vowel duration inside syllable i
    #   mu_diff_consonant:
    #       computed from adjacent duration_consonant values
    #       each duration_consonant[i] is the non-vowel duration inside syllable i

    # Important:
    #   The formula naturally produces one value per interval, so these fields
    #   should ideally be list[float], not float.
    #   If the dataclass must keep float, store an aggregate value such as mean(feature_list).
    devi_mu_syllable: list[float]
    mu_diff_syllable: list[float]

    devi_mu_vowel: list[float]
    mu_diff_vowel: list[float]

    devi_mu_consonant: list[float]
    mu_diff_consonant: list[float]

    filepath: str
