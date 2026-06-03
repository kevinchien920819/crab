import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .dataclass import Batch, Duration, Sample
from .utils import to_frame_idx_list
from torchaudio.datasets import IEMOCAP
class EmotionDataset(Dataset):
    def __init__(self, tokenizer=None, text_max_len: int = 128):
        """Initialize an emotion dataset with optional text tokenization."""
        self.data: list[Sample] = []
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def _get_cache_path(self, dataset_dir: Path, dataset_name: str, subset_list: list) -> Path:
        """Build the cache path for a dataset and subset list."""
        cache_dir = dataset_dir / ".cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        subset_str = "_".join(sorted([str(s) for s in subset_list]))
        return cache_dir / f"{dataset_name}_{subset_str}.pkl"

    def _load_cache(self, cache_path: Path) -> bool:
        """Load cached samples into memory when a non-empty cache exists."""
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                if cached_data and len(cached_data) > 0:
                    print(f"[Dataset] Loading {len(cached_data)} samples from cache: {cache_path}")
                    self.data.extend(cached_data)
                    return True
                else:
                    print(f"[Dataset] Cache found but empty, ignoring: {cache_path}")
        return False

    def _save_cache(self, cache_path: Path, data: list[Sample]):
        """Persist preloaded samples to a cache file."""
        if not data:
            print(f"[Dataset] No data to save for cache: {cache_path}")
            return
        print(f"[Dataset] Saving {len(data)} samples to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def preload_meld(self, dataset_path: Path, subset_list: list[str]):
        """Preload MELD metadata and audio paths for the requested subsets."""
        cache_path = self._get_cache_path(dataset_path, "MELD", subset_list)
        if self._load_cache(cache_path):
            return

        emotion_mapping = {
            'neutral': 0,
            'surprise': 1,
            'fear': 2,
            'sadness': 3,
            'joy': 4,
            'disgust': 5,
            'anger': 6
        }
        sentiment_mapping = {
            'neutral': 0,
            'positive': 1,
            'negative': 2
        }

        data = []
        for subset in subset_list:
            subset_split = subset + '_splits'
            label_csv_path = dataset_path / 'MELD' / 'csv' / f'{subset}_sent_emo.csv'
            mp4_dir = dataset_path / 'MELD' / subset_split

            if not label_csv_path.exists():
                print(f"Warning: Label CSV not found at {label_csv_path}")
                continue

            label_df = pl.read_csv(label_csv_path)

            for row in tqdm(label_df.iter_rows(named=True), total=len(label_df), desc=f'Preloading MELD {subset}'):
                dia_id = row['Dialogue_ID']
                utt_id = row['Utterance_ID']
                emotion = row['Emotion']
                sentiment = row['Sentiment']
                text = row['Utterance']

                filename = f"dia{dia_id}_utt{utt_id}.mp4"
                audio_path = mp4_dir / filename

                if not audio_path.exists():
                    audio_path = audio_path.with_suffix('.wav')
                    if not audio_path.exists():
                        continue
                    filename = audio_path.name

                try:
                    info = torchaudio.info(str(audio_path))
                except Exception:
                    print(f"Warning: Could not read audio info for {audio_path}")
                    continue

                data.append(
                    Sample(
                        filename    = filename,
                        path        = str(audio_path),
                        length      = info.num_frames,
                        emotion_label = emotion_mapping.get(emotion.lower()),
                        sentiment_label = sentiment_mapping.get(sentiment.lower()),
                        sentence = text,
                        dataset_name = "MELD",
                    )
                )

        self.data.extend(data)
        self._save_cache(cache_path, data)
    def preload_msp_podcast(self, dataset_dir: Path, subset_list: list):
        """Preload MSP-Podcast metadata and audio paths for the requested subsets."""
        cache_path = self._get_cache_path(dataset_dir, "MSP_Podcast", subset_list)
        if self._load_cache(cache_path):
            return

        # 預期的 pkl 路徑 (根據之前 preprocess/MSP.py 的預設)
        pkl_path = dataset_dir /"MSP_Podcast"/ "MSP_combined.pkl"
        if not pkl_path.exists():
            # 嘗試根目錄
            pkl_path = dataset_dir / "MSP_combined.pkl"

        if not pkl_path.exists():
            print(f"Warning: MSP pkl not found at {pkl_path}. Please run preprocess/MSP.py first.")
            return

        print(f"[Dataset] Loading MSP-Podcast from {pkl_path}")
        with open(pkl_path, "rb") as f:
            metadata_list = pickle.load(f)

        emotion_mapping = {
            'N': 0, 'NEUTRAL': 0,
            'U': 1, 'SURPRISE': 1,
            'F': 2, 'FEAR': 2,
            'S': 3, 'SADNESS': 3,
            'H': 4, 'HAPPINESS': 4, 'JOY': 4,
            'D': 5, 'DISGUST': 5,
            'A': 6, 'ANGER': 6,
            'C': 7, 'CONTEMPT': 7,
            'O': 7, 'OTHER': 7
        }

        # 建立音檔索引以便快速查找 (音檔通常在 Audio 資料夾下)
        audio_dir = dataset_dir / "Audio"
        if not audio_dir.exists():
            audio_dir = dataset_dir # 退而求其次搜尋全目錄

        print(f"[Dataset] Indexing MSP audio files in {audio_dir}...")
        audio_map = {f.name: f for f in audio_dir.rglob("*.wav")}

        data = []
        subset_list_lower = [s.lower() for s in subset_list]

        for item in tqdm(metadata_list, desc=f"Loading MSP-Podcast {subset_list}"):
            split = str(item.get("Split_Set", "")).lower()
            if split not in subset_list_lower:
                continue

            filename = item["FileName"]
            # 有些 CSV 存的是檔名，有些含副檔名，統一處理
            if not filename.endswith(".wav"):
                filename += ".wav"

            audio_path = audio_map.get(filename)
            if not audio_path or not audio_path.exists():
                continue

            try:
                info = torchaudio.info(str(audio_path))
                data.append(
                    Sample(
                        filename    = filename,
                        path        = str(audio_path),
                        length      = info.num_frames,
                        emotion_label = emotion_mapping.get(item["EmoClass"].upper(), 7),
                        sentence = item.get("Sentence", ""),
                        dataset_name = "MSP_Podcast",
                    )
                )
            except Exception:
                continue

        self.data.extend(data)
        self._save_cache(cache_path, data)

    def preload_iemocap(self, dataset_dir: Path, subset_list: list):
        """Preload IEMOCAP metadata and audio paths for the requested subsets."""
        cache_path = self._get_cache_path(dataset_dir, "IEMOCAP", subset_list)
        if self._load_cache(cache_path):
            return

        pkl_path = dataset_dir / "IEMOCAP" / "IEMOCAP_combined.pkl"
        if not pkl_path.exists():
            print(f"Warning: IEMOCAP pkl not found at {pkl_path}. Please run preprocess/IEMOCAP.py first.")
            return

        print(f"[Dataset] Loading IEMOCAP from {pkl_path}")
        with open(pkl_path, "rb") as f:
            metadata_list = pickle.load(f)

        # 嚴格遵循官方/使用者指定的 6 種情緒: neu, hap, ang, sad, exc, fru
        emotion_mapping = {
            'NEU': 0,
            'EXC': 1,
            'HAP': 1,
            'ANG': 2,
            'SAD': 3,
            # 'FRU': 5
        }

        # 建立音檔索引
        print(f"[Dataset] Indexing IEMOCAP audio files in {dataset_dir}...")
        audio_map = {f.name: f for f in dataset_dir.rglob("*.wav")}

        data = []
        subset_list_lower = [s.lower() for s in subset_list]

        for item in tqdm(metadata_list, desc=f"Loading IEMOCAP {subset_list}"):
            split = str(item.get("Split_Set", "")).lower()
            if split not in subset_list_lower:
                continue

            emotion = item["EmoClass"].upper()
            if emotion not in emotion_mapping:
                continue # 過濾掉非指定情緒

            filename = item["FileName"]
            audio_path = audio_map.get(filename)
            if not audio_path or not audio_path.exists():
                continue

            try:
                info = torchaudio.info(str(audio_path))
                data.append(Sample(
                    filename=filename,
                    path=str(audio_path),
                    length=info.num_frames,
                    emotion_label=emotion_mapping[emotion],
                    sentence=item.get("Sentence", ""),
                    dataset_name="IEMOCAP",
                ))
            except Exception:
                continue

        self.data.extend(data)
        self._save_cache(cache_path, data)



    def __len__(self):
        """Return the number of preloaded emotion samples."""
        return len(self.data)

    def get_lengths(self) -> list[int]:
        """Return raw waveform lengths for preloaded emotion samples."""
        return [sample.length for sample in self.data]

    def __getitem__(self, idx) -> Sample:
        """Load, resample, normalize, tokenize, and return one emotion sample."""
        sample: Sample = self.data[idx]

        if not os.path.exists(sample.path):
            raise FileNotFoundError(f"Audio file not found: {sample.path}")

        try:
            try:
                wavform, sr = torchaudio.load(sample.path)
            except Exception as e:
                import subprocess
                import io
                proc = subprocess.run([
                    'ffmpeg', '-v', 'error', '-y', '-i', sample.path,
                    '-vn', '-ac', '1', '-ar', '16000', '-f', 'wav', '-'
                ], capture_output=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg fallback failed: {proc.stderr.decode('utf-8', errors='ignore')}") from e
                wavform, sr = torchaudio.load(io.BytesIO(proc.stdout))

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wavform = resampler(wavform)
                sr = 16000
            if wavform.ndim == 2:
                wavform = wavform.mean(axis=0)
            if wavform.numel() == 0:
                print(f"[Dataset] Warning: Empty waveform for {sample.path}, using 1s of silence")
                wavform = torch.zeros(16000)

            peak = wavform.abs().max()
            if peak > 0:
                wavform = wavform / peak
            tokens = None
            token_mask = None
            if self.tokenizer is not None and sample.sentence is not None:
                inputs = self.tokenizer(
                    sample.sentence,
                    add_special_tokens=True,
                    truncation=True,
                    padding='max_length',
                    max_length=self.text_max_len,
                    return_attention_mask=True,
                )
                tokens = torch.tensor(inputs['input_ids'])
                token_mask = torch.tensor(inputs['attention_mask'], dtype=torch.bool)

            return Sample(
                filename        =sample.filename,
                path            =sample.path,
                length          = len(wavform),
                wavform         = wavform,
                emotion_label   =sample.emotion_label,
                sentiment_label =sample.sentiment_label,
                sentence        =sample.sentence,
                tokens          =tokens,
                token_mask      =token_mask,
                dataset_name    =sample.dataset_name,
            )
        except Exception as e:
            print(f"[Dataset] Error loading file: {sample.path}")
            raise RuntimeError(f"Error loading {sample.path}: {e}") from e
    def _pad_batch(self, batch: list[Sample], field: str, max_len=None, padding_value=0.0) -> torch.Tensor:
        """Pad a tensor field from samples and return padded data with lengths."""
        if max_len is None:
            max_len = max(len(getattr(item, field)) for item in batch)
        x = [getattr(item, field)[:max_len] for item in batch]
        lengths = [len(v) for v in x]
        return pad_sequence(x, batch_first=True, padding_value=padding_value), torch.tensor(lengths, dtype=torch.int32)

    def _get_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create a boolean padding mask from sequence lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask

    def collate_fn_padded(self, batch: list[Sample]) -> Batch:
        """Collate emotion samples into a padded Batch object."""
        wavform, wav_lengths = self._pad_batch(batch, 'wavform')
        audio_mask = self._get_mask(wav_lengths, wavform.size(1))

        tokens = None
        text_mask = None
        if any(item.tokens is not None for item in batch):
            tokens = torch.stack([item.tokens for item in batch if item.tokens is not None])
            text_mask = torch.stack([item.token_mask for item in batch if item.token_mask is not None])

        return Batch(
            filenames       = [item.filename for item in batch],
            path            = [item.path for item in batch],
            wavform         = wavform,
            length          = wav_lengths,
            audio_mask      = audio_mask,
            sentences       = [item.sentence for item in batch if item.sentence is not None],
            tokens          = tokens,
            text_mask       = text_mask,
            emotion_labels  = torch.tensor([item.emotion_label for item in batch if item.emotion_label is not None], dtype=torch.long),
            sentiment_labels= torch.tensor([item.sentiment_label for item in batch if item.sentiment_label is not None], dtype=torch.long),
        )
    def get_lengths(self) -> list[int]:
        """Return raw waveform lengths for preloaded emotion samples."""
        return [item.length for item in self.data]


class DeepfakeDataset(Dataset):
    def __init__(self, downsample_factor: int = 320):
        """Initialize a deepfake dataset and duration downsampling settings."""
        self.data: list[Sample] = []
        self.is_labeled = False
        self.downsample_factor = downsample_factor

    def preload_asvspoof(self, dataset_path: Path, year = '2019_LA', subset_list: list[str] = ['train'], use_duration: bool = False):
        # subset_list: ['train', 'dev', 'eval']
        """Preload ASVspoof metadata and optional duration features into samples."""
        data = []
        self.use_duration = use_duration
        self.sample_rate = 16000

        if year == '5':
            year_prefix = '5'
        else:
            year_prefix, track= year.split('_')

        asvspoof_str = f'ASVspoof{year_prefix}'

        subset_tag = '-'.join(sorted(subset_list))
        duration_tag = 'with_duration' if self.use_duration else 'no_duration'
        cache_path = dataset_path / asvspoof_str / f'cache_{year}_{subset_tag}_{duration_tag}_ds{self.downsample_factor}.pkl'

        if cache_path.exists():
            with cache_path.open('rb') as f:
                cached = pickle.load(f)
            self.data.extend(cached)
            self.is_labeled = True
            print(f'[Dataset] Loaded ASVspoof cache: {cache_path} ({len(cached)} samples)')
            return

        for subset in subset_list:
            if year == '2019_LA':
                flac_dir = dataset_path / asvspoof_str / f'ASVspoof{year}_{subset}' / 'flac'
            elif year == '2021_LA' or year == '2021_DF':
                assert subset == 'eval', f"subset in 2021 must be 'eval', not {subset}"
                flac_dir = dataset_path / asvspoof_str / f'ASVspoof{year}_{subset}' / 'flac'
            elif year == '5':
                subset_dict = {'train': 'T', 'dev': 'D', 'eval': 'E_eval',}
                flac_dir = dataset_path / asvspoof_str / f'flac_{subset_dict[subset]}'
            else:
                raise NotImplementedError(f'Unsupported ASVspoof year: {year}')

            protocol_suffix = 'trn' if subset == 'train' else 'trl'
            year_dot = year.replace('_', '.')

            if year == '2019_LA':
                label_txt_path = dataset_path / asvspoof_str / f'ASVspoof{year}_cm_protocols' / f'ASVspoof{year_dot}.cm.{subset}.{protocol_suffix}.txt'
            elif year == '2021_LA' or year == '2021_DF':
                assert subset == 'eval',f'{year} only support eval set, not {subset}'
                label_txt_path = dataset_path / asvspoof_str / 'keys' / track / 'CM' / 'trial_metadata.txt'
            elif year == '5':
                label_txt_path = dataset_path / asvspoof_str / f'ASVspoof_cm_protocols' / f'ASVspoof{year_dot}.{subset}.txt'
            duration_csv_path = dataset_path / asvspoof_str / f'ASVspoof{year}_csv' / f'ASVspoof{year_dot}.{subset}.csv'

            if self.use_duration:
                self.duration_df = pl.read_csv(duration_csv_path)
                for col in ['utt', 'word', 'vowel', 'consonant']:
                    span = pl.col(f'{col}_start_end').str.split('-').list.eval(
                        pl.element().str.split('_').list.eval(pl.element().cast(pl.Float64))
                    )
                    start_t = span.list.eval(pl.element().list.get(0))
                    end_t = span.list.eval(pl.element().list.get(1))
                    self.duration_df = self.duration_df.with_columns(
                        span.alias(f'{col}_span'),
                        start_t.alias(f'{col}_start_time'),
                        end_t.alias(f'{col}_end_time'),
                        (end_t - start_t).alias(f'{col}_duration'),
                        to_frame_idx_list(start_t, self.sample_rate, self.downsample_factor).alias(f'{col}_sid'),
                        to_frame_idx_list(end_t, self.sample_rate, self.downsample_factor).alias(f'{col}_eid'),
                    ).drop(f'{col}_start_end')

            with open(label_txt_path, 'r') as f:
                lines = f.read().splitlines()
                for line in tqdm(lines, desc=f'Parsing labels for {subset}'):
                    # LA_0069 LA_D_1047731 - - bonafide
                    if year == '2019_LA' or year == '5':
                        speaker_id, filename, _, _, label_str = line.split()
                    elif year == '2021_LA':
                        speaker_id, filename, _, _, _, label_str, _, _ = line.split()
                    elif year == '2021_DF':
                        line = line.split()
                        speaker_id = line[0]
                        filename = line[1]
                        label_str = line[5]

                    label = 0 if label_str == 'bonafide' else 1

                    flac_path = flac_dir / f'{filename}.flac'

                    if not flac_path.exists():
                        raise FileNotFoundError(f'Missing audio file: {flac_path}')

                    utt_data = word_data = vowel_data = consonant_data = None
                    if self.use_duration:
                        d_data = self.duration_df.filter(pl.col('file_name') == f'{filename}')
                        if d_data.is_empty():
                            # raise ValueError(f'Missing duration info for sample: {filename}')
                            print(f'[Dataset] Warning: Missing duration info for sample: {filename}, using default 0s')
                            continue
                        if d_data.shape[0] > 1:
                            raise ValueError(f'Multiple duration entries for sample: {filename}')
                        d_data = d_data[0]

                        for col in ['utt', 'word', 'vowel', 'consonant']:
                            duration = Duration(
                                st = d_data[f'{col}_start_time'].to_numpy().item(),
                                et = d_data[f'{col}_end_time'].to_numpy().item(),
                                sid = d_data[f'{col}_sid'].to_numpy().item(),
                                eid = d_data[f'{col}_eid'].to_numpy().item(),
                                d = d_data[f'{col}_duration'].to_numpy().item(),
                            )
                            if col == 'utt':
                                utt_data = duration
                            elif col == 'word':
                                word_data = duration
                            elif col == 'vowel':
                                vowel_data = duration
                            elif col == 'consonant':
                                consonant_data = duration

                    data.append(
                        Sample(
                            filename        = filename,
                            path            = str(flac_path),
                            length          = 0, # to be filled in __getitem__
                            deepfake_label  = label,
                            utt_data        = utt_data,
                            word_data       = word_data,
                            vowel_data      = vowel_data,
                            consonant_data  = consonant_data,
                        )
                    )

        self.data.extend(data)
        self.is_labeled = True
        with cache_path.open('wb') as f:
            pickle.dump(data, f)
        print(f'[Dataset] Saved ASVspoof cache: {cache_path} ({len(data)} samples)')

    def get_lengths(self) -> list[int]:
        """Return raw waveform lengths for preloaded deepfake samples."""
        return [item.length for item in self.data]

    def __len__(self):
        """Return the number of preloaded deepfake samples."""
        return len(self.data)

    def __getitem__(self, idx) -> Sample:
        """Load, resample, normalize, and return one deepfake sample."""
        sample: Sample = self.data[idx]

        if not os.path.exists(sample.path):
            raise FileNotFoundError(f"Audio file not found: {sample.path}")

        try:
            # wavform, _ = sf.read(sample.path, dtype='float32', always_2d=False)
            wavform, sr = torchaudio.load(sample.path)
            # wavform, sr = librosa.load(sample.path, sr=self.sample_rate)

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wavform = resampler(wavform)
                sr = 16000
            if wavform.ndim == 2:
                wavform = wavform.mean(axis=0)
            if wavform.numel() == 0:
                print(f"[Dataset] Warning: Empty waveform for {sample.path}, using 1s of silence")
                wavform = torch.zeros(self.sample_rate)

            peak = wavform.abs().max()
            if peak > 0:
                wavform = wavform / peak

            return Sample(
                filename        = sample.filename,
                path            = sample.path,
                length          = len(wavform),
                wavform         = wavform,
                deepfake_label  = sample.deepfake_label,
                utt_data        = sample.utt_data,
                word_data       = sample.word_data,
                vowel_data      = sample.vowel_data,
                consonant_data  = sample.consonant_data,
            )
        except Exception as e:
            print(f"[Dataset] Error loading file: {sample.path}")
            raise RuntimeError(f"Error loading {sample.path}: {e}") from e

    def _concat_batch(self, batch: list[Sample], field: str, dtype) -> torch.Tensor | None:
        """Concatenate a tensor field from samples into a packed batch tensor."""
        values = [
            torch.as_tensor(getattr(item, field), dtype=dtype) for item in batch
        ]
        if any(len(v) == 0 for v in values):
            return None
        return torch.cat(values, dim=0).unsqueeze(0) # [B, T]

    def collate_fn_packed(self, batch: list[Sample]) -> Batch:
        """Collate deepfake samples into a packed Batch object."""
        lengths = np.array([item.length for item in batch], dtype=np.int32)

        return Batch(
            path            = [item.path for item in batch],
            wavform         = self._concat_batch(batch, 'wavform', torch.float32).unsqueeze(-1),  # [B, T, 1]

            # length
            length          = lengths,
        )

    def _pad_batch(self, batch: list[Batch], field: str, max_len=None, padding_value=0.0) -> torch.Tensor:
        """Pad a tensor field from deepfake samples and return lengths."""
        if max_len is None:
            max_len = max(len(getattr(item, field)) for item in batch)
        x = [getattr(item, field)[:max_len] for item in batch]
        lengths = [len(v) for v in x]
        return pad_sequence(x, batch_first=True, padding_value=padding_value), torch.tensor(lengths, dtype=torch.int32)

    def collate_fn_padded(self, batch: list[Sample]) -> Batch:
        """Collate deepfake samples into a padded Batch with optional duration fields."""
        wavform, length = self._pad_batch(batch, 'wavform')

        word_d = word_sid = None
        vowel_d = vowel_sid = vowel_devi = vowel_mu_diff = None
        consonant_d = consonant_sid = consonant_devi = consonant_mu_diff = None
        if self.use_duration:
            max_len = length.max().item() # token length
            max_time_sec = max_len / self.sample_rate

            word_sid = [torch.as_tensor(item.word_data.sid[item.word_data.et <= (max_time_sec)]) for item in batch]
            word_sid = pad_sequence(word_sid, batch_first=True, padding_value=-1.0)

            word_d = [torch.as_tensor(item.word_data.d[item.word_data.et <= (max_time_sec)], dtype=torch.float32) for item in batch]
            word_d = pad_sequence(word_d, batch_first=True, padding_value=-1.0)

            vowel_sid = [
                torch.as_tensor(item.vowel_data.sid[item.vowel_data.et <= (max_time_sec)])
                if (item.vowel_data is not None and item.vowel_data.et is not None)
                else torch.tensor([], dtype=torch.long)
                for item in batch
            ]
            vowel_d = [
                torch.as_tensor(item.vowel_data.d[item.vowel_data.et <= max_time_sec], dtype=torch.float32)
                if (item.vowel_data is not None and item.vowel_data.et is not None)
                else torch.tensor([], dtype=torch.float32)
                for item in batch
            ]

            # vowel_d = [torch.as_tensor(item.vowel_data.d[item.vowel_data.et <= (max_time_sec)], dtype=torch.float32)for item in batch]
            vowel_devi = [(d - d.mean()) if len(d) > 0 else torch.zeros_like(d) for d in vowel_d]
            vowel_mu_diff = [torch.full_like(d, (((d[:-1] - d[1:]) / ((d[:-1] + d[1:]) / 2)).abs().sum() / (len(d)-1) ) if len(d) > 1 else 0.0) for d in vowel_d]

            vowel_sid = pad_sequence(vowel_sid, batch_first=True, padding_value=-1.0)
            vowel_d = pad_sequence(vowel_d,batch_first=True, padding_value=-1.0)
            vowel_devi = pad_sequence(vowel_devi,batch_first=True, padding_value=-1.0)
            vowel_mu_diff = pad_sequence(vowel_mu_diff,batch_first=True, padding_value=-1.0)

            consonant_sid = [
                torch.as_tensor(item.consonant_data.sid[item.consonant_data.et <= (max_time_sec)])
                if (item.consonant_data is not None and item.consonant_data.et is not None)
                else torch.tensor([], dtype=torch.long)
                for item in batch
            ]
            consonant_d = [
                torch.as_tensor(item.consonant_data.d[item.consonant_data.et <= max_time_sec], dtype=torch.float32)
                if (item.consonant_data is not None and item.consonant_data.et is not None)
                else torch.tensor([], dtype=torch.float32)
                for item in batch
            ]

            consonant_devi = [(d - d.mean()) if len(d) > 0 else torch.zeros_like(d) for d in consonant_d]
            consonant_mu_diff = [torch.full_like(d, (((d[:-1] - d[1:]) / ((d[:-1] + d[1:]) / 2)).abs().sum() / (len(d)-1) ) if len(d) > 1 else 0.0) for d in consonant_d]

            consonant_sid = pad_sequence(consonant_sid, batch_first=True, padding_value=-1.0)
            consonant_d = pad_sequence(consonant_d,batch_first=True, padding_value=-1.0)
            consonant_devi = pad_sequence(consonant_devi,batch_first=True, padding_value=-1.0)
            consonant_mu_diff = pad_sequence(consonant_mu_diff,batch_first=True, padding_value=-1.0)

            # print("\n--- Batch Duration Info ---")
            # for i, sample in enumerate(batch):
            #     print(f"File: {sample.filename}")
            #     print(f"  Vowel Durations: {vowel_d[i]}")
            #     print(f"  Vowel Deviation: {vowel_devi[i]}")
            #     print(f"  Word SIDs: {word_sid[i]}")
            # print("---------------------------\n")
        return Batch(
            filenames       = [item.filename for item in batch],
            path            = [item.path for item in batch],
            wavform         = wavform,
            length          = length,

            # for deepfake detection
            deepfake_labels = torch.tensor([item.deepfake_label for item in batch], dtype=torch.long)    if self.is_labeled else None,

            word_d          = word_d,

            vowel_d         = vowel_d,
            vowel_devi      = vowel_devi,
            vowel_mu_diff   = vowel_mu_diff,

            consonant_d     = consonant_d,
            consonant_devi  = consonant_devi,
            consonant_mu_diff = consonant_mu_diff,

            word_sid        = word_sid,
            vowel_sid       = vowel_sid,
            consonant_sid   = consonant_sid,
        )


if __name__ == "__main__":
    # Example usage
    dataset = EmotionDataset()
    dataset_root = Path('/home/icebird/01_proj/ssl-mamba/dataset')
    dataset.preload_meld(
        dataset_path=dataset_root,
        subset_list=['train', 'dev', 'test']
    )
    print(f"Loaded {len(dataset)} samples from MELD")
