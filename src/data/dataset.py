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
        self.data: list[Sample] = []
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def _get_cache_path(self, dataset_dir: Path, dataset_name: str, subset_list: list) -> Path:
        cache_dir = dataset_dir / ".cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        subset_str = "_".join(sorted([str(s) for s in subset_list]))
        return cache_dir / f"{dataset_name}_{subset_str}.pkl"

    def _load_cache(self, cache_path: Path) -> bool:
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
        if not data:
            print(f"[Dataset] No data to save for cache: {cache_path}")
            return
        print(f"[Dataset] Saving {len(data)} samples to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    
    def preload_meld(self, dataset_path: Path, subset_list: list[str]):
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
        return len(self.data)

    def get_lengths(self) -> list[int]:
        return [sample.length for sample in self.data]

    def __getitem__(self, idx) -> Sample:
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
        if max_len is None:
            max_len = max(len(getattr(item, field)) for item in batch)
        x = [getattr(item, field)[:max_len] for item in batch]
        lengths = [len(v) for v in x]
        return pad_sequence(x, batch_first=True, padding_value=padding_value), torch.tensor(lengths, dtype=torch.int32)
    
    def _get_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask

    def collate_fn_padded(self, batch: list[Sample]) -> Batch:
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
        return [item.length for item in self.data]

if __name__ == "__main__":
    # Example usage
    dataset = EmotionDataset()
    dataset_root = Path('/home/icebird/01_proj/ssl-mamba/dataset')
    dataset.preload_meld(
        dataset_path=dataset_root,
        subset_list=['train', 'dev', 'test']
    )
    print(f"Loaded {len(dataset)} samples from MELD")
