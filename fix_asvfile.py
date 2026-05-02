import os
import torchaudio
from pathlib import Path
from tqdm import tqdm
import subprocess

def batch_fix_dataset(root_dir):
    root = Path(root_dir)
    flac_files = list(root.rglob("*.flac"))
    print(f"開始掃描 {len(flac_files)} 個檔案...")

    fixed_count = 0
    error_count = 0

    for f_path in tqdm(flac_files):
        try:
            # 測試讀取
            _ , _ = torchaudio.load(str(f_path))
        except Exception:
            # 讀取失敗，啟動救援
            temp_p = str(f_path) + ".tmp.flac"
            cmd = ["ffmpeg", "-i", str(f_path), "-y", "-loglevel", "error", temp_p]
            
            res = subprocess.run(cmd)
            if res.returncode == 0:
                os.replace(temp_p, str(f_path))
                fixed_count += 1
            else:
                error_count += 1
                print(f"無法修復: {f_path}")

    print(f"\n掃描完成！")
    print(f"成功修復: {fixed_count} 個檔案")
    print(f"完全無法修復 (FFmpeg也失敗): {error_count} 個檔案")

if __name__ == "__main__":
    # 指向你的數據集路徑
    dataset_dir = "/home/icebird/01_proj/ssl-mamba/dataset/ASVspoof2021/ASVspoof2021_DF_eval/flac"
    batch_fix_dataset(dataset_dir)
