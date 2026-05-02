import os
import re
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

# 官方指定的目標情緒
TARGET_EMOTIONS = ["neu", "hap", "ang", "sad", "exc", "fru"]

def parse_iemocap_emotions(session_path: Path):
    """
    參考官方邏輯解析 EmoEvaluation 目錄下的標籤。
    """
    emo_map = {}
    label_dir = session_path / "dialog" / "EmoEvaluation"
    if not label_dir.exists():
        return emo_map

    for label_path in label_dir.glob("*.txt"):
        try:
            with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    
                    # 參考官方: 使用 tab 或 換行分割
                    parts = re.split("[\t\n]", line)
                    if len(parts) < 3:
                        continue
                        
                    wav_stem = parts[1]
                    label = parts[2]
                    
                    # 過濾官方指定的情緒 (可選，這裡我們先保留所有但轉大寫)
                    # 如果您想嚴格遵守官方過濾，可以取消下面註解
                    # if label not in TARGET_EMOTIONS:
                    #     continue
                        
                    emo_map[wav_stem] = label
        except Exception as e:
            print(f"讀取標籤檔失敗 {label_path}: {e}")
            
    return emo_map

def parse_iemocap_transcripts(session_path: Path):
    """
    解析 dialog/transcriptions 目錄下的逐字稿。
    """
    trans_map = {}
    trans_dir = session_path / "dialog" / "transcriptions"
    if not trans_dir.exists():
        return trans_map

    for trans_file in trans_dir.glob("*.txt"):
        try:
            with open(trans_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    
                    parts = line.split(":", 1)
                    meta = parts[0].strip()
                    text = parts[1].strip()
                    
                    # ID 可能是 "Ses01F_impro01_F000 [12.0-15.0]"，取第一項
                    utt_id = meta.split(" ")[0]
                    trans_map[utt_id] = text
        except Exception as e:
            print(f"讀取文字檔失敗 {trans_file}: {e}")
    return trans_map

def preprocess_iemocap(data_root: str, output_pkl: str):
    """
    整合處理邏輯。
    """
    root_path = Path(data_root)
    # 支援兩種結構: 直接在 root 下或 root/IEMOCAP 下
    if (root_path / "IEMOCAP").exists():
        root_path = root_path / "IEMOCAP"

    final_data = []
    # 搜尋 Session1-5
    sessions = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Session")])
    
    if not sessions:
        print(f"錯誤: 在 {root_path} 下找不到任何 Session 目錄")
        return

    # 指定作為開發集的演員
    DEV_ACTOR = "Ses04F"

    for session in sessions:
        print(f"正在處理 {session.name}...")
        
        emo_map = parse_iemocap_emotions(session)
        trans_map = parse_iemocap_transcripts(session)
        
        count = 0
        for utt_id, emotion in emo_map.items():
            # 決定 Split_Set
            actor_id = utt_id[:6]
            if "Session5" in session.name:
                split = "test"
            elif actor_id == DEV_ACTOR:
                split = "dev"
            else:
                split = "train"
            
            final_data.append({
                "FileName": f"{utt_id}.wav",
                "EmoClass": emotion.upper(),
                "Sentence": trans_map.get(utt_id, ""),
                "Split_Set": split,
                "Session": session.name,
                "Actor": actor_id
            })
            count += 1
        print(f"  已成功加入 {count} 筆樣本")

    # 儲存
    output_file = Path(output_pkl)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(final_data, f)

    print(f"\nIEMOCAP 預處理完成！")
    print(f"輸出檔案: {output_pkl}")
    print(f"總計樣本: {len(final_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEMOCAP 官方風格預處理工具")
    parser.add_argument("--input", "-i", type=str, default="dataset/IEMOCAP", help="IEMOCAP 根目錄")
    parser.add_argument("--output", "-o", type=str, default="dataset/IEMOCAP/IEMOCAP_combined.pkl", help="輸出路徑")
    
    args = parser.parse_args()
    preprocess_iemocap(args.input, args.output)
