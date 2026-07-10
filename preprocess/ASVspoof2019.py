import argparse
import pickle
from pathlib import Path

from tqdm import tqdm


SPLIT_CONFIG = {
    "train": {
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": "ASVspoof2019_LA_train/flac",
        "output": "ASVspoof2019_LA_train.pkl",
    },
    "dev": {
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": "ASVspoof2019_LA_dev/flac",
        "output": "ASVspoof2019_LA_dev.pkl",
    },
    "eval": {
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir": "ASVspoof2019_LA_eval/flac",
        "output": "ASVspoof2019_LA_eval.pkl",
    },
}

PROTOCOL_COLUMNS = (
    "SpeakerID",
    "FileStem",
    "Codec",
    "AttackLabel",
    "Key",
)

KEY_TO_LABEL = {
    "spoof": 0,
    "bonafide": 1,
}


def parse_protocol_row(line: str, protocol_path: Path, line_no: int) -> dict[str, str] | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != len(PROTOCOL_COLUMNS):
        print(f"警告: {protocol_path}:{line_no} 欄位數不是 5，已跳過: {line.strip()}")
        return None
    return dict(zip(PROTOCOL_COLUMNS, parts))


def preprocess_split(data_root: Path, output_dir: Path, split: str) -> list[dict]:
    config = SPLIT_CONFIG[split]
    protocol_path = data_root / config["protocol"]
    audio_dir = data_root / config["audio_dir"]
    output_path = output_dir / config["output"]

    if not protocol_path.exists():
        print(f"錯誤: 找不到 protocol 檔案 {protocol_path}")
        return []

    with open(protocol_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    final_data = []
    unknown_key_count = 0

    for line_no, line in enumerate(tqdm(lines, desc=f"處理 {split} protocol"), start=1):
        row = parse_protocol_row(line, protocol_path, line_no)
        if row is None:
            continue

        file_stem = row["FileStem"]
        filename = f"{file_stem}.flac"
        key = row["Key"].lower()
        deepfake_label = KEY_TO_LABEL.get(key)
        if deepfake_label is None:
            unknown_key_count += 1

        final_data.append(
            {
                "FileName": filename,
                "AudioPath": str(audio_dir / filename),
                "Sentence": "",
                "Split_Set": split,
                "SpeakerID": row["SpeakerID"],
                "SpeakerGender": "-",
                "Codec": row["Codec"],
                "CodecQ": "-",
                "CodecSeed": "-",
                "AttackTag": "-",
                "AttackLabel": row["AttackLabel"],
                "Key": key,
                "DeepfakeLabel": deepfake_label,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(final_data, f)

    print(f"\n{split} 預處理完成")
    print(f"輸出檔案: {output_path}")
    print(f"樣本總數: {len(final_data)}")
    if unknown_key_count:
        print(f"警告: {split} 有 {unknown_key_count} 筆未知 Key，DeepfakeLabel 設為 None")

    return final_data


def preprocess_asvspoof2019(data_root: str, output_dir: str) -> dict[str, list[dict]]:
    root_path = Path(data_root)
    out_path = Path(output_dir)

    results = {}
    for split in ("train", "dev", "eval"):
        results[split] = preprocess_split(root_path, out_path, split)

    total = sum(len(items) for items in results.values())
    print("\nASVspoof2019 LA 預處理完成！")
    print(f"總計樣本: {total}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof2019 LA protocol 整合工具")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/home/public/dataset/ASVspoof2019",
        help="ASVspoof2019 根目錄",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/home/public/dataset/ASVspoof2019",
        help="輸出 pkl 目錄",
    )

    args = parser.parse_args()
    preprocess_asvspoof2019(args.input, args.output_dir)
