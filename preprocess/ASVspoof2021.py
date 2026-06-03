import argparse
import pickle
from pathlib import Path

from tqdm import tqdm


TRACK_CONFIG = {
    "LA": {
        "protocol": "keys/LA/CM/trial_metadata.txt",
        "audio_dir": "ASVspoof2021_LA_eval/flac",
        "output": "ASVspoof2021_LA_eval.pkl",
        "columns": (
            "SpeakerID",
            "FileStem",
            "Codec",
            "Transmission",
            "AttackLabel",
            "Key",
            "Trim",
            "Subset",
        ),
    },
    "DF": {
        "protocol": "keys/DF/CM/trial_metadata.txt",
        "audio_dir": "ASVspoof2021_DF_eval/flac",
        "output": "ASVspoof2021_DF_eval.pkl",
        "columns": (
            "SpeakerID",
            "FileStem",
            "Codec",
            "Source",
            "AttackLabel",
            "Key",
            "Trim",
            "Subset",
            "Vocoder",
            "Task",
            "Team",
            "Gender",
            "Language",
        ),
    },
}

KEY_TO_LABEL = {
    "bonafide": 0,
    "spoof": 1,
}


def parse_protocol_row(
    line: str,
    protocol_path: Path,
    line_no: int,
    columns: tuple[str, ...],
) -> dict[str, str] | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != len(columns):
        print(f"警告: {protocol_path}:{line_no} 欄位數不是 {len(columns)}，已跳過: {line.strip()}")
        return None
    return dict(zip(columns, parts))


def build_common_item(row: dict[str, str], audio_dir: Path, track: str) -> dict:
    file_stem = row["FileStem"]
    filename = f"{file_stem}.flac"
    key = row["Key"].lower()

    return {
        "FileName": filename,
        "AudioPath": str(audio_dir / filename),
        "Sentence": "",
        "Split_Set": "eval",
        "Track": track,
        "SpeakerID": row["SpeakerID"],
        "SpeakerGender": row.get("Gender", "-"),
        "Codec": row["Codec"],
        "CodecQ": "-",
        "CodecSeed": "-",
        "AttackTag": row.get("Transmission", row.get("Source", "-")),
        "AttackLabel": row["AttackLabel"],
        "Key": key,
        "DeepfakeLabel": KEY_TO_LABEL.get(key),
        "Trim": row["Trim"],
        "Subset": row["Subset"],
    }


def preprocess_track(data_root: Path, output_dir: Path, track: str) -> list[dict]:
    track = track.upper()
    config = TRACK_CONFIG[track]
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

    for line_no, line in enumerate(tqdm(lines, desc=f"處理 {track} eval protocol"), start=1):
        row = parse_protocol_row(line, protocol_path, line_no, config["columns"])
        if row is None:
            continue

        item = build_common_item(row, audio_dir, track)
        if item["DeepfakeLabel"] is None:
            unknown_key_count += 1

        if track == "LA":
            item.update(
                {
                    "Transmission": row["Transmission"],
                }
            )
        elif track == "DF":
            item.update(
                {
                    "Source": row["Source"],
                    "Vocoder": row["Vocoder"],
                    "Task": row["Task"],
                    "Team": row["Team"],
                    "Gender": row["Gender"],
                    "Language": row["Language"],
                }
            )

        final_data.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(final_data, f)

    print(f"\n{track} eval 預處理完成")
    print(f"輸出檔案: {output_path}")
    print(f"樣本總數: {len(final_data)}")
    if unknown_key_count:
        print(f"警告: {track} 有 {unknown_key_count} 筆未知 Key，DeepfakeLabel 設為 None")

    return final_data


def preprocess_asvspoof2021(data_root: str, output_dir: str, tracks: list[str]) -> dict[str, list[dict]]:
    root_path = Path(data_root)
    out_path = Path(output_dir)

    results = {}
    for track in tracks:
        track = track.upper()
        if track not in TRACK_CONFIG:
            raise ValueError(f"Unsupported ASVspoof2021 track: {track}")
        results[track] = preprocess_track(root_path, out_path, track)

    total = sum(len(items) for items in results.values())
    print("\nASVspoof2021 預處理完成！")
    print(f"總計樣本: {total}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof2021 LA/DF protocol 與 key metadata 整合工具")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/home/public/dataset/ASVspoof2021",
        help="ASVspoof2021 根目錄",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/home/public/dataset/ASVspoof2021",
        help="輸出 pkl 目錄",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        type=str.upper,
        default=["LA", "DF"],
        choices=sorted(TRACK_CONFIG.keys()),
        help="要處理的 track",
    )

    args = parser.parse_args()
    preprocess_asvspoof2021(args.input, args.output_dir, args.tracks)
