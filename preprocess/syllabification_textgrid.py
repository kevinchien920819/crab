import argparse
import csv
import pickle
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal, TypeAlias

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this utility script.
    def tqdm(iterable, **_: object):
        return iterable

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from preprocess import textgrid
    from preprocess.dataclass import DatasetItem, SplitConfig
else:
    from . import textgrid
    from .dataclass import DatasetItem, SplitConfig

SplitName: TypeAlias = Literal["train", "dev", "eval"]
IntervalTuple: TypeAlias = tuple[float, float, str]
TranscriptData: TypeAlias = dict[str, str | int | float | list[float]]

SPLITS: tuple[SplitName, ...] = ("train", "dev", "eval")
DEFAULT_TRANSCRIPT_ROOT = Path("dataset/ASVspoof5/ASVspoof5_transcript")
DEFAULT_OUTPUT_ROOT = Path("dataset/ASVspoof5/ASVspoof5_syllabified_textgrid")
DEFAULT_DATA_ROOT = Path("dataset/ASVspoof5")

KEY_TO_LABEL = {
    "bonafide": 0,
    "spoof": 1,
}

PROTOCOL_COLUMNS = (
    "SpeakerID",
    "FileStem",
    "SpeakerGender",
    "Codec",
    "CodecQ",
    "CodecSeed",
    "AttackTag",
    "AttackLabel",
    "Key",
    "Tmp",
)

VOWEL_PHONES = {
    "AA", "AE", "AH", "AO", "AW",
    "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW",
}

CONSONANT_PHONES = {
    "B", "CH", "D", "DH", "F",
    "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R",
    "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",
}

SPLIT_CONFIG: dict[SplitName, SplitConfig] = {
    "train": SplitConfig(
        protocols="ASVspoof5.train.tsv",
        audio_dir="flac_T",
        output="ASVspoof5_train.pkl",
    ),
    "dev": SplitConfig(
        protocols="ASVspoof5.dev.track_1.tsv",
        audio_dir="flac_D",
        output="ASVspoof5_dev.pkl",
    ),
    "eval": SplitConfig(
        protocols="ASVspoof5.eval.track_1.tsv",
        audio_dir="flac_E_eval",
        output="ASVspoof5_eval.pkl",
    ),
}


def _is_textgrid_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".textgrid"


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _selected_splits(split: SplitName | Literal["all"]) -> tuple[SplitName, ...]:
    return SPLITS if split == "all" else (split,)


def _candidate_roots(input_root: Path, splits: tuple[SplitName, ...]) -> list[Path]:
    split_roots = [input_root / split for split in splits if (input_root / split).is_dir()]
    return split_roots if split_roots else [input_root]


def _iter_textgrid_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if _is_textgrid_file(path):
            yield path


def collect_textgrid_files(
    input_root: Path,
    splits: tuple[SplitName, ...] = SPLITS,
    output_root: Path | None = None,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in _candidate_roots(input_root, splits):
        for path in _iter_textgrid_files(root):
            if output_root is not None and _is_under(path, output_root):
                continue
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def default_single_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.syllables{input_path.suffix}")


def resolve_single_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return default_single_output_path(input_path)
    if output_path.suffix.lower() == ".textgrid":
        return output_path
    return output_path / input_path.name


def syllabify_textgrid_file(
    input_path: Path,
    output_path: Path,
    tier_name: str = "syllables",
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve() and not overwrite:
        raise ValueError(
            f"Refusing to overwrite input TextGrid without --overwrite: {input_path}"
        )

    grid = textgrid.read_textgrid(input_path)
    textgrid.add_syllable_tier(grid, tier_name=tier_name)
    textgrid.write_textgrid(grid, output_path)
    return output_path


def syllabify_textgrid_directory(
    input_root: Path,
    output_root: Path,
    splits: tuple[SplitName, ...] = SPLITS,
    limit: int | None = None,
    tier_name: str = "syllables",
    overwrite: bool = False,
    skip_errors: bool = False,
) -> tuple[list[Path], list[tuple[Path, Exception]]]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    textgrid_paths = collect_textgrid_files(input_root, splits=splits, output_root=output_root)
    if limit is not None:
        textgrid_paths = textgrid_paths[:limit]

    written: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    iterator = tqdm(textgrid_paths, desc=f"Syllabifying {input_root.name}", unit="file")
    for input_file in iterator:
        output_file = output_root / input_file.relative_to(input_root)
        try:
            written.append(
                syllabify_textgrid_file(
                    input_file,
                    output_file,
                    tier_name=tier_name,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            if not skip_errors:
                raise
            failures.append((input_file, exc))
            print(f"Warning: skipped {input_file}: {exc}")

    return written, failures



def _phone_base(phone: str) -> str:
    return re.sub(r"\d+$", "", phone.strip().upper())


def _clean_phone(phone: str) -> str:
    return phone.strip().upper()


def is_vowel(phone: str) -> bool:
    return _phone_base(phone) in VOWEL_PHONES


def is_consonant(phone: str) -> bool:
    return _phone_base(phone) in CONSONANT_PHONES


def get_protocol_path(data_root: Path, split: SplitName) -> Path:
    return data_root / SPLIT_CONFIG[split].protocols


def split_transcript_path(transcript_root: Path, split: SplitName) -> Path:
    split_path = transcript_root / split
    return split_path if split_path.exists() else transcript_root


def textgrid_path_map(transcript_path: Path) -> dict[str, Path]:
    if not transcript_path.exists():
        return {}
    return {path.stem: path for path in _iter_textgrid_files(transcript_path)}


def parse_protocol_row(line: str, protocol_path: Path, line_no: int) -> dict[str, str] | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != len(PROTOCOL_COLUMNS):
        print(
            f"Warning: {protocol_path}:{line_no} expected {len(PROTOCOL_COLUMNS)} "
            f"columns, got {len(parts)}. Skipped: {line.strip()}"
        )
        return None
    return dict(zip(PROTOCOL_COLUMNS, parts))


def read_protocol_rows(
    protocol_path: Path,
    limit: int | None = None,
    available_stems: set[str] | None = None,
    require_transcript: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with protocol_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            row = parse_protocol_row(line, protocol_path, line_no)
            if row is None:
                continue
            if (
                require_transcript
                and available_stems is not None
                and row["FileStem"] not in available_stems
            ):
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _grid_has_item(grid: textgrid.TextGrid, tier_name: str) -> bool:
    return any(item.name.lower() == tier_name.lower() for item in grid.items)


def read_textgrid_intervals(
    textgrid_path: Path,
    syllable_tier_name: str = "syllables",
    add_missing_syllables: bool = True,
) -> dict[str, list[IntervalTuple]]:
    grid = textgrid.read_textgrid(textgrid_path)
    if (
        add_missing_syllables
        and not _grid_has_item(grid, syllable_tier_name)
        and _grid_has_item(grid, "phones")
    ):
        textgrid.add_syllable_tier(grid, tier_name=syllable_tier_name)

    tiers: dict[str, list[IntervalTuple]] = {}
    for item in grid.items:
        tier_name = item.name.strip().lower()
        tiers.setdefault(tier_name, []).extend(
            (interval.xmin, interval.xmax, interval.text)
            for interval in item.intervals
        )
    return tiers


def _non_empty_intervals(intervals: list[IntervalTuple]) -> list[IntervalTuple]:
    return [(start, end, label.strip()) for start, end, label in intervals if label.strip()]


def _content_text(intervals: list[IntervalTuple]) -> str:
    return ", ".join(label for _, _, label in intervals)


def _start_times(intervals: list[IntervalTuple]) -> list[float]:
    return [start for start, _, _ in intervals]


def _end_times(intervals: list[IntervalTuple]) -> list[float]:
    return [end for _, end, _ in intervals]


def _round_duration(value: float) -> float:
    return round(max(0.0, value), 4)


def _round_feature(value: float) -> float:
    return round(value, 4)


def _durations(intervals: list[IntervalTuple]) -> list[float]:
    return [_round_duration(end - start) for start, end, _ in intervals]


def _duration(interval: IntervalTuple) -> float:
    start, end, _ = interval
    return _round_duration(end - start)


def _interval_midpoint(interval: IntervalTuple) -> float:
    start, end, _ = interval
    return (start + end) / 2


def _intervals_inside(parent: IntervalTuple, children: list[IntervalTuple]) -> list[IntervalTuple]:
    parent_start, parent_end, _ = parent
    return [
        child
        for child in children
        if parent_start - 1e-7 <= _interval_midpoint(child) <= parent_end + 1e-7
    ]


def _group_start_end(group: list[IntervalTuple], anchor: IntervalTuple) -> tuple[float, float]:
    if group:
        return group[0][0], group[-1][1]
    return anchor[0], anchor[0]


def _sentence_timing(
    word_intervals: list[IntervalTuple],
    syllable_intervals: list[IntervalTuple],
    phone_intervals: list[IntervalTuple],
) -> tuple[list[float], list[float], float]:
    source = word_intervals or syllable_intervals or phone_intervals
    if not source:
        return [], [], 0.0
    start = source[0][0]
    end = source[-1][1]
    return [start], [end], _round_duration(end - start)


def _deviation_from_mean(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    return [_round_feature(value - mean_value) for value in values]


def _normalized_pairwise_diff(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]

    pairwise: list[float] = []
    abs_pairwise: list[float] = []
    for left, right in zip(values, values[1:]):
        denominator = (left + right) / 2
        diff = 0.0 if denominator == 0 else (left - right) / denominator
        pairwise.append(_round_feature(diff))
        abs_pairwise.append(abs(diff))

    npvi = _round_feature(sum(abs_pairwise) / len(abs_pairwise)) if abs_pairwise else 0.0
    return [*pairwise, npvi]


def _per_syllable_vowel_consonant_data(
    syllable_intervals: list[IntervalTuple],
    phone_intervals: list[IntervalTuple],
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    starttime_vowel: list[float] = []
    endtime_vowel: list[float] = []
    duration_vowel: list[float] = []
    starttime_consonant: list[float] = []
    endtime_consonant: list[float] = []
    duration_consonant: list[float] = []

    for syllable in syllable_intervals:
        syllable_phones = _intervals_inside(syllable, phone_intervals)
        vowel_phones = [phone for phone in syllable_phones if is_vowel(phone[2])]
        consonant_phones = [phone for phone in syllable_phones if is_consonant(phone[2])]

        vowel_start, vowel_end = _group_start_end(vowel_phones, syllable)
        consonant_start, consonant_end = _group_start_end(consonant_phones, syllable)
        vowel_duration = _round_duration(sum(_duration(phone) for phone in vowel_phones))
        syllable_duration = _duration(syllable)

        starttime_vowel.append(vowel_start)
        endtime_vowel.append(vowel_end)
        duration_vowel.append(vowel_duration)
        starttime_consonant.append(consonant_start)
        endtime_consonant.append(consonant_end)
        duration_consonant.append(_round_duration(syllable_duration - vowel_duration))

    return (
        starttime_vowel,
        endtime_vowel,
        duration_vowel,
        starttime_consonant,
        endtime_consonant,
        duration_consonant,
    )


def extract_transcript_data(
    textgrid_path: Path,
    syllable_tier_name: str = "syllables",
) -> TranscriptData:
    tiers = read_textgrid_intervals(textgrid_path, syllable_tier_name=syllable_tier_name)
    word_intervals = _non_empty_intervals(tiers.get("words", []))
    syllable_intervals = _non_empty_intervals(tiers.get(syllable_tier_name.lower(), []))
    phone_intervals = [
        (start, end, _clean_phone(phone))
        for start, end, phone in _non_empty_intervals(tiers.get("phones", []))
    ]
    vowel_phone_intervals = [phone for phone in phone_intervals if is_vowel(phone[2])]

    sentence_start, sentence_end, sentence_duration = _sentence_timing(
        word_intervals,
        syllable_intervals,
        phone_intervals,
    )
    syllable_durations = _durations(syllable_intervals)
    word_durations = _durations(word_intervals)
    phoneme_durations = _durations(phone_intervals)
    (
        starttime_vowel,
        endtime_vowel,
        duration_vowel,
        starttime_consonant,
        endtime_consonant,
        duration_consonant,
    ) = _per_syllable_vowel_consonant_data(syllable_intervals, phone_intervals)

    return {
        "content_sentence": " ".join(label for _, _, label in word_intervals),
        "starttime_sentence": sentence_start,
        "endtime_sentence": sentence_end,
        "duration_sentence": sentence_duration,
        "content_word": _content_text(word_intervals),
        "word_count": len(word_intervals),
        "starttime_word": _start_times(word_intervals),
        "endtime_word": _end_times(word_intervals),
        "duration_word": word_durations,
        "content_syllable": _content_text(syllable_intervals),
        "syllable_count": len(syllable_intervals),
        "starttime_syllable": _start_times(syllable_intervals),
        "endtime_syllable": _end_times(syllable_intervals),
        "duration_syllable": syllable_durations,
        "content_phoneme": _content_text(phone_intervals),
        "phoneme_count": len(phone_intervals),
        "starttime_phoneme": _start_times(phone_intervals),
        "endtime_phoneme": _end_times(phone_intervals),
        "duration_phoneme": phoneme_durations,
        "content_vowel": _content_text(vowel_phone_intervals),
        "vowel_count": len(vowel_phone_intervals),
        "starttime_vowel": starttime_vowel,
        "endtime_vowel": endtime_vowel,
        "duration_vowel": duration_vowel,
        "starttime_consonant": starttime_consonant,
        "endtime_consonant": endtime_consonant,
        "duration_consonant": duration_consonant,
        "devi_mu_syllable": _deviation_from_mean(syllable_durations),
        "mu_diff_syllable": _normalized_pairwise_diff(syllable_durations),
        "devi_mu_vowel": _deviation_from_mean(duration_vowel),
        "mu_diff_vowel": _normalized_pairwise_diff(duration_vowel),
        "devi_mu_consonant": _deviation_from_mean(duration_consonant),
        "mu_diff_consonant": _normalized_pairwise_diff(duration_consonant),
    }


def read_transcripts(
    transcript_path: Path,
    file_stems: set[str] | None = None,
    syllable_tier_name: str = "syllables",
) -> dict[str, TranscriptData]:
    transcripts: dict[str, TranscriptData] = {}
    path_map = textgrid_path_map(transcript_path)
    if not path_map:
        print(f"Warning: transcript path has no TextGrid files: {transcript_path}")
        return transcripts

    selected_stems = sorted(file_stems) if file_stems is not None else sorted(path_map)
    selected = [(stem, path_map[stem]) for stem in selected_stems if stem in path_map]
    iterator = tqdm(selected, desc=f"Reading TextGrid {transcript_path.name}", unit="file")
    for file_stem, txt_path in iterator:
        transcripts[file_stem] = extract_transcript_data(
            txt_path,
            syllable_tier_name=syllable_tier_name,
        )
    return transcripts


def _get_list(transcript: TranscriptData, key: str) -> list[float]:
    value = transcript.get(key, [])
    return value if isinstance(value, list) else []


def _get_float(transcript: TranscriptData, key: str) -> float:
    value = transcript.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _get_int(transcript: TranscriptData, key: str) -> int:
    value = transcript.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _get_str(transcript: TranscriptData, key: str) -> str:
    value = transcript.get(key, "")
    return value if isinstance(value, str) else ""


def _protocol_word_count(row: dict[str, str], transcript: TranscriptData) -> int:
    word_count = _get_int(transcript, "word_count")
    if word_count:
        return word_count
    try:
        return int(row["Tmp"])
    except ValueError:
        return 0


def build_dataset_item(
    row: dict[str, str],
    transcript_data: dict[str, TranscriptData],
    audio_dir: Path,
) -> DatasetItem:
    file_stem = row["FileStem"]
    filename = f"{file_stem}.flac"
    transcript = transcript_data.get(file_stem, {})
    label = KEY_TO_LABEL.get(row["Key"].lower(), -1)

    return DatasetItem(
        speaker_id=row["SpeakerID"],
        flac_file_name=filename,
        speaker_gender=row["SpeakerGender"],
        codec=row["Codec"],
        codec_q=row["CodecQ"],
        codec_seed=row["CodecSeed"],
        attack_tag=row["AttackTag"],
        attack_label=row["AttackLabel"],
        label=label,
        content_sentence=_get_str(transcript, "content_sentence"),
        starttime_sentence=_get_list(transcript, "starttime_sentence"),
        endtime_sentence=_get_list(transcript, "endtime_sentence"),
        duration_sentence=_get_float(transcript, "duration_sentence"),
        content_word=_get_str(transcript, "content_word"),
        word_count=_protocol_word_count(row, transcript),
        starttime_word=_get_list(transcript, "starttime_word"),
        endtime_word=_get_list(transcript, "endtime_word"),
        duration_word=_get_list(transcript, "duration_word"),
        content_syllable=_get_str(transcript, "content_syllable"),
        syllable_count=_get_int(transcript, "syllable_count"),
        starttime_syllable=_get_list(transcript, "starttime_syllable"),
        endtime_syllable=_get_list(transcript, "endtime_syllable"),
        duration_syllable=_get_list(transcript, "duration_syllable"),
        content_phoneme=_get_str(transcript, "content_phoneme"),
        phoneme_count=_get_int(transcript, "phoneme_count"),
        starttime_phoneme=_get_list(transcript, "starttime_phoneme"),
        endtime_phoneme=_get_list(transcript, "endtime_phoneme"),
        duration_phoneme=_get_list(transcript, "duration_phoneme"),
        content_vowel=_get_str(transcript, "content_vowel"),
        vowel_count=_get_int(transcript, "vowel_count"),
        starttime_vowel=_get_list(transcript, "starttime_vowel"),
        endtime_vowel=_get_list(transcript, "endtime_vowel"),
        duration_vowel=_get_list(transcript, "duration_vowel"),
        starttime_consonant=_get_list(transcript, "starttime_consonant"),
        endtime_consonant=_get_list(transcript, "endtime_consonant"),
        duration_consonant=_get_list(transcript, "duration_consonant"),
        devi_mu_syllable=_get_list(transcript, "devi_mu_syllable"),
        mu_diff_syllable=_get_list(transcript, "mu_diff_syllable"),
        devi_mu_vowel=_get_list(transcript, "devi_mu_vowel"),
        mu_diff_vowel=_get_list(transcript, "mu_diff_vowel"),
        devi_mu_consonant=_get_list(transcript, "devi_mu_consonant"),
        mu_diff_consonant=_get_list(transcript, "mu_diff_consonant"),
        filepath=str(audio_dir / filename),
    )


def _output_path_for_limit(output_path: Path, limit: int | None) -> Path:
    if limit is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_first{limit}{output_path.suffix}")


def _uses_fixed_4_decimal_format(key: str) -> bool:
    return key.startswith(("duration_", "devi_mu_", "mu_diff_"))


def _csv_value(key: str, value: object) -> object:
    if isinstance(value, list):
        if _uses_fixed_4_decimal_format(key):
            return ", ".join(
                f"{item:.4f}" if isinstance(item, int | float) else str(item)
                for item in value
            )
        return ", ".join(str(item) for item in value)
    if _uses_fixed_4_decimal_format(key) and isinstance(value, int | float):
        return f"{value:.4f}"
    return value


def write_dataset_csv(items: list[DatasetItem], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(DatasetItem)]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for item in items:
            writer.writerow({key: _csv_value(key, value) for key, value in asdict(item).items()})


def print_dataset_preview(items: list[DatasetItem], limit: int = 10) -> None:
    print(f"\nPreview first {min(limit, len(items))} item(s):")
    for index, item in enumerate(items[:limit], start=1):
        sentence = item.content_sentence[:90]
        if len(item.content_sentence) > 90:
            sentence = f"{sentence}..."
        print(
            f"{index:02d}. {item.flac_file_name} label={item.label} "
            f"words={item.word_count} syllables={item.syllable_count} "
            f"phonemes={item.phoneme_count} vowels={item.vowel_count} "
            f"sentence={sentence!r}"
        )


def preprocess_split(
    data_root: Path,
    transcript_root: Path,
    output_dir: Path,
    split: SplitName,
    limit: int | None = None,
    export_csv: bool = False,
    require_transcript: bool = False,
    print_preview: bool = False,
    syllable_tier_name: str = "syllables",
) -> list[DatasetItem]:
    config = SPLIT_CONFIG[split]
    protocol_path = get_protocol_path(data_root, split)
    audio_dir = data_root / config.audio_dir
    output_path = _output_path_for_limit(output_dir / config.output, limit)

    if not protocol_path.exists():
        print(f"Error: protocol file not found: {protocol_path}")
        return []

    transcript_path = split_transcript_path(transcript_root, split)
    available_stems = set(textgrid_path_map(transcript_path))
    rows = read_protocol_rows(
        protocol_path,
        limit=limit,
        available_stems=available_stems,
        require_transcript=require_transcript,
    )
    transcript_data = read_transcripts(
        transcript_path,
        {row["FileStem"] for row in rows},
        syllable_tier_name=syllable_tier_name,
    )

    final_data: list[DatasetItem] = []
    missing_transcripts = 0
    unknown_key_count = 0
    for row in tqdm(rows, desc=f"Processing {split} protocol", unit="line"):
        if row["FileStem"] not in transcript_data:
            missing_transcripts += 1
        if row["Key"].lower() not in KEY_TO_LABEL:
            unknown_key_count += 1
        final_data.append(build_dataset_item(row, transcript_data, audio_dir))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(final_data, f)

    print(f"\n{split} preprocessing finished")
    print(f"Output file: {output_path}")
    if export_csv:
        csv_path = output_path.with_suffix(".csv")
        write_dataset_csv(final_data, csv_path)
        print(f"CSV file: {csv_path}")
    print(f"Total samples: {len(final_data)}")
    if missing_transcripts:
        print(f"Warning: {split} has {missing_transcripts} samples without TextGrid transcripts")
    if unknown_key_count:
        print(f"Warning: {split} has {unknown_key_count} samples with unknown Key; key set to -1")
    if print_preview:
        print_dataset_preview(final_data, limit=10)

    return final_data


def preprocess_asvspoof5(
    data_root: str | Path,
    transcript_root: str | Path,
    output_dir: str | Path,
    splits: tuple[SplitName, ...] = SPLITS,
    limit: int | None = None,
    export_csv: bool = False,
    require_transcript: bool = False,
    print_preview: bool = False,
    syllable_tier_name: str = "syllables",
) -> dict[SplitName, list[DatasetItem]]:
    results: dict[SplitName, list[DatasetItem]] = {}
    for split in splits:
        results[split] = preprocess_split(
            Path(data_root),
            Path(transcript_root),
            Path(output_dir),
            split,
            limit=limit,
            export_csv=export_csv,
            require_transcript=require_transcript,
            print_preview=print_preview,
            syllable_tier_name=syllable_tier_name,
        )

    total = sum(len(items) for items in results.values())
    print("\nASVspoof5 preprocessing finished")
    print(f"Total samples: {total}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add a syllable tier to ASVspoof5 TextGrid transcripts or build "
            "DatasetItem pickle/CSV files from syllabified TextGrid transcripts."
        )
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Build DatasetItem pkl/csv files from ASVspoof5 protocols and TextGrid transcripts.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_TRANSCRIPT_ROOT,
        help="Input TextGrid file or transcript root directory for syllabification mode.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="ASVspoof5 dataset root for preprocessing mode.",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Syllabified TextGrid transcript root for preprocessing mode.",
    )
    parser.add_argument(
        "--output",
        "--output_dir",
        "-o",
        dest="output",
        type=Path,
        default=None,
        help=(
            "Syllabification output file/root, or preprocessing output directory. "
            "Defaults to the syllabified transcript root for syllabification mode "
            "and the ASVspoof5 dataset root for preprocessing mode."
        ),
    )
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="ASVspoof5 split to process.",
    )
    parser.add_argument(
        "--tier_name",
        default="syllables",
        help="Name of the generated/read syllable tier.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N TextGrid files or protocol rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing to the same path as the input TextGrid in syllabification mode.",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Continue syllabification when one TextGrid cannot be parsed.",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Also export DatasetItem rows to CSV in preprocessing mode.",
    )
    parser.add_argument(
        "--require_transcript",
        action="store_true",
        help="Skip protocol rows without matching TextGrid transcripts in preprocessing mode.",
    )
    parser.add_argument(
        "--print_preview",
        action="store_true",
        help="Print a compact preview of the first 10 DatasetItem rows.",
    )
    parser.add_argument(
        "--test_first_10",
        action="store_true",
        help=(
            "Preprocessing smoke test: output the first 10 rows with matching "
            "TextGrid transcripts, export CSV, and print a compact preview."
        ),
    )
    args = parser.parse_args()

    splits = _selected_splits(args.split)
    if args.preprocess or args.test_first_10:
        output_dir = args.output if args.output is not None else DEFAULT_DATA_ROOT
        limit = 10 if args.test_first_10 else args.limit
        preprocess_asvspoof5(
            args.data_root,
            args.transcript,
            output_dir,
            splits=splits,
            limit=limit,
            export_csv=args.export_csv or args.test_first_10,
            require_transcript=args.require_transcript or args.test_first_10,
            print_preview=args.print_preview or args.test_first_10,
            syllable_tier_name=args.tier_name,
        )
        return

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if input_path.is_file():
        output_path = resolve_single_output_path(input_path, args.output)
        written_path = syllabify_textgrid_file(
            input_path,
            output_path,
            tier_name=args.tier_name,
            overwrite=args.overwrite,
        )
        print(f"Wrote {written_path}")
        return

    output_root = args.output if args.output is not None else DEFAULT_OUTPUT_ROOT
    written, failures = syllabify_textgrid_directory(
        input_path,
        output_root,
        splits=splits,
        limit=args.limit,
        tier_name=args.tier_name,
        overwrite=args.overwrite,
        skip_errors=args.skip_errors,
    )
    print(f"Wrote {len(written)} TextGrid file(s) to {output_root}")
    if failures:
        print(f"Skipped {len(failures)} TextGrid file(s)")


if __name__ == "__main__":
    main()
