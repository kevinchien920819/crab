import argparse
import csv
import pickle
import re
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal, TypeAlias
from textgrid import read_textgrid

from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.dataclass import DatasetItem, SplitConfig

SplitName: TypeAlias = Literal["train", "dev", "eval"]
Interval: TypeAlias = tuple[float, float, str]
TranscriptData: TypeAlias = dict[str, str | float | list[float]]

SPLITS: tuple[SplitName, ...] = ("train", "dev", "eval")

KEY_TO_LABEL = {
    "spoof": 0,
    "bonafide": 1,
}

# Protocol columns based on ASVspoof5 protocol documentation
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

# Configuration for each split, including protocol file, audio directory, and output file name
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

# Phoneme sets for vowel and consonant phones based on ARPAbet phoneme set
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


def get_phoneme(phone_label: str) -> str:
    """Normalize an ARPAbet phone label by removing lexical stress markers.

    Args:
        phone_label: Raw phone label from a TextGrid phone interval. The label
            may include a trailing stress marker such as ``0``, ``1``, or ``2``.

    Returns:
        Uppercase phone label without trailing stress digits. For example,
        ``IY1`` becomes ``IY``. Empty input returns an empty string.
    """
    phone_label = phone_label.strip().upper()
    if not phone_label:
        return ""

    # Remove lexical stress marker at the end of the phone label.
    return re.sub(r"\d+$", "", phone_label)


def is_vowel(phone: str) -> bool:
    """Check whether a phone belongs to the ARPAbet vowel set.

    Args:
        phone: Raw or normalized phone label. Stress markers are accepted and
            removed before lookup.

    Returns:
        ``True`` if the phone is a vowel, otherwise ``False``.
    """
    return get_phoneme(phone) in VOWEL_PHONES


def is_consonant(phone: str) -> bool:
    """Check whether a phone belongs to the ARPAbet consonant set.

    Args:
        phone: Raw or normalized phone label. Stress markers are accepted and
            removed before lookup.

    Returns:
        ``True`` if the phone is a consonant, otherwise ``False``.
    """
    return get_phoneme(phone) in CONSONANT_PHONES


def get_protocol_path(data_root: Path, split: SplitName) -> Path:
    """Build the ASVspoof5 protocol path for one split.

    Args:
        data_root: Root directory of the ASVspoof5 dataset.
        split: Dataset split name, one of ``train``, ``dev``, or ``eval``.

    Returns:
        Absolute or relative path to the split protocol TSV file, depending on
        how ``data_root`` was provided.
    """
    protocol_rel_path = SPLIT_CONFIG[split].protocols
    return data_root / protocol_rel_path


def parse_protocol_row(line: str, protocol_path: Path, line_no: int) -> dict[str, str] | None:
    """Parse and validate one ASVspoof5 protocol line.

    Args:
        line: Raw line read from the protocol TSV file.
        protocol_path: Path to the protocol file, used only for warning output.
        line_no: One-based line number, used only for warning output.

    Returns:
        Dictionary mapping ``PROTOCOL_COLUMNS`` to field values. Empty lines or
        malformed rows return ``None``.
    """
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


def read_textgrid_intervals(textgrid_path: Path) -> dict[str, list[Interval]]:
    """Read interval tiers from a Praat TextGrid file via ``read_textgrid``.

    Args:
        textgrid_path: Path to one TextGrid transcript file.

    Returns:
        Mapping from lowercased tier name to interval tuples. Each interval is
        ``(start_time, end_time, text)`` in seconds. Empty interval labels are
        preserved here and filtered later by callers.
    """
    tiers: dict[str, list[Interval]] = {}
    textgrid = read_textgrid(textgrid_path)

    for item in textgrid.items:
        tier_name = item.name.strip().lower()
        tiers.setdefault(tier_name, []).extend(
            (interval.xmin, interval.xmax, interval.text)
            for interval in item.intervals
        )

    return tiers


def _non_empty_intervals(intervals: list[Interval]) -> list[Interval]:
    """Remove intervals whose text label is empty.

    Args:
        intervals: TextGrid intervals represented as ``(start, end, text)``.

    Returns:
        Intervals with non-empty stripped labels. Returned labels are stripped.
    """
    return [(start, end, text.strip()) for start, end, text in intervals if text.strip()]


def _content_text(intervals: list[Interval]) -> str:
    """Format interval labels as comma-separated content text.

    Args:
        intervals: TextGrid intervals represented as ``(start, end, text)``.

    Returns:
        Comma-separated interval labels, suitable for CSV fields such as
        phoneme, vowel, and consonant content.
    """
    return ", ".join(text for _, _, text in intervals)


def _start_times(intervals: list[Interval]) -> list[float]:
    """Extract start times from intervals.

    Args:
        intervals: TextGrid intervals represented as ``(start, end, text)``.

    Returns:
        List of interval start times in seconds.
    """
    return [start for start, _, _ in intervals]


def _end_times(intervals: list[Interval]) -> list[float]:
    """Extract end times from intervals.

    Args:
        intervals: TextGrid intervals represented as ``(start, end, text)``.

    Returns:
        List of interval end times in seconds.
    """
    return [end for _, end, _ in intervals]


def _durations(intervals: list[Interval]) -> list[float]:
    """Compute durations for intervals.

    Args:
        intervals: TextGrid intervals represented as ``(start, end, text)``.

    Returns:
        List of non-negative interval durations in seconds.
    """
    return [max(0.0, end - start) for start, end, _ in intervals]


def _mean_abs_deviation(values: list[float]) -> float:
    """Compute the mean absolute deviation from the sequence mean.

    Args:
        values: Numeric values, typically per-phone or per-word durations.

    Returns:
        Mean absolute deviation. Empty input returns ``0.0``.
    """
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum(abs(value - mean_value) for value in values) / len(values)


def _mean_relative_neighbor_diff(values: list[float]) -> float:
    """Compute average relative difference between neighboring values.

    Args:
        values: Numeric values ordered by time, typically duration values.

    Returns:
        Mean absolute relative difference between adjacent values, using the
        pair mean as denominator. Inputs with fewer than two values return
        ``0.0``.
    """
    if len(values) < 2:
        return 0.0

    diffs: list[float] = []
    for left, right in zip(values, values[1:]):
        denominator = (left + right) / 2
        if denominator == 0:
            continue
        diffs.append(abs((left - right) / denominator))
    return sum(diffs) / len(diffs) if diffs else 0.0


def _sentence_timing(word_intervals: list[Interval]) -> tuple[list[float], list[float], float]:
    """Derive utterance-level timing from word intervals.

    Args:
        word_intervals: Non-empty word intervals from a TextGrid ``words`` tier.

    Returns:
        A tuple of ``(start_times, end_times, duration)``. Start and end times
        are single-item lists to match ``DatasetItem`` field types. Empty input
        returns ``([], [], 0.0)``.
    """
    if not word_intervals:
        return [], [], 0.0

    start = word_intervals[0][0]
    end = word_intervals[-1][1]
    return [start], [end], max(0.0, end - start)


def read_transcript(
    transcript_path: Path,
    file_stems: set[str] | None = None,
) -> dict[str, TranscriptData]:
    """Read TextGrid transcripts and convert them into model-ready metadata.

    Args:
        transcript_path: Directory containing TextGrid files. The directory may
            be a split directory such as ``train`` or a root containing nested
            split directories.
        file_stems: Optional set of ASVspoof5 file stems to keep. When provided,
            only matching TextGrid files are parsed.

    Returns:
        Mapping from file stem to transcript metadata. Metadata includes the
        sentence text, full phoneme content, word timing, vowel/consonant
        counts and content, syllable proxy timing, and duration-derived
        statistics.
    """
    transcripts: dict[str, TranscriptData] = {}
    if not transcript_path.exists():
        print(f"Warning: transcript path not found: {transcript_path}")
        return transcripts

    textgrid_paths = sorted(transcript_path.rglob("*.TextGrid"))
    if file_stems is not None:
        textgrid_paths = [txt_path for txt_path in textgrid_paths if txt_path.stem in file_stems]

    for txt_path in tqdm(textgrid_paths, desc=f"Reading TextGrid {transcript_path.name}", unit="file"):
        file_stem = txt_path.stem
        tiers = read_textgrid_intervals(txt_path)

        word_intervals = _non_empty_intervals(tiers.get("words", []))
        phone_intervals = _non_empty_intervals(tiers.get("phones", []))
        phoneme_intervals = [
            (start, end, get_phoneme(phone))
            for start, end, phone in phone_intervals
        ]
        vowel_intervals = [
            (start, end, phone)
            for start, end, phone in phoneme_intervals
            if is_vowel(phone)
        ]
        consonant_intervals = [
            (start, end, phone)
            for start, end, phone in phoneme_intervals
            if is_consonant(phone)
        ]

        sentence_start, sentence_end, sentence_duration = _sentence_timing(word_intervals)
        word_durations = _durations(word_intervals)
        vowel_durations = _durations(vowel_intervals)
        consonant_durations = _durations(consonant_intervals)

        # TextGrid has no explicit syllable tier; vowel nuclei are the closest available proxy.
        syllable_intervals = vowel_intervals
        syllable_durations = vowel_durations

        transcripts[file_stem] = {
            "sentence": " ".join(text for _, _, text in word_intervals),
            "starttime_sentence": sentence_start,
            "endtime_sentence": sentence_end,
            "duration_sentence": sentence_duration,
            "content_syllable": _content_text(phoneme_intervals),
            "starttime_syllable": _start_times(syllable_intervals),
            "endtime_syllable": _end_times(syllable_intervals),
            "starttime_word": _start_times(word_intervals),
            "endtime_word": _end_times(word_intervals),
            "duration_word": word_durations,
            "vowel_count": len(vowel_intervals),
            "vowel_content": _content_text(vowel_intervals),
            "starttime_vowel": _start_times(vowel_intervals),
            "endtime_vowel": _end_times(vowel_intervals),
            "duration_vowel": vowel_durations,
            "constanant_count": len(consonant_intervals),
            "constanant_content": _content_text(consonant_intervals),
            "starttime_constanant": _start_times(consonant_intervals),
            "endtime_constanant": _end_times(consonant_intervals),
            "duration_constanant": consonant_durations,
            "devi_mu_syllable": _mean_abs_deviation(syllable_durations),
            "mu_diff_syllable": _mean_relative_neighbor_diff(syllable_durations),
            "devi_mu_vowel": _mean_abs_deviation(vowel_durations),
            "mu_diff_vowel": _mean_relative_neighbor_diff(vowel_durations),
            "devi_mu_constanant": _mean_abs_deviation(consonant_durations),
            "mu_diff_constanant": _mean_relative_neighbor_diff(consonant_durations),
        }

    return transcripts


def _get_list(transcript: TranscriptData, key: str) -> list[float]:
    """Safely read a list-valued transcript field.

    Args:
        transcript: Metadata dictionary for one utterance.
        key: Field name to read from ``transcript``.

    Returns:
        The field value when it is a list, otherwise an empty list.
    """
    value = transcript.get(key, [])
    return value if isinstance(value, list) else []


def _get_float(transcript: TranscriptData, key: str) -> float:
    """Safely read a float-valued transcript field.

    Args:
        transcript: Metadata dictionary for one utterance.
        key: Field name to read from ``transcript``.

    Returns:
        The field value converted to ``float`` when numeric, otherwise ``0.0``.
    """
    value = transcript.get(key, 0.0)
    return float(value) if isinstance(value, (float, int)) else 0.0


def _get_str(transcript: TranscriptData, key: str) -> str:
    """Safely read a string-valued transcript field.

    Args:
        transcript: Metadata dictionary for one utterance.
        key: Field name to read from ``transcript``.

    Returns:
        The field value when it is a string, otherwise an empty string.
    """
    value = transcript.get(key, "")
    return value if isinstance(value, str) else ""


def _protocol_word_count(row: dict[str, str], transcript: TranscriptData) -> int:
    """Resolve the word count for one protocol row.

    Args:
        row: Parsed ASVspoof5 protocol row.
        transcript: Transcript metadata for the same file stem, if available.

    Returns:
        Number of word intervals from the transcript when available. Otherwise,
        falls back to the protocol ``Tmp`` field when it is an integer. If both
        are unavailable, returns ``0``.
    """
    word_count = len(_get_list(transcript, "starttime_word"))
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
    """Build one ASVspoof5 ``DatasetItem`` from protocol and transcript data.

    Args:
        row: Parsed protocol row with keys defined by ``PROTOCOL_COLUMNS``.
        transcript_data: Mapping from ASVspoof5 file stem to TextGrid-derived
            transcript metadata.
        audio_dir: Directory containing the split's FLAC audio files.

    Returns:
        Fully populated ``DatasetItem``. Missing transcript fields are filled
        with empty strings, empty lists, or ``0.0`` defaults. Unknown protocol
        labels are encoded as ``-1``.
    """
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
        num_word=_protocol_word_count(row, transcript),
        content_sentence=_get_str(transcript, "sentence"),
        starttime_sentence=_get_list(transcript, "starttime_sentence"),
        endtime_sentence=_get_list(transcript, "endtime_sentence"),
        duration_sentence=_get_float(transcript, "duration_sentence"),
        content_syllable=_get_str(transcript, "content_syllable"),
        starttime_syllable=_get_list(transcript, "starttime_syllable"),
        endtime_syllable=_get_list(transcript, "endtime_syllable"),
        starttime_word=_get_list(transcript, "starttime_word"),
        endtime_word=_get_list(transcript, "endtime_word"),
        duration_word=_get_list(transcript, "duration_word"),
        vowel_count=int(_get_float(transcript, "vowel_count")),
        vowel_content=_get_str(transcript, "vowel_content"),
        starttime_vowel=_get_list(transcript, "starttime_vowel"),
        endtime_vowel=_get_list(transcript, "endtime_vowel"),
        duration_vowel=_get_list(transcript, "duration_vowel"),
        constanant_count=int(_get_float(transcript, "constanant_count")),
        constanant_content=_get_str(transcript, "constanant_content"),
        starttime_constanant=_get_list(transcript, "starttime_constanant"),
        endtime_constanant=_get_list(transcript, "endtime_constanant"),
        duration_constanant=_get_list(transcript, "duration_constanant"),
        devi_mu_syllable=_get_float(transcript, "devi_mu_syllable"),
        mu_diff_syllable=_get_float(transcript, "mu_diff_syllable"),
        devi_mu_vowel=_get_float(transcript, "devi_mu_vowel"),
        mu_diff_vowel=_get_float(transcript, "mu_diff_vowel"),
        devi_mu_constanant=_get_float(transcript, "devi_mu_constanant"),
        mu_diff_constanant=_get_float(transcript, "mu_diff_constanant"),
        filepath=str(audio_dir / filename),
    )


def _output_path_for_limit(output_path: Path, limit: int | None) -> Path:
    """Add a first-N suffix to an output path when running a limited test.

    Args:
        output_path: Original output pickle path from ``SPLIT_CONFIG``.
        limit: Optional maximum number of protocol rows to process.

    Returns:
        Original path when ``limit`` is ``None``. Otherwise, returns a sibling
        path with ``_first{limit}`` appended before the file suffix.
    """
    if limit is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_first{limit}{output_path.suffix}")


def _csv_value(value: object) -> object:
    """Convert one ``DatasetItem`` field into a CSV-friendly value.

    Args:
        value: Field value from ``dataclasses.asdict``.

    Returns:
        Comma-separated string for list values, otherwise the original scalar
        value. The CSV writer will quote list strings because they contain
        commas.
    """
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def write_dataset_csv(items: list[DatasetItem], csv_path: Path) -> None:
    """Write preprocessed ``DatasetItem`` objects to a CSV file.

    Args:
        items: Preprocessed dataset items to export.
        csv_path: Destination CSV path. Parent directories are created if needed.

    Returns:
        None. The function writes a CSV file with one row per item. List-valued
        fields are stored as quoted comma-separated strings.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(DatasetItem)]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for item in items:
            row = {key: _csv_value(value) for key, value in asdict(item).items()}
            writer.writerow(row)


def preprocess_split(
    data_root: Path,
    transcript_root: Path,
    output_dir: Path,
    split: SplitName,
    limit: int | None = None,
    export_csv: bool = False,
) -> list[DatasetItem]:
    """Preprocess one ASVspoof5 split and write its pickle file.

    Args:
        data_root: Root directory of the ASVspoof5 dataset, containing protocol
            files and split audio directories.
        transcript_root: Root directory containing TextGrid transcripts. If a
            matching split subdirectory exists, it is used for that split.
        output_dir: Directory where the split pickle file will be written.
        split: Dataset split to preprocess, one of ``train``, ``dev``, or
            ``eval``.
        limit: Optional maximum number of valid protocol rows to process. When
            provided, the output pickle path gets a ``_first{limit}`` suffix.
        export_csv: Whether to also write a CSV file beside the pickle file.

    Returns:
        List of ``DatasetItem`` objects produced for the split. If the protocol
        file is missing, returns an empty list.
    """
    config = SPLIT_CONFIG[split]
    protocol_path = get_protocol_path(data_root, split)
    audio_dir = data_root / config.audio_dir
    output_path = _output_path_for_limit(output_dir / config.output, limit)

    if not protocol_path.exists():
        print(f"Error: protocol file not found: {protocol_path}")
        return []

    rows: list[dict[str, str]] = []
    with protocol_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            row = parse_protocol_row(line, protocol_path, line_no)
            if row is None:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    split_transcript_root = transcript_root / split
    transcript_path = split_transcript_root if split_transcript_root.exists() else transcript_root
    transcript_data = read_transcript(transcript_path, {row["FileStem"] for row in rows})
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

    return final_data


def preprocess_asvspoof5(
    data_root: str | Path,
    transcript_root: str | Path,
    output_dir: str | Path,
    splits: tuple[SplitName, ...] = SPLITS,
    limit: int | None = None,
    export_csv: bool = False,
) -> dict[SplitName, list[DatasetItem]]:
    """Preprocess all ASVspoof5 splits.

    Args:
        data_root: Root directory of the ASVspoof5 dataset.
        transcript_root: Root directory containing TextGrid transcripts.
        output_dir: Directory where generated pickle files are written.
        splits: Splits to preprocess. Defaults to all ASVspoof5 splits.
        limit: Optional maximum number of valid protocol rows to process per
            split.
        export_csv: Whether to also write a CSV file for each split.

    Returns:
        Dictionary keyed by split name. Each value is the list of ``DatasetItem``
        objects generated for that split.
    """
    root_path = Path(data_root)
    transcript_path = Path(transcript_root)
    out_path = Path(output_dir)

    results: dict[SplitName, list[DatasetItem]] = {}
    for split in splits:
        results[split] = preprocess_split(
            root_path,
            transcript_path,
            out_path,
            split,
            limit=limit,
            export_csv=export_csv,
        )

    total = sum(len(items) for items in results.values())
    print("\nASVspoof5 preprocessing finished")
    print(f"Total samples: {total}")

    return results


def main() -> None:
    """Parse command-line arguments and run ASVspoof5 preprocessing.

    Args:
        None.

    Returns:
        None. The function writes pickle files to ``--output_dir`` and prints a
        preprocessing summary. Use ``--test_first_10`` for a small smoke test
        that processes only the first 10 valid protocol rows and exports CSV.
    """
    parser = argparse.ArgumentParser(
        description="Build ASVspoof5 pkl files from TextGrid transcripts and TSV protocols"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("dataset/ASVspoof5"),
        help="ASVspoof5 root directory",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        type=Path,
        default=Path("dataset/ASVspoof5/ASVspoof5_syllabified_textgrid"),
        help="ASVspoof5 transcript root directory",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=Path("dataset/ASVspoof5"),
        help="Output pkl directory",
    )
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Split to preprocess. Use 'all' to preprocess every split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N valid protocol rows per selected split.",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Also export the preprocessed items to CSV.",
    )
    parser.add_argument(
        "--test_first_10",
        action="store_true",
        help="Small smoke test: process first 10 rows and export CSV with _first10 suffix.",
    )
    args = parser.parse_args()

    limit = 10 if args.test_first_10 else args.limit
    export_csv = args.export_csv or args.test_first_10
    splits = SPLITS if args.split == "all" else (args.split,)
    preprocess_asvspoof5(
        args.input,
        args.transcript,
        args.output_dir,
        splits=splits,
        limit=limit,
        export_csv=export_csv,
    )


if __name__ == "__main__":
    main()
