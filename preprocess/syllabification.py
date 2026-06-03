import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.ASVspoof5 import (
    CONSONANT_PHONES,
    KEY_TO_LABEL,
    SPLIT_CONFIG,
    SPLITS,
    SplitName,
    get_phoneme,
    get_protocol_path,
    is_consonant,
    is_vowel,
    parse_protocol_row,
    read_textgrid_intervals,
)

Syllables: TypeAlias = list[str]
Lexicon: TypeAlias = dict[str, list[Syllables]]

WORD_PHONE_SEPARATOR = " | "
SYLLABLE_SEPARATOR = " - "
PHONE_SEPARATOR = " "
PHONE_ASSIGNMENT_TOLERANCE = 0.02

OUTPUT_FILES: dict[SplitName, str] = {
    "train": "ASVspoof5_train_syllabification.csv",
    "dev": "ASVspoof5_dev_syllabification.csv",
    "eval": "ASVspoof5_eval_syllabification.csv",
}

LEGAL_ONSETS = {
    ("B", "L"),
    ("B", "R"),
    ("B", "Y"),
    ("D", "R"),
    ("D", "W"),
    ("D", "Y"),
    ("F", "L"),
    ("F", "R"),
    ("G", "L"),
    ("G", "R"),
    ("G", "W"),
    ("K", "L"),
    ("K", "R"),
    ("K", "W"),
    ("K", "Y"),
    ("P", "L"),
    ("P", "R"),
    ("P", "Y"),
    ("S", "K"),
    ("S", "L"),
    ("S", "M"),
    ("S", "N"),
    ("S", "P"),
    ("S", "T"),
    ("S", "W"),
    ("SH", "R"),
    ("T", "R"),
    ("T", "W"),
    ("T", "Y"),
    ("TH", "R"),
    ("V", "L"),
    ("V", "R"),
    ("Z", "W"),
    ("S", "K", "L"),
    ("S", "K", "R"),
    ("S", "K", "W"),
    ("S", "P", "L"),
    ("S", "P", "R"),
    ("S", "T", "R"),
}

CSV_FIELD_NAMES = [
    "split",
    "speaker_id",
    "flac_file_name",
    "speaker_gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
    "label",
    "num_word",
    "content_sentence",
    "word_phones",
    "bartlett_svm_syllable_count",
    "bartlett_svm_syllables",
    "bartlett_svm_missing_words",
    "syllabifyr_syllable_count",
    "syllabifyr_syllables",
    "vowel_per_syllable_count",
    "vowel_per_syllable_syllables",
    "manual_lexicon_syllable_count",
    "manual_lexicon_syllables",
    "manual_lexicon_missing_words",
    "notes",
    "filepath",
]


@dataclass(frozen=True)
class WordPronunciation:
    word: str
    normalized_word: str
    phones: list[str]


@dataclass(frozen=True)
class MethodResult:
    syllable_count: int
    syllables: str
    missing_words: str = ""
    note: str = ""


def _non_empty_intervals(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    return [(start, end, text.strip()) for start, end, text in intervals if text.strip()]


def _output_path_for_limit(output_path: Path, limit: int | None) -> Path:
    if limit is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_first{limit}{output_path.suffix}")


def normalize_word(word: str) -> str:
    """Normalize transcript words for lexicon lookup."""
    word = word.upper().strip()
    word = re.sub(r"^\W+|\W+$", "", word)
    word = re.sub(r"[^A-Z0-9']", "", word)
    return word


def normalize_phones(phones: list[str]) -> list[str]:
    """Normalize ARPAbet phones and drop non-phone labels such as SPN."""
    normalized = []
    for phone in phones:
        clean_phone = get_phoneme(phone)
        if is_vowel(clean_phone) or is_consonant(clean_phone):
            normalized.append(clean_phone)
    return normalized


def _phones_from_interval_text(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    phones = []
    for start, end, phone in intervals:
        clean_phone = get_phoneme(phone)
        if is_vowel(clean_phone) or is_consonant(clean_phone):
            phones.append((start, end, clean_phone))
    return phones


def _assign_phones_to_words(
    word_intervals: list[tuple[float, float, str]],
    phone_intervals: list[tuple[float, float, str]],
) -> list[WordPronunciation]:
    word_pronunciations: list[WordPronunciation] = []
    phone_cursor = 0

    for word_start, word_end, word in word_intervals:
        while (
            phone_cursor < len(phone_intervals)
            and phone_intervals[phone_cursor][1] < word_start - PHONE_ASSIGNMENT_TOLERANCE
        ):
            phone_cursor += 1

        assigned_phones: list[str] = []
        phone_index = phone_cursor
        while (
            phone_index < len(phone_intervals)
            and phone_intervals[phone_index][0] <= word_end + PHONE_ASSIGNMENT_TOLERANCE
        ):
            phone_start, phone_end, phone = phone_intervals[phone_index]
            midpoint = (phone_start + phone_end) / 2
            if word_start - PHONE_ASSIGNMENT_TOLERANCE <= midpoint <= word_end + PHONE_ASSIGNMENT_TOLERANCE:
                assigned_phones.append(phone)
            phone_index += 1

        word_pronunciations.append(
            WordPronunciation(
                word=word,
                normalized_word=normalize_word(word),
                phones=assigned_phones,
            )
        )

    return word_pronunciations


def read_word_pronunciations(
    transcript_path: Path,
    file_stems: set[str] | None = None,
) -> dict[str, list[WordPronunciation]]:
    """Read TextGrid transcripts and align word intervals to phone intervals."""
    transcripts: dict[str, list[WordPronunciation]] = {}
    if not transcript_path.exists():
        print(f"Warning: transcript path not found: {transcript_path}")
        return transcripts

    textgrid_paths = sorted(transcript_path.rglob("*.TextGrid"))
    if file_stems is not None:
        textgrid_paths = [txt_path for txt_path in textgrid_paths if txt_path.stem in file_stems]

    for txt_path in tqdm(textgrid_paths, desc=f"Reading TextGrid {transcript_path.name}", unit="file"):
        tiers = read_textgrid_intervals(txt_path)
        word_intervals = _non_empty_intervals(tiers.get("words", []))
        phone_intervals = _phones_from_interval_text(_non_empty_intervals(tiers.get("phones", [])))
        transcripts[txt_path.stem] = _assign_phones_to_words(word_intervals, phone_intervals)

    return transcripts


def _format_syllables(syllables: Syllables) -> str:
    return SYLLABLE_SEPARATOR.join(syllable for syllable in syllables if syllable)


def _format_word_syllables(results: list[tuple[str, Syllables]]) -> str:
    formatted = []
    for word, syllables in results:
        if syllables:
            formatted.append(f"{word}={_format_syllables(syllables)}")
    return WORD_PHONE_SEPARATOR.join(formatted)


def _format_word_phones(word_pronunciations: list[WordPronunciation]) -> str:
    return WORD_PHONE_SEPARATOR.join(
        f"{word.word}={PHONE_SEPARATOR.join(word.phones)}" for word in word_pronunciations
    )


def _is_valid_onset(cluster: tuple[str, ...]) -> bool:
    if not cluster:
        return True
    if len(cluster) == 1:
        return cluster[0] in CONSONANT_PHONES and cluster[0] != "NG"
    return cluster in LEGAL_ONSETS


def syllabify_by_maximal_onset(phones: list[str]) -> list[list[str]]:
    """Approximate syllabifyr/Kyle Gorman-style ARPAbet syllabification."""
    phones = normalize_phones(phones)
    vowel_indices = [index for index, phone in enumerate(phones) if is_vowel(phone)]
    if not phones:
        return []
    if not vowel_indices:
        return [phones]

    boundaries = [0]
    for left_vowel, right_vowel in zip(vowel_indices, vowel_indices[1:]):
        cluster_start = left_vowel + 1
        cluster_end = right_vowel
        cluster = phones[cluster_start:cluster_end]
        onset_start = cluster_end

        for offset in range(len(cluster) + 1):
            candidate = tuple(cluster[offset:])
            if _is_valid_onset(candidate):
                onset_start = cluster_start + offset
                break

        if onset_start > boundaries[-1]:
            boundaries.append(onset_start)

    boundaries.append(len(phones))
    syllables = []
    for start, end in zip(boundaries, boundaries[1:]):
        syllable = phones[start:end]
        if syllable:
            syllables.append(syllable)
    return syllables


def syllabify_each_vowel(phones: list[str]) -> list[list[str]]:
    """Very simple baseline: each vowel nucleus is one syllable."""
    return [[phone] for phone in normalize_phones(phones) if is_vowel(phone)]


def _phone_groups_to_syllables(phone_groups: list[list[str]]) -> Syllables:
    return [PHONE_SEPARATOR.join(group) for group in phone_groups if group]


def _method_from_phone_rule(
    word_pronunciations: list[WordPronunciation],
    rule,
    no_phone_note: str,
) -> MethodResult:
    word_results: list[tuple[str, Syllables]] = []
    missing_words = []
    count = 0

    for word in word_pronunciations:
        if not word.phones:
            missing_words.append(word.word)
            continue
        syllables = _phone_groups_to_syllables(rule(word.phones))
        count += len(syllables)
        word_results.append((word.word, syllables))

    note = no_phone_note if missing_words else ""
    return MethodResult(
        syllable_count=count,
        syllables=_format_word_syllables(word_results),
        missing_words=", ".join(missing_words),
        note=note,
    )


def _parse_lexicon_syllables(spec: str) -> Syllables:
    spec = re.sub(r"\s+", " ", spec.strip())
    if not spec:
        return []

    if re.search(r"[.\-|/]", spec):
        parts = re.split(r"\s*(?:[.\-|/]+)\s*", spec)
        return [part.strip() for part in parts if part.strip()]

    return [spec]


def _split_lexicon_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith(";;;"):
        return None

    if "\t" in line:
        word, spec = line.split("\t", 1)
    elif "  " in line:
        word, spec = re.split(r"\s{2,}", line, maxsplit=1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return None
        word, spec = parts

    word = re.sub(r"\(\d+\)$", "", word.strip())
    spec = spec.strip()
    spec_parts = spec.split(maxsplit=1)
    if len(spec_parts) == 2 and spec_parts[0].isdigit():
        spec = spec_parts[1]

    normalized_word = normalize_word(word)
    if not normalized_word or not spec:
        return None
    return normalized_word, spec


def _read_csv_lexicon(path: Path) -> Lexicon:
    lexicon: Lexicon = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                lowered = {key.lower(): value for key, value in row.items() if key is not None}
                word = lowered.get("word") or lowered.get("entry") or lowered.get("token")
                spec = (
                    lowered.get("syllables")
                    or lowered.get("syllabification")
                    or lowered.get("pronunciation")
                )
                if not word or not spec:
                    continue
                normalized_word = normalize_word(word)
                syllables = _parse_lexicon_syllables(spec)
                if normalized_word and syllables:
                    lexicon.setdefault(normalized_word, []).append(syllables)
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                normalized_word = normalize_word(row[0])
                syllables = _parse_lexicon_syllables(row[1])
                if normalized_word and syllables:
                    lexicon.setdefault(normalized_word, []).append(syllables)
    return lexicon


def load_syllable_lexicon(path: Path | None) -> Lexicon:
    """Load a syllabified lexicon from CSV, TSV, or CMUDict-like text."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Syllable lexicon not found: {path}")

    if path.suffix.lower() == ".csv":
        return _read_csv_lexicon(path)

    lexicon: Lexicon = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _split_lexicon_line(line)
            if parsed is None:
                continue
            normalized_word, spec = parsed
            syllables = _parse_lexicon_syllables(spec)
            if syllables:
                lexicon.setdefault(normalized_word, []).append(syllables)
    return lexicon


def _candidate_phone_sequence(syllables: Syllables) -> list[str]:
    phones = []
    for syllable in syllables:
        for token in syllable.split():
            phone = get_phoneme(token)
            if is_vowel(phone) or is_consonant(phone):
                phones.append(phone)
    return phones


def _lookup_lexicon_syllables(word: WordPronunciation, lexicon: Lexicon) -> Syllables | None:
    candidates = lexicon.get(word.normalized_word)
    if not candidates:
        return None

    word_phones = normalize_phones(word.phones)
    if word_phones:
        for candidate in candidates:
            candidate_phones = _candidate_phone_sequence(candidate)
            if candidate_phones and candidate_phones == word_phones:
                return candidate

    return candidates[0]


def _method_from_lexicon(
    word_pronunciations: list[WordPronunciation],
    lexicon: Lexicon,
    missing_lexicon_note: str,
) -> MethodResult:
    if not lexicon:
        return MethodResult(
            syllable_count=0,
            syllables="",
            missing_words="",
            note=missing_lexicon_note,
        )

    word_results: list[tuple[str, Syllables]] = []
    missing_words = []
    count = 0
    for word in word_pronunciations:
        syllables = _lookup_lexicon_syllables(word, lexicon)
        if syllables is None:
            missing_words.append(word.word)
            continue
        count += len(syllables)
        word_results.append((word.word, syllables))

    note = "missing words in lexicon" if missing_words else ""
    return MethodResult(
        syllable_count=count,
        syllables=_format_word_syllables(word_results),
        missing_words=", ".join(missing_words),
        note=note,
    )


def build_syllabification_row(
    split: SplitName,
    protocol_row: dict[str, str],
    audio_dir: Path,
    word_pronunciations: list[WordPronunciation],
    bartlett_lexicon: Lexicon,
    manual_lexicon: Lexicon,
) -> dict[str, object]:
    filename = f"{protocol_row['FileStem']}.flac"
    label = KEY_TO_LABEL.get(protocol_row["Key"].lower(), -1)
    sentence = " ".join(word.word for word in word_pronunciations)
    bartlett = _method_from_lexicon(
        word_pronunciations,
        bartlett_lexicon,
        "bartlett lexicon not provided",
    )
    syllabifyr = _method_from_phone_rule(
        word_pronunciations,
        syllabify_by_maximal_onset,
        "missing phone alignment",
    )
    vowel_per_syllable = _method_from_phone_rule(
        word_pronunciations,
        syllabify_each_vowel,
        "missing phone alignment",
    )
    manual = _method_from_lexicon(
        word_pronunciations,
        manual_lexicon,
        "manual lexicon not provided",
    )

    notes = []
    for note in (bartlett.note, syllabifyr.note, vowel_per_syllable.note, manual.note):
        if note and note not in notes:
            notes.append(note)
    if not word_pronunciations:
        notes.append("TextGrid transcript missing or empty")

    return {
        "split": split,
        "speaker_id": protocol_row["SpeakerID"],
        "flac_file_name": filename,
        "speaker_gender": protocol_row["SpeakerGender"],
        "codec": protocol_row["Codec"],
        "codec_q": protocol_row["CodecQ"],
        "codec_seed": protocol_row["CodecSeed"],
        "attack_tag": protocol_row["AttackTag"],
        "attack_label": protocol_row["AttackLabel"],
        "key": protocol_row["Key"],
        "label": label,
        "num_word": len(word_pronunciations),
        "content_sentence": sentence,
        "word_phones": _format_word_phones(word_pronunciations),
        "bartlett_svm_syllable_count": bartlett.syllable_count,
        "bartlett_svm_syllables": bartlett.syllables,
        "bartlett_svm_missing_words": bartlett.missing_words,
        "syllabifyr_syllable_count": syllabifyr.syllable_count,
        "syllabifyr_syllables": syllabifyr.syllables,
        "vowel_per_syllable_count": vowel_per_syllable.syllable_count,
        "vowel_per_syllable_syllables": vowel_per_syllable.syllables,
        "manual_lexicon_syllable_count": manual.syllable_count,
        "manual_lexicon_syllables": manual.syllables,
        "manual_lexicon_missing_words": manual.missing_words,
        "notes": "; ".join(notes),
        "filepath": str(audio_dir / filename),
    }


def write_syllabification_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_protocol_rows(
    protocol_path: Path,
    limit: int | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with protocol_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            row = parse_protocol_row(line, protocol_path, line_no)
            if row is None:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def syllabify_split(
    data_root: Path,
    transcript_root: Path,
    output_dir: Path,
    split: SplitName,
    bartlett_lexicon: Lexicon,
    manual_lexicon: Lexicon,
    limit: int | None = None,
) -> list[dict[str, object]]:
    config = SPLIT_CONFIG[split]
    protocol_path = get_protocol_path(data_root, split)
    audio_dir = data_root / config.audio_dir
    output_path = _output_path_for_limit(output_dir / OUTPUT_FILES[split], limit)

    if not protocol_path.exists():
        print(f"Error: protocol file not found: {protocol_path}")
        return []

    rows = read_protocol_rows(protocol_path, limit=limit)
    split_transcript_root = transcript_root / split
    transcript_path = split_transcript_root if split_transcript_root.exists() else transcript_root
    transcripts = read_word_pronunciations(transcript_path, {row["FileStem"] for row in rows})

    csv_rows: list[dict[str, object]] = []
    missing_transcripts = 0
    for row in tqdm(rows, desc=f"Syllabifying {split} protocol", unit="line"):
        word_pronunciations = transcripts.get(row["FileStem"], [])
        if not word_pronunciations:
            missing_transcripts += 1
        csv_rows.append(
            build_syllabification_row(
                split,
                row,
                audio_dir,
                word_pronunciations,
                bartlett_lexicon,
                manual_lexicon,
            )
        )

    write_syllabification_csv(csv_rows, output_path)
    print(f"\n{split} syllabification finished")
    print(f"CSV file: {output_path}")
    print(f"Total samples: {len(csv_rows)}")
    if missing_transcripts:
        print(f"Warning: {split} has {missing_transcripts} samples without TextGrid transcripts")

    return csv_rows


def syllabify_asvspoof5(
    data_root: str | Path,
    transcript_root: str | Path,
    output_dir: str | Path,
    splits: tuple[SplitName, ...] = SPLITS,
    bartlett_lexicon_path: str | Path | None = None,
    manual_lexicon_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[SplitName, list[dict[str, object]]]:
    root_path = Path(data_root)
    transcript_path = Path(transcript_root)
    out_path = Path(output_dir)
    bartlett_lexicon = load_syllable_lexicon(
        Path(bartlett_lexicon_path) if bartlett_lexicon_path is not None else None
    )
    manual_lexicon = load_syllable_lexicon(
        Path(manual_lexicon_path) if manual_lexicon_path is not None else None
    )

    if bartlett_lexicon_path is not None:
        print(f"Bartlett lexicon entries: {len(bartlett_lexicon)}")
    else:
        print("Bartlett lexicon not provided; Bartlett columns will be empty.")
    if manual_lexicon_path is not None:
        print(f"Manual lexicon entries: {len(manual_lexicon)}")
    else:
        print("Manual lexicon not provided; manual lexicon columns will be empty.")

    results: dict[SplitName, list[dict[str, object]]] = {}
    for split in splits:
        results[split] = syllabify_split(
            root_path,
            transcript_path,
            out_path,
            split,
            bartlett_lexicon,
            manual_lexicon,
            limit=limit,
        )

    total = sum(len(rows) for rows in results.values())
    print("\nASVspoof5 syllabification finished")
    print(f"Total samples: {total}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ASVspoof5 syllabification CSV files from TextGrid transcripts and TSV protocols"
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
        default=Path("dataset/ASVspoof5/ASVspoof5_transcript"),
        help="ASVspoof5 transcript root directory",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=Path("dataset/ASVspoof5"),
        help="Output CSV directory",
    )
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Split to syllabify. Use 'all' to process every split.",
    )
    parser.add_argument(
        "--bartlett_lexicon",
        type=Path,
        default=None,
        help="Optional Bartlett/Kondrak syllabified CMU dictionary path, e.g. cmudict.0.6d.syl.",
    )
    parser.add_argument(
        "--manual_lexicon",
        type=Path,
        default=None,
        help="Optional hand-syllabified lexicon path in CSV, TSV, or CMUDict-like text format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N valid protocol rows per selected split.",
    )
    parser.add_argument(
        "--test_first_10",
        action="store_true",
        help="Small smoke test: process first 10 rows with _first10 suffix.",
    )
    args = parser.parse_args()

    limit = 10 if args.test_first_10 else args.limit
    splits = SPLITS if args.split == "all" else (args.split,)
    syllabify_asvspoof5(
        args.input,
        args.transcript,
        args.output_dir,
        splits=splits,
        bartlett_lexicon_path=args.bartlett_lexicon,
        manual_lexicon_path=args.manual_lexicon,
        limit=limit,
    )


if __name__ == "__main__":
    main()
