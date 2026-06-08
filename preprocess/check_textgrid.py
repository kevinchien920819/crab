import argparse
import csv
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this utility script.
    def tqdm(iterable, **_: object):
        return iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import textgrid
else:
    from . import textgrid


SPN_PATTERN = re.compile(r"(?<![A-Z0-9])SPN(?![A-Z0-9])", re.IGNORECASE)


def label_has_spn(label: str) -> bool:
    return SPN_PATTERN.search(label) is not None


def find_spn_intervals(path: Path) -> list[tuple[str, float, float, str]]:
    grid = textgrid.read_textgrid(path)
    matches: list[tuple[str, float, float, str]] = []
    for item in grid.items:
        for interval in item.intervals:
            if label_has_spn(interval.text):
                matches.append((item.name, interval.xmin, interval.xmax, interval.text))
    return matches


def iter_textgrid_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.TextGrid"))


def split_name(path: Path) -> str:
    for part in path.parts:
        if part in {"train", "dev", "eval"}:
            return part
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check TextGrid files for SPN interval labels."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input TextGrid file or directory.",
    )
    parser.add_argument(
        "--show_intervals",
        action="store_true",
        help="Print matched tier, time range, and label for each SPN interval.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Check only the first N TextGrid files after sorting.",
    )
    parser.add_argument(
        "--output_csv",
        "--csv",
        type=Path,
        default=None,
        help="Write one row per TextGrid file containing SPN to this CSV file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print each matched TextGrid path.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input path not found: {args.input}")

    paths = iter_textgrid_paths(args.input)
    if args.limit is not None:
        paths = paths[:args.limit]

    files_with_spn = 0
    total_matches = 0
    csv_file = None
    csv_writer = None
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.output_csv.open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "path",
                "split",
                "file_stem",
                "spn_interval_count",
                "tiers",
                "first_xmin",
                "first_xmax",
                "labels",
            ),
        )
        csv_writer.writeheader()

    try:
        for path in tqdm(paths, desc="Checking TextGrid", unit="file"):
            matches = find_spn_intervals(path)
            if not matches:
                continue

            files_with_spn += 1
            total_matches += len(matches)
            if not args.quiet:
                print(f"{path}: {len(matches)} SPN interval(s)")
            if csv_writer is not None:
                _first_tier, first_xmin, first_xmax, _first_label = min(
                    matches,
                    key=lambda match: (match[1], match[2], match[0]),
                )
                csv_writer.writerow(
                    {
                        "path": str(path),
                        "split": split_name(path),
                        "file_stem": path.stem,
                        "spn_interval_count": len(matches),
                        "tiers": ";".join(sorted({tier for tier, _, _, _ in matches})),
                        "first_xmin": f"{first_xmin:g}",
                        "first_xmax": f"{first_xmax:g}",
                        "labels": ";".join(sorted({label for _, _, _, label in matches})),
                    }
                )
            for tier, xmin, xmax, label in matches:
                if args.show_intervals and not args.quiet:
                    print(f"  {tier} {xmin:g}-{xmax:g}: {label}")
    finally:
        if csv_file is not None:
            csv_file.close()

    print(f"Checked {len(paths)} TextGrid file(s)")
    print(f"Files with SPN: {files_with_spn}")
    print(f"Total SPN intervals: {total_matches}")
    if args.output_csv is not None:
        print(f"CSV file: {args.output_csv}")

    raise SystemExit(1 if files_with_spn else 0)


if __name__ == "__main__":
    main()
