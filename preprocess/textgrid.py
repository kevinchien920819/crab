from dataclasses import dataclass
from pathlib import Path

try:
    from . import syllabify as arpabet_syllabify
except ImportError:
    import syllabify as arpabet_syllabify


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass
class Item:
    class_type: str
    name: str
    xmin: float
    xmax: float
    intervals_size: int
    intervals: list[Interval]


@dataclass
class TextGrid:
    file_type: str
    object_class: str
    xmin: float
    xmax: float
    tiers_exists: bool
    size: int
    items: list[Item]


def _split_assignment(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


def _is_indexed_block(line: str, name: str) -> bool:
    prefix = f"{name} ["
    if not line.startswith(prefix) or not line.endswith("]:"):
        return False
    index = line[len(prefix):-2]
    return index.isdigit()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise ValueError(f"Expected quoted TextGrid value, got: {value}")
    return value[1:-1].replace('""', '"')


def _escape_text(text: str) -> str:
    return text.replace('"', '""')


def _format_time(value: float) -> str:
    return f"{value:.15g}"


VOWEL_PHONES = {
    "AA", "AE", "AH", "AO", "AW",
    "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW",
}


def _phone_base(phone: str) -> str:
    phone = phone.strip().upper()
    while phone and phone[-1].isdigit():
        phone = phone[:-1]
    return phone


def _is_vowel_phone(phone: str) -> bool:
    return _phone_base(phone) in VOWEL_PHONES


def _find_item(textgrid: TextGrid, name: str) -> Item | None:
    name = name.lower()
    for item in textgrid.items:
        if item.name.lower() == name:
            return item
    return None


def _non_empty_intervals(intervals: list[Interval]) -> list[Interval]:
    return [interval for interval in intervals if interval.text.strip()]


def _interval_midpoint(interval: Interval) -> float:
    return (interval.xmin + interval.xmax) / 2


def _phones_inside_interval(
    interval: Interval,
    phones: list[Interval],
    tolerance: float = 1e-7,
) -> list[Interval]:
    return [
        phone
        for phone in phones
        if interval.xmin - tolerance <= _interval_midpoint(phone) <= interval.xmax + tolerance
    ]


def _fallback_split_phones_to_syllables(phones: list[Interval]) -> list[list[Interval]]:
    if not phones:
        return []

    vowel_indexes = [
        index
        for index, phone in enumerate(phones)
        if _is_vowel_phone(phone.text)
    ]
    if not vowel_indexes:
        return [phones]

    boundaries = [0]
    for previous_vowel_index in vowel_indexes[:-1]:
        boundaries.append(previous_vowel_index + 1)
    boundaries.append(len(phones))

    syllables: list[list[Interval]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        group = phones[start:end]
        if group:
            syllables.append(group)
    return syllables


def _split_phones_to_syllables(phones: list[Interval]) -> list[list[Interval]]:
    if not phones:
        return []
    if not any(_is_vowel_phone(phone.text) for phone in phones):
        return [phones]

    pronunciation = [phone.text.strip().upper() for phone in phones]
    try:
        parsed_syllables = arpabet_syllabify.syllabify(pronunciation)
    except ValueError:
        return _fallback_split_phones_to_syllables(phones)

    syllable_groups: list[list[Interval]] = []
    cursor = 0
    for onset, nucleus, coda in parsed_syllables:
        group_size = len(onset) + len(nucleus) + len(coda)
        if group_size <= 0:
            continue
        group = phones[cursor:cursor + group_size]
        if len(group) != group_size:
            return _fallback_split_phones_to_syllables(phones)
        syllable_groups.append(group)
        cursor += group_size

    if cursor != len(phones) or not syllable_groups:
        return _fallback_split_phones_to_syllables(phones)
    return syllable_groups


def _syllable_label(
    _word: str,
    phones: list[Interval],
    _syllable_index: int,
    _syllable_count: int,
) -> str:
    return " ".join(phone.text.strip() for phone in phones if phone.text.strip())


def _merge_empty_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in intervals:
        if (
            not interval.text
            and merged
            and not merged[-1].text
            and abs(merged[-1].xmax - interval.xmin) < 1e-9
        ):
            merged[-1] = Interval(
                xmin=merged[-1].xmin,
                xmax=interval.xmax,
                text="",
            )
        else:
            merged.append(interval)
    return merged


def _make_contiguous_intervals(
    intervals: list[Interval],
    xmin: float,
    xmax: float,
) -> list[Interval]:
    contiguous: list[Interval] = []
    cursor = xmin

    for interval in sorted(intervals, key=lambda item: (item.xmin, item.xmax)):
        start = interval.xmin
        end = interval.xmax
        if end <= start:
            continue
        if start > cursor:
            contiguous.append(Interval(xmin=cursor, xmax=start, text=""))
        if start < cursor:
            start = cursor
        if end > start:
            contiguous.append(Interval(xmin=start, xmax=end, text=interval.text))
            cursor = end

    if cursor < xmax:
        contiguous.append(Interval(xmin=cursor, xmax=xmax, text=""))

    return _merge_empty_intervals(contiguous)


def read_textgrid(path: Path) -> TextGrid:
    path = Path(path)

    file_type = ""
    object_class = ""
    xmin: float | None = None
    xmax: float | None = None
    tiers_exists = False
    size = 0
    items: list[Item] = []

    in_item = False
    item_class_type = ""
    item_name = ""
    item_xmin: float | None = None
    item_xmax: float | None = None
    item_intervals_size = 0
    item_intervals: list[Interval] = []

    in_interval = False
    interval_xmin: float | None = None
    interval_xmax: float | None = None

    def finish_item() -> None:
        nonlocal in_item
        if not in_item:
            return
        if item_xmin is None or item_xmax is None:
            raise ValueError(f"Incomplete TextGrid item in {path}")
        items.append(
            Item(
                class_type=item_class_type,
                name=item_name,
                xmin=item_xmin,
                xmax=item_xmax,
                intervals_size=item_intervals_size,
                intervals=list(item_intervals),
            )
        )
        in_item = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("tiers?"):
                tiers_exists = "<exists>" in line
                continue

            if _is_indexed_block(line, "item"):
                if in_interval:
                    raise ValueError(f"Unfinished TextGrid interval before {line!r} in {path}")
                finish_item()
                in_item = True
                item_class_type = ""
                item_name = ""
                item_xmin = None
                item_xmax = None
                item_intervals_size = 0
                item_intervals = []
                continue

            if _is_indexed_block(line, "intervals"):
                if not in_item:
                    raise ValueError(f"TextGrid interval appears outside an item in {path}")
                in_interval = True
                interval_xmin = None
                interval_xmax = None
                continue

            assignment = _split_assignment(line)
            if assignment is None:
                continue
            key, value = assignment

            if in_interval:
                if key == "xmin":
                    interval_xmin = float(value)
                elif key == "xmax":
                    interval_xmax = float(value)
                elif key == "text":
                    if interval_xmin is None or interval_xmax is None:
                        raise ValueError(f"Incomplete TextGrid interval in {path}")
                    item_intervals.append(
                        Interval(
                            xmin=interval_xmin,
                            xmax=interval_xmax,
                            text=_unquote(value),
                        )
                    )
                    in_interval = False
                continue

            if in_item:
                if key == "class":
                    item_class_type = _unquote(value)
                elif key == "name":
                    item_name = _unquote(value)
                elif key == "xmin":
                    item_xmin = float(value)
                elif key == "xmax":
                    item_xmax = float(value)
                elif key == "intervals: size":
                    item_intervals_size = int(value)
                continue

            if key == "File type":
                file_type = _unquote(value)
            elif key == "Object class":
                object_class = _unquote(value)
            elif key == "xmin":
                xmin = float(value)
            elif key == "xmax":
                xmax = float(value)
            elif key == "size":
                size = int(value)

    if in_interval:
        raise ValueError(f"Unfinished TextGrid interval at end of {path}")
    finish_item()

    if xmin is None or xmax is None:
        raise ValueError(f"Missing TextGrid bounds in {path}")

    return TextGrid(
        file_type=file_type,
        object_class=object_class,
        xmin=xmin,
        xmax=xmax,
        tiers_exists=tiers_exists,
        size=size,
        items=items,
    )


def write_textgrid(
    textgrid: TextGrid,
    output_path: Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f'File type = "{_escape_text(textgrid.file_type)}"\n')
        f.write(f'Object class = "{_escape_text(textgrid.object_class)}"\n\n')
        f.write(f"xmin = {_format_time(textgrid.xmin)} \n")
        f.write(f"xmax = {_format_time(textgrid.xmax)} \n")
        f.write(f"tiers? {'<exists>' if textgrid.tiers_exists else '<absent>'} \n")
        f.write(f"size = {len(textgrid.items)} \n")
        f.write("item []: \n")

        for item_index, item in enumerate(textgrid.items, start=1):
            f.write(f"    item [{item_index}]:\n")
            f.write(f'        class = "{_escape_text(item.class_type)}" \n')
            f.write(f'        name = "{_escape_text(item.name)}" \n')
            f.write(f"        xmin = {_format_time(item.xmin)} \n")
            f.write(f"        xmax = {_format_time(item.xmax)} \n")
            f.write(f"        intervals: size = {len(item.intervals)} \n")

            for interval_index, interval in enumerate(item.intervals, start=1):
                f.write(f"        intervals [{interval_index}]:\n")
                f.write(f"            xmin = {_format_time(interval.xmin)} \n")
                f.write(f"            xmax = {_format_time(interval.xmax)} \n")
                f.write(f'            text = "{_escape_text(interval.text)}" \n')

def create_interval(
        xmin: float = 0.0,
        xmax: float = 0.0,
        text: str = ""
    ) -> Interval:
    return Interval(
        xmin=xmin,
        xmax=xmax,
        text=text
    )

def create_item(
        class_type: str = "",
        name: str = "",
        xmin: float = 0.0,
        xmax: float = 0.0,
        intervals: list[Interval] | None = None
    ) -> Item:
    if intervals is None:
        intervals = []
    return Item(
        class_type=class_type,
        name=name,
        xmin=xmin,
        xmax=xmax,
        intervals_size=len(intervals),
        intervals=intervals
    )
    

def _build_syllable_intervals(textgrid: TextGrid) -> list[Interval]:
    phone_item = _find_item(textgrid, "phones")
    if phone_item is None:
        raise ValueError("TextGrid must contain a phones tier before adding syllables")

    phones = _non_empty_intervals(phone_item.intervals)
    word_item = _find_item(textgrid, "words")
    syllable_intervals: list[Interval] = []

    if word_item is None:
        syllable_groups = _split_phones_to_syllables(phones)
        for index, group in enumerate(syllable_groups, start=1):
            syllable_intervals.append(
                create_interval(
                    xmin=group[0].xmin,
                    xmax=group[-1].xmax,
                    text=_syllable_label("", group, index, len(syllable_groups)),
                )
            )
        return _make_contiguous_intervals(syllable_intervals, textgrid.xmin, textgrid.xmax)

    for word_interval in _non_empty_intervals(word_item.intervals):
        word_phones = _phones_inside_interval(word_interval, phones)
        syllable_groups = _split_phones_to_syllables(word_phones)
        for index, group in enumerate(syllable_groups, start=1):
            syllable_intervals.append(
                create_interval(
                    xmin=group[0].xmin,
                    xmax=group[-1].xmax,
                    text=_syllable_label(
                        word_interval.text.strip(),
                        group,
                        index,
                        len(syllable_groups),
                    ),
                )
            )

    return _make_contiguous_intervals(syllable_intervals, textgrid.xmin, textgrid.xmax)


def add_syllable_tier(
    textgrid: TextGrid,
    tier_name: str = "syllables",
) -> TextGrid:
    syllable_intervals = _build_syllable_intervals(textgrid)
    textgrid.items = [
        item
        for item in textgrid.items
        if item.name.lower() != tier_name.lower()
    ]
    syllable_item = create_item(
        class_type="IntervalTier",
        name=tier_name,
        xmin=textgrid.xmin,
        xmax=textgrid.xmax,
        intervals=syllable_intervals,
    )
    textgrid.items.insert(1, syllable_item)
    textgrid.size = len(textgrid.items)
    return textgrid


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Read a Praat TextGrid file and write it back in long TextGrid format."
    )
    parser.add_argument("input_path", type=Path, help="Input .TextGrid file")
    parser.add_argument("output_path", type=Path, help="Output .TextGrid file")
    args = parser.parse_args()

    textgrid = read_textgrid(args.input_path)
    add_syllable_tier(textgrid)
    write_textgrid(textgrid, args.output_path)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
