#!/usr/bin/env python3
"""Export each LaTeX table in main.tex as a standalone PNG image."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


TABLE_RE = re.compile(
    r"\\begin\{(?P<env>table\*?)\}(?:\[[^\]]*\])?(?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
BIBCITE_RE = re.compile(r"\\bibcite\{([^}]+)\}\{([^}]+)\}")


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Missing required command: {name}")
    return path


def sanitize_label(label: str | None, index: int) -> str:
    if not label:
        return f"table_{index:02d}"

    label = label.split(":", 1)[-1]
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
    return f"table_{index:02d}_{label or 'table'}"


def find_label(body: str) -> str | None:
    match = re.search(r"\\label\{([^}]+)\}", body)
    return match.group(1) if match else None


def read_bibcites(report_dir: Path) -> str:
    aux_path = report_dir / "build" / "main.aux"
    if not aux_path.exists():
        return ""

    lines = []
    for line in aux_path.read_text(encoding="utf-8").splitlines():
        match = BIBCITE_RE.match(line)
        if match:
            key, number = match.groups()
            lines.append(rf"\global\@namedef{{b@{key}}}{{{number}}}")
    if not lines:
        return ""

    return "\n".join(
        [
            r"\makeatletter",
            *lines,
            r"\makeatother",
        ]
    )


def table_document(body: str, table_index: int, width: str, bibcites: str) -> str:
    body = re.sub(r"\\caption(\[[^\]]*\])?\{", r"\\captionof{table}\1{", body)
    body = body.replace(r"\columnwidth", r"\linewidth")

    return rf"""\documentclass[12pt]{{article}}
\usepackage{{caption}}
\input{{packages.tex}}
\input{{settings.tex}}
\geometry{{
    paperwidth=12cm,
    paperheight=20cm,
    left=1cm,
    right=1cm,
    top=1cm,
    bottom=1cm,
    noheadfoot
}}

\newcommand{{\crab}}{{\textsc{{Crab}}}}
\renewcommand{{\thetable}}{{\Roman{{table}}}}
\definecolor{{crabgreen}}{{HTML}}{{79E889}}
\sisetup{{
    group-separator = {{,}},
    group-minimum-digits = 4,
    group-digits = integer
}}
{bibcites}

\begin{{document}}
\thispagestyle{{empty}}
\noindent
\setcounter{{table}}{{{table_index - 1}}}
\begin{{minipage}}{{{width}}}
{body.strip()}
\end{{minipage}}
\end{{document}}
"""


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def export_tables(report_dir: Path, tex_name: str, out_dir: Path, dpi: int, width: str) -> list[Path]:
    require_tool("latexmk")
    require_tool("pdfcrop")
    require_tool("gs")

    source = report_dir / tex_name
    build_dir = report_dir / "build" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    content = source.read_text(encoding="utf-8")
    tables = list(TABLE_RE.finditer(content))
    if not tables:
        raise SystemExit(f"No table environments found in {source}")

    bibcites = read_bibcites(report_dir)
    outputs: list[Path] = []
    for index, match in enumerate(tables, start=1):
        body = match.group("body")
        name = sanitize_label(find_label(body), index)
        tex_path = build_dir / f"{name}.tex"
        pdf_path = build_dir / f"{name}.pdf"
        cropped_pdf_path = build_dir / f"{name}-crop.pdf"
        png_path = out_dir / f"{name}.png"

        tex_path.write_text(table_document(body, index, width, bibcites), encoding="utf-8")

        run_command(
            [
                "latexmk",
                "-xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-outdir=build/tables",
                str(tex_path.relative_to(report_dir)),
            ],
            cwd=report_dir,
        )
        run_command(["pdfcrop", str(pdf_path), str(cropped_pdf_path)], cwd=report_dir)
        run_command(
            [
                "gs",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pngalpha",
                f"-r{dpi}",
                f"-sOutputFile={png_path}",
                str(cropped_pdf_path),
            ],
            cwd=report_dir,
        )
        outputs.append(png_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default="main.tex", help="LaTeX source file under report/")
    parser.add_argument(
        "--out-dir",
        default="figure/tables",
        help="Output directory under report/ for PNG files",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG render resolution")
    parser.add_argument(
        "--width",
        default="9.2cm",
        help="Minipage width used for each exported table",
    )
    args = parser.parse_args()

    report_dir = Path(__file__).resolve().parent
    out_dir = report_dir / args.out_dir
    outputs = export_tables(report_dir, args.tex, out_dir, args.dpi, args.width)

    for output in outputs:
        print(output.relative_to(report_dir))


if __name__ == "__main__":
    main()
