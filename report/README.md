# Report

This directory contains the LaTeX source and generated PDF for the project
report.

## Layout

- `main.tex` - report source.
- `packages.tex` - LaTeX package imports.
- `settings.tex` - page, font, and header settings.
- `references.bib` - BibTeX references.
- `figure/` - report image assets.
- `fonts/` - bundled fonts used by XeLaTeX.
- `main.pdf` - generated report PDF.
- `build/` - LaTeX intermediate files kept out of the root directory.
- `archive/` - preserved old backups.

## Build

From this directory, run:

```sh
make pdf
```

The build writes LaTeX intermediate files to `build/` and writes the latest
PDF to `main.pdf`.

The files currently in `build/` and `archive/main.tex.bak` are kept for
traceability. They can be removed later only after explicit approval.
