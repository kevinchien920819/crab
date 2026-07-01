MAIN := main
BUILD_DIR := build

.PHONY: pdf table-images
pdf:
	mkdir -p $(BUILD_DIR)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -auxdir=$(BUILD_DIR) -outdir=. $(MAIN).tex

table-images: pdf
	python3 export_tables.py
