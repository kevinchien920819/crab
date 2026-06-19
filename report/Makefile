MAIN := main
BUILD_DIR := build

.PHONY: pdf
pdf:
	mkdir -p $(BUILD_DIR)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -auxdir=$(BUILD_DIR) -outdir=. $(MAIN).tex
