# MCTM report — regenerate data, then build the PDF.
#
#   make          build report.pdf (two pdflatex passes)
#   make data     regenerate data/*.csv and macros.tex from ../results/*.json
#   make clean    remove build artefacts, keep the PDF

PDFLATEX ?= pdflatex
TEXFLAGS  = -interaction=nonstopmode -file-line-error
MAIN      = main
SOURCES   = $(MAIN).tex ttwhitepaper.sty macros.tex $(wildcard sections/*.tex) $(wildcard data/*.csv)

.PHONY: all data clean distclean

all: $(MAIN).pdf

data:
	cd .. && python report_data.py

$(MAIN).pdf: $(SOURCES)
	-$(PDFLATEX) $(TEXFLAGS) $(MAIN).tex
	-$(PDFLATEX) $(TEXFLAGS) $(MAIN).tex
	@grep -E "Output written" $(MAIN).log || true
	@./check.sh $(MAIN).log

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf
