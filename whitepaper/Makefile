# torchtsetlin white paper
#
#   make          build main.pdf (two pdflatex passes for refs and the ToC)
#   make summary  build summary.pdf, the five-page report to cite
#   make watch    rebuild on every change (needs latexmk)
#   make clean    remove build artefacts, keep the PDF
#   make distclean remove everything generated

PDFLATEX ?= pdflatex
TEXFLAGS  = -interaction=nonstopmode -file-line-error
MAIN     = main
PDF      = $(MAIN).pdf
SOURCES  = $(MAIN).tex ttwhitepaper.sty $(wildcard sections/*.tex) \
           $(wildcard figures/*.tex) $(wildcard data/*.csv)

.PHONY: all summary watch clean distclean

all: $(PDF)

$(PDF): $(SOURCES)
	$(PDFLATEX) $(TEXFLAGS) $(MAIN).tex
	$(PDFLATEX) $(TEXFLAGS) $(MAIN).tex
	@echo
	@grep -E "Output written|LaTeX Warning: There were undefined" $(MAIN).log || true

summary: summary.pdf

summary.pdf: summary.tex ttwhitepaper.sty $(wildcard figures/*.tex) $(wildcard data/*.csv) data/lit-stats.tex
	$(PDFLATEX) $(TEXFLAGS) summary.tex
	$(PDFLATEX) $(TEXFLAGS) summary.tex
	@grep -E "Output written|LaTeX Warning: There were undefined" summary.log || true

watch:
	latexmk -pdf -pvc $(MAIN).tex

clean:
	rm -f summary.aux summary.log summary.out
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).nav \
	      $(MAIN).snm $(MAIN).synctex.gz build*.log

distclean: clean
	rm -f $(PDF) summary.pdf
