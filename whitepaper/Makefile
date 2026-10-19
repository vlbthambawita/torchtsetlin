# torchtsetlin white paper
#
#   make          build main.pdf (two pdflatex passes for refs and the ToC)
#   make summary  build summary.pdf, the five-page report to cite
#   make arxiv    torchtsetlin-arxiv.zip: the report's self-contained source for arXiv
#   make watch    rebuild on every change (needs latexmk)
#   make clean    remove build artefacts, keep the PDF
#   make distclean remove everything generated

PDFLATEX ?= pdflatex
TEXFLAGS  = -interaction=nonstopmode -file-line-error
MAIN     = main
PDF      = $(MAIN).pdf
SOURCES  = $(MAIN).tex ttwhitepaper.sty $(wildcard sections/*.tex) \
           $(wildcard figures/*.tex) $(wildcard data/*.csv)

.PHONY: all summary arxiv watch clean distclean

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

# Exactly the files summary.tex reads; summary.tex becomes main.tex at the top level.
ARXIV_FILES = ttwhitepaper.sty figures/fig-batched.tex figures/fig-bench-batch.tex \
              figures/fig-bench-fidelity.tex figures/fig-literature.tex \
              data/batch-mnist.csv data/batch-xor.csv data/lit-best.csv data/lit-top.csv \
              data/lit-stats.tex

arxiv: torchtsetlin-arxiv.zip

torchtsetlin-arxiv.zip: summary.tex $(ARXIV_FILES)
	rm -rf .arxiv $@ && mkdir -p .arxiv
	cp summary.tex .arxiv/main.tex
	cp --parents $(ARXIV_FILES) .arxiv/
	cd .arxiv && zip -q -r -X ../$@ main.tex $(ARXIV_FILES)
	rm -rf .arxiv
	@unzip -l $@ | tail -1

watch:
	latexmk -pdf -pvc $(MAIN).tex

clean:
	rm -f summary.aux summary.log summary.out
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).nav \
	      $(MAIN).snm $(MAIN).synctex.gz build*.log

distclean: clean
	rm -f $(PDF) summary.pdf
