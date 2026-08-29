SECTIONS := $(wildcard sections/*.tex)
DATA := data/processed/bins_summary.csv data/processed/ratio_profiles.csv \
        data/processed/absolute.csv data/processed/decomposition.csv \
        data/processed/catalog_meta.csv
INNER := data/processed/inner_study.csv data/processed/inner_study_summary.csv
FIGS := build/fig_ratio_grid.pdf build/fig_absolute.pdf \
        build/fig_decomposition.pdf build/fig_inner_study.pdf \
        build/fig_one2one.pdf

all: build/report.pdf

# heavy: reads the 530 MB mock and evaluates the model in all 12 bins.
# Only rebuilt when the script (or the model code upstream) changes.
$(DATA): scripts/make_results.py
	python scripts/make_results.py

$(INNER): scripts/make_inner_study.py
	python scripts/make_inner_study.py

$(FIGS): scripts/make_figures.py $(DATA) $(INNER)
	python scripts/make_figures.py

build/values.tex build/table_bins.tex: scripts/make_values.py $(DATA) $(INNER)
	python scripts/make_values.py

build/report.pdf: report.tex $(SECTIONS) build/values.tex \
                  build/table_bins.tex $(FIGS)
	TEXINPUTS=".:build:" pdflatex -interaction=nonstopmode -output-directory=build report.tex
	TEXINPUTS=".:build:" pdflatex -interaction=nonstopmode -output-directory=build report.tex

clean:
	rm -f build/*.pdf build/*.aux build/*.log build/*.out build/*.toc \
	      build/values.tex build/table_bins.tex

distclean: clean
	rm -f data/processed/*.csv

.PHONY: all clean distclean
