SECTIONS := $(wildcard sections/*.tex)
DATA := data/processed/bins_summary.csv data/processed/ratio_profiles.csv \
        data/processed/absolute.csv data/processed/decomposition.csv \
        data/processed/catalog_meta.csv
INNER := data/processed/inner_study.csv data/processed/inner_study_summary.csv
AREA_OVERLAP := data/processed/area_overlap_check.csv \
                data/processed/area_overlap_summary.csv \
                data/processed/area_overlap_conc_sensitivity.csv
FIGS := build/fig_ratio_grid.pdf build/fig_paper6.pdf build/fig_absolute.pdf \
        build/fig_decomposition.pdf build/fig_inner_study.pdf \
        build/fig_one2one.pdf build/fig_area_overlap.pdf
MODEL_SRC := ../../src/clenspy/lensing/projection.py \
             ../../src/clenspy/selection/bsel.py \
             ../../src/clenspy/selection/richness_kernel.py \
             ../../validation/validate_sigma_prj_mock.py
AREA_OVERLAP_SRC := ../../src/clenspy/selection/geometry.py

all: build/report.pdf

# Multi-output recipes go through stamp files: a plain multi-target rule
# runs its recipe once PER out-of-date target (make < 4.3 has no grouped
# targets), so `make -B` would re-run the heavy scripts several times.

# heavy: reads the 530 MB mock and evaluates the model in all 12 bins.
# Only rebuilt when the script (or the model code upstream) changes.
$(DATA): data/processed/.results.stamp ;
data/processed/.results.stamp: scripts/make_results.py $(MODEL_SRC)
	python scripts/make_results.py
	@touch $@

$(INNER): data/processed/.inner.stamp ;
data/processed/.inner.stamp: scripts/make_inner_study.py $(MODEL_SRC)
	python scripts/make_inner_study.py
	@touch $@

# independent of the mock/model chain above: a standalone geometric check
# of area_overlap against the exact NFW ring integral (see report sec.
# "Area overlap"). Only rebuilt when its own script or geometry.py changes.
$(AREA_OVERLAP): data/processed/.area_overlap.stamp ;
data/processed/.area_overlap.stamp: scripts/make_area_overlap_check.py $(AREA_OVERLAP_SRC)
	python scripts/make_area_overlap_check.py
	@touch $@

$(FIGS): build/.figs.stamp ;
build/.figs.stamp: scripts/make_figures.py $(DATA) $(INNER) $(AREA_OVERLAP)
	python scripts/make_figures.py
	@touch $@

build/values.tex build/table_bins.tex build/table_absolute.tex: build/.values.stamp ;
build/.values.stamp: scripts/make_values.py $(DATA) $(INNER) $(AREA_OVERLAP)
	python scripts/make_values.py
	@touch $@

build/report.pdf: report.tex $(SECTIONS) build/values.tex \
                  build/table_bins.tex build/table_absolute.tex $(FIGS)
	TEXINPUTS=".:build:" pdflatex -interaction=nonstopmode -output-directory=build report.tex
	TEXINPUTS=".:build:" pdflatex -interaction=nonstopmode -output-directory=build report.tex

clean:
	rm -f build/*.pdf build/*.aux build/*.log build/*.out build/*.toc \
	      build/values.tex build/table_bins.tex build/table_absolute.tex \
	      build/.figs.stamp build/.values.stamp

distclean: clean
	rm -f data/processed/*.csv data/processed/.results.stamp \
	      data/processed/.inner.stamp data/processed/.area_overlap.stamp

.PHONY: all clean distclean
