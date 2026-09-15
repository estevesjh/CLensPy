SECTIONS := $(wildcard sections/*.tex)
FIG6_DATA := data/processed/fig6_points.csv data/processed/fig6_panels.csv
MOCK_DATA := data/processed/mock_legD.csv data/processed/mock_legC.csv \
             data/processed/mock_legE.csv
ALPHA_DATA := data/processed/alpha_sensitivity.csv
COSMO_DATA := data/processed/cosmo_sensitivity.csv
DATA := $(FIG6_DATA) $(MOCK_DATA) $(ALPHA_DATA) $(COSMO_DATA)
FIGS := build/fig_fig6.pdf build/fig_mock_ratio.png build/fig_alpha_sensitivity.pdf \
        build/fig_cosmo_sensitivity.pdf
MODEL_SRC := ../../src/clenspy/selection/bsel.py \
             ../../src/clenspy/selection/scaling_relation.py \
             ../../src/clenspy/selection/richness_kernel.py \
             ../../src/clenspy/lensing/projection.py \
             ../../validation/validate_sigma_prj_mock.py
FIG6_REF := ../../validation/data/costanzi2026_fig6.csv
MOCK_RATIO_PNG := ../../docs/_static/validation/sigma_prj_ratio_grid.png

all: build/report.pdf

# heavy: rebuilds CAMB/HMF/bias once and evaluates the model on the
# digitized Fig.6 curve. Only rebuilt when the script or the model
# upstream changes.
$(FIG6_DATA): data/processed/.fig6.stamp ;
data/processed/.fig6.stamp: scripts/make_fig6.py $(MODEL_SRC) $(FIG6_REF)
	python scripts/make_fig6.py
	@touch $@

# heaviest: reads the ~530MB mock catalogue. Only rebuilt when the
# script or the model upstream changes.
$(MOCK_DATA): data/processed/.mock.stamp ;
data/processed/.mock.stamp: scripts/make_mock.py $(MODEL_SRC)
	python scripts/make_mock.py
	@touch $@

$(ALPHA_DATA): data/processed/.alpha.stamp ;
data/processed/.alpha.stamp: scripts/make_alpha_sensitivity.py $(MODEL_SRC)
	python scripts/make_alpha_sensitivity.py
	@touch $@

# heavy: rebuilds CAMB/HMF/bias once per cosmology variant (20 variants).
$(COSMO_DATA): data/processed/.cosmo.stamp ;
data/processed/.cosmo.stamp: scripts/make_cosmo_sensitivity.py $(MODEL_SRC)
	python scripts/make_cosmo_sensitivity.py
	@touch $@

# fig_mock_ratio.png is copied from validate_sigma_prj_mock.py's own
# --plot output (regenerate that with
# `SELECTION_BIAS_DIR=... python ../../validation/validate_sigma_prj_mock.py --plot`
# to refresh it before rebuilding this report).
$(FIGS): build/.figs.stamp ;
build/.figs.stamp: scripts/make_figures.py $(DATA) $(MOCK_RATIO_PNG)
	python scripts/make_figures.py
	@touch $@

build/values.tex build/table_fig6.tex build/table_mock.tex build/table_legE.tex: build/.values.stamp ;
build/.values.stamp: scripts/make_values.py $(DATA)
	python scripts/make_values.py
	@touch $@

build/report.pdf: report.tex $(SECTIONS) build/values.tex \
                  build/table_fig6.tex build/table_mock.tex build/table_legE.tex $(FIGS)
	TEXINPUTS=".:build:" pdflatex -interaction=nonstopmode -output-directory=build report.tex
	TEXINPUTS=".:build:" pdflatex -interaction=nonstopmode -output-directory=build report.tex

clean:
	rm -f build/*.pdf build/*.aux build/*.log build/*.out build/*.toc \
	      build/values.tex build/table_fig6.tex build/table_mock.tex \
	      build/table_legE.tex build/.figs.stamp build/.values.stamp \
	      build/fig_*.pdf build/fig_*.png

distclean: clean
	rm -f data/processed/*.csv data/processed/.fig6.stamp \
	      data/processed/.mock.stamp data/processed/.alpha.stamp \
	      data/processed/.cosmo.stamp

.PHONY: all clean distclean
