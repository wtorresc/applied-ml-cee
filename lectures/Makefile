# Lecture decks: Markdown -> PowerPoint (.pptx) via pandoc, plus a LibreOffice PDF preview.
#
#   make                            every cee_*.md -> build/<stem>.pptx + .pdf
#   make build/cee_l02_nearest_neighbors.pptx
#                                   one deck (figures/diagrams rebuild when their sources change)
#   make template                   regenerate template/aml-reference.pptx (overwrites edits made
#                                   to it in PowerPoint)
#   make claims                     every comparison, cause, and universal claim in every deck
#   make numbers                    every deck number that no numbers.json value renders to
#   make verify-cee_l21_graph_networks
#                                   rerun one figure script into build/verify/ and diff its
#                                   numbers.json against build/fig/'s (the stamp is not touched)
#   make clean
#
# Figures: figures/<stem>.py (matplotlib, run in the pixi env `aml-env`) and
# figures/diagrams/<stem>/*.tex (TikZ, lualatex) -> build/fig/<stem>/*.png. Each figure script also writes
# numbers.json with every number quoted on the slides.

SHELL        := /bin/bash
STEMS        := $(basename $(wildcard cee_*.md))
# PYTHONHASHSEED only takes effect when set before the interpreter starts.
PY           := PYTHONHASHSEED=0 pixi run -e aml-env python
BUILD        := build
TEMPLATE     := template/aml-reference.pptx
ATKINSON_DIR := $(dir $(shell kpsewhich AtkinsonHyperlegibleNext-Regular.otf))

PANDOC := pandoc --slide-level=2 --reference-doc=$(TEMPLATE) \
          -V monofont="Atkinson Hyperlegible Mono" --highlight-style=template/aml-code.theme \
          --resource-path=$(BUILD)/fig

diagrams = $(patsubst figures/diagrams/%.tex,$(BUILD)/fig/%.png,$(wildcard figures/diagrams/$(1)/*.tex))

.PHONY: all template clean claims numbers
.SECONDARY:
.SECONDEXPANSION:
# A recipe that fails must not leave a target that looks up to date.
.DELETE_ON_ERROR:

all: $(STEMS:%=$(BUILD)/%.pptx) $(STEMS:%=$(BUILD)/%.pdf)

# Every figure script imports SHARED. A dataset or simulator module, and the data it reads, are
# prerequisites only of the lectures that import it, so re-running a fetch script or editing
# bridges.py rebuilds those lectures and not, say, Lecture 16 (about 50 minutes on the GPU).
SHARED   := $(addprefix figures/,aml_style.py common.py title_art.py)
DATA     := ../data
CONCRETE := figures/concrete.py $(DATA)/derived/Concrete_Data.csv
NBI      := figures/bridges.py $(DATA)/derived/nbi_il_bridges.csv
BDG2     := figures/buildings.py $(DATA)/derived/bdg2_daily_electricity.npz \
            $(DATA)/raw/bdg2/metadata.csv
LTOWN    := figures/ltown.py $(addprefix $(DATA)/raw/battledim/,L-TOWN.inp 2018_Leakages.csv)
LTPP     := $(DATA)/derived/ltpp_iri_sections.csv $(addprefix $(DATA)/derived/ltpp/,\
            MON_HSS_PROFILE_SECTION.csv MON_HSS_VISIT_NO.csv PROJECT_HIST_AGE_EXP.csv)
# A feature cache is its .npy and the .json beside it (model, revision, preprocessing), which the
# loader checks before it trusts the array.
cache    = $(foreach c,$(1),$(DATA)/derived/$(c).npy $(DATA)/derived/$(c).json)
# Old figures are removed first, so a figure a script no longer draws cannot linger in a deck.
# The script's printed output goes to build/fig/<stem>.log (beside the directory, so it survives
# the next run's rm); warnings and errors still reach the terminal.
$(BUILD)/fig/%/.stamp: figures/%.py $(SHARED)
	$(PY) tools/check_estimators.py figures/$*.py
	rm -rf $(BUILD)/fig/$*
	$(PY) figures/$*.py --out $(BUILD)/fig/$* > $(BUILD)/fig/$*.log
	touch $@

$(BUILD)/fig/cee_l01_framing/.stamp: $(CONCRETE) $(NBI) $(BDG2)
$(BUILD)/fig/cee_l02_nearest_neighbors/.stamp: $(CONCRETE)
$(BUILD)/fig/cee_l03_generalization/.stamp: $(NBI)
$(BUILD)/fig/cee_l04_load_profiles/.stamp: $(BDG2)
$(BUILD)/fig/cee_l05_least_squares/.stamp: $(CONCRETE)
$(BUILD)/fig/cee_l06_linear_classifiers/.stamp: $(NBI)
$(BUILD)/fig/cee_l07_mixtures_bayes/.stamp: $(DATA)/derived/portland_i205_5min.csv
$(BUILD)/fig/cee_l08_trees_explanations/.stamp: $(LTPP)
$(BUILD)/fig/cee_l09_robust_estimation/.stamp: \
  $(addprefix $(DATA)/raw/battledim/,2018_SCADA_Pressures.csv 2018_Leakages.csv \
    2018_SCADA_Flows.csv L-TOWN.inp dataset_configuration.yaml)
$(BUILD)/fig/cee_l10_calibration_quantiles/.stamp: $(CONCRETE)
$(BUILD)/fig/cee_l11_conformal_shift/.stamp: $(DATA)/derived/usgs_05572000_daily.csv
$(BUILD)/fig/cee_l12_gaussian_processes/.stamp: $(LTOWN)
$(BUILD)/fig/cee_l13_stochastic_gradient/.stamp: $(DATA)/derived/nbi_us_deck_transitions.csv.gz
$(BUILD)/fig/cee_l14_neural_networks/.stamp: $(LTOWN)
$(BUILD)/fig/cee_l15_convolutional_networks/.stamp: figures/sdnet.py figures/probes.py \
  $(DATA)/derived/sdnet2018_tiles.csv $(call cache,sdnet2018_resnet18)
$(BUILD)/fig/cee_l16_segmentation/.stamp: figures/sdnet.py figures/crack3d.py figures/spacenet8.py \
  $(DATA)/raw/3dcrack/3DCrack_v1.zip $(DATA)/raw/spacenet8/Germany_Training_Public.tar.gz
$(BUILD)/fig/cee_l17_transformers/.stamp: $(CONCRETE) figures/sdnet.py figures/probes.py figures/osha.py \
  $(DATA)/derived/sdnet2018_tiles.csv $(call cache,sdnet2018_resnet18 sdnet2018_vit_b_16) \
  $(DATA)/raw/osha_sir/January2015toNovember2025.zip
$(BUILD)/fig/cee_l18_vision_foundation/.stamp: figures/sdnet.py figures/probes.py \
  figures/crack3d.py figures/spacenet8.py $(DATA)/derived/sdnet2018_tiles.csv \
  $(call cache,sdnet2018_resnet18 sdnet2018_vit_b_16 sdnet2018_clip_l14_336) \
  $(DATA)/raw/3dcrack/3DCrack_v1.zip $(DATA)/raw/spacenet8/Germany_Training_Public.tar.gz
$(BUILD)/fig/cee_l19_language_models/.stamp: figures/osha.py \
  $(DATA)/raw/osha_sir/January2015toNovember2025.zip $(DATA)/raw/mutcd/mutcd11thedition.pdf
$(BUILD)/fig/cee_l21_graph_networks/.stamp: $(LTOWN) $(addprefix $(DATA)/raw/battledim/,\
  2018_SCADA_Pressures.csv 2018_SCADA_Levels.csv 2019_SCADA_Pressures.csv 2019_SCADA_Levels.csv)
$(BUILD)/fig/cee_l20_time_series/.stamp: figures/camels.py \
  $(addprefix $(DATA)/derived/,camels_daily.npz camels_attributes.csv) \
  $(addprefix $(DATA)/raw/kw51/,traindata_201810.zip trackedmodes.zip)
$(BUILD)/fig/cee_l22_point_clouds/.stamp: figures/craslab.py \
  $(addprefix $(DATA)/derived/,craslab_sample.npz craslab_stats.json) \
  $(DATA)/raw/craslab/craslabbim.ifc
$(BUILD)/fig/cee_l24_learning_dynamics/.stamp: figures/camels.py \
  $(addprefix $(DATA)/derived/,camels_daily.npz camels_attributes.csv)
$(BUILD)/fig/cee_l25_agents/.stamp: $(LTOWN)
$(BUILD)/fig/cee_l26_reinforcement_learning/.stamp: $(NBI)
$(BUILD)/fig/cee_l27_fairness/.stamp: $(NBI) figures/cee_l06_linear_classifiers.py \
  $(DATA)/raw/svi/SVI_2022_Illinois_county.csv
# L28's GPU ledger reads each GPU lecture's recorded run time (and the file's time) from its
# numbers.json, so a rerun of any of them makes L28 stale. The files have no rule of their own
# (build those lectures first; in a parallel make, build L28 last), and only those that exist are
# prerequisites: while one is being rebuilt (its directory removed), the other decks and L28's
# deck still build, and a rerun of L28 itself stops with its script's FileNotFoundError.
GPU_NUMBERS := $(foreach s,cee_l16_segmentation cee_l17_transformers cee_l18_vision_foundation \
  cee_l19_language_models cee_l20_time_series cee_l21_graph_networks cee_l22_point_clouds \
  cee_l25_agents,$(BUILD)/fig/$(s)/numbers.json)
$(BUILD)/fig/cee_l28_ai_infrastructure/.stamp: \
  $(addprefix $(DATA)/derived/egrid2023/,BA23.csv ST23.csv PLNT23.csv) \
  $(addprefix $(DATA)/raw/grid/,EIA930_BALANCE_2024_Jan_Jun.csv EIA930_BALANCE_2024_Jul_Dec.csv) \
  $(addprefix $(DATA)/derived/gpu_power/,rtx3090ti_2026-11-20.csv rtx3090ti_2026-11-20.json) \
  $(wildcard $(GPU_NUMBERS))
# Rerun a figure script into build/verify/<stem>/ and diff its numbers.json against the
# one the deck was written from (tools/diff_numbers.py: numbers equal within 1e-12 and of one type,
# the timing keys it lists ignored; exit 1 on any difference). build/fig/<stem>/ and its stamp are
# left alone. The script's printed output goes to build/verify/<stem>.log.
verify-%: figures/%.py
	$(PY) tools/check_estimators.py figures/$*.py
	rm -rf $(BUILD)/verify/$*
	mkdir -p $(BUILD)/verify
	$(PY) figures/$*.py --out $(BUILD)/verify/$* > $(BUILD)/verify/$*.log
	$(PY) tools/diff_numbers.py $(BUILD)/fig/$*/numbers.json $(BUILD)/verify/$*/numbers.json

# Reading lists for the decks; neither runs a figure script.
# claims: every sentence that compares, gives a cause, or says "every"/"only" (always exit 0).
# numbers: every number no numbers.json value renders to; exit 1 while any is left. Judge each
# line; `$(PY) tools/deck_numbers.py DECK.md --skipped` also lists what was skipped, and why.
claims:
	@$(PY) tools/claim_words.py $(STEMS:%=%.md)

numbers:
	@$(PY) tools/deck_numbers.py $(STEMS:%=%.md)

# The backpropagation build: four frames of one picture
$(BUILD)/fig/cee_l14_neural_networks/backprop_%.png: \
  figures/diagrams/cee_l14_neural_networks/backprop_body.tikz

# A diagram may \input{numbers}: LaTeX macros that its lecture's figure script writes next to the
# PNGs. Hence the dependency on that script's stamp and the build directory in TEXINPUTS.
$(BUILD)/fig/%.png: figures/diagrams/%.tex figures/diagrams/diagram_style.tex $$(dir $$@).stamp
	TEXINPUTS=$(CURDIR)/figures/diagrams:$(CURDIR)/$(dir $@): \
	  lualatex -interaction=nonstopmode -halt-on-error -output-directory=$(dir $@) $< \
	  > $(@:.png=.buildlog) 2>&1 || (tail -20 $(@:.png=.buildlog); exit 1)
	pdftoppm -r 250 -png -singlefile $(@:.png=.pdf) $(@:.png=)

$(BUILD)/%.pptx: %.md $(TEMPLATE) template/aml-code.theme tools/postprocess_pptx.py tools/math_fallback.py $(BUILD)/fig/%/.stamp $$(call diagrams,$$*)
	$(PANDOC) $< -o $(BUILD)/$*.raw.pptx
	$(PY) tools/postprocess_pptx.py --max-title-chars 40 \
	  $$(test -f $(BUILD)/fig/$*/title_art.png && echo --title-art $(BUILD)/fig/$*/title_art.png) \
	  $(BUILD)/$*.raw.pptx $< $@
	rm $(BUILD)/$*.raw.pptx

# PDF via LibreOffice, tagged for screen readers. Equations come out typeset because each
# equation box carries a typeset fallback picture (tools/math_fallback.py). One LibreOffice
# profile per deck: instances that share a profile fail under make -j. The old PDF is removed
# first: soffice can exit 0 without writing one, and the check below must not find a stale copy.
$(BUILD)/%.pdf: $(BUILD)/%.pptx $(BUILD)/fonts.conf
	rm -f $@
	FONTCONFIG_FILE=$(CURDIR)/$(BUILD)/fonts.conf soffice \
	  -env:UserInstallation=file://$(CURDIR)/$(BUILD)/lo-profile/$* \
	  --headless --convert-to 'pdf:impress_pdf_Export:{"UseTaggedPDF":{"type":"boolean","value":"true"}}' --outdir $(BUILD) $< > /dev/null
	@test -s $@ || (echo "LibreOffice wrote no PDF for $<" >&2; exit 1)

# Lets LibreOffice find the Atkinson fonts that ship with TeX Live.
$(BUILD)/fonts.conf:
	$(if $(ATKINSON_DIR),,$(error Atkinson Hyperlegible fonts not found by kpsewhich; install the TeX Live atkinson package))
	mkdir -p $(BUILD)
	printf '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>\n  <dir>%s</dir>\n</fontconfig>\n' '$(ATKINSON_DIR)' > $@

template:
	$(PY) tools/make_template.py --out $(TEMPLATE)

clean:
	rm -rf $(BUILD)
