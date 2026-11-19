PYTHON ?= python
BUILD := build/paper
TEX_IMAGE := texlive/texlive@sha256:c572bf2d14631545769e5a8f8d315e7b8f8896b14720b1c7a063917d503fd46a
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export MPLBACKEND ?= Agg

.PHONY: test reproduce paper check-paper paper-container package
test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
	$(PYTHON) examples/feedback.py

reproduce:
	$(PYTHON) -m experiments.controls --check results/controls.json
	$(PYTHON) -m experiments.budget_benchmark --check results/budget.json

paper:
	$(PYTHON) scripts/paper_results.py --check
	mkdir -p $(BUILD)
	cd paper && latexmk $(LATEXMK_FLAGS) -pdf -halt-on-error -interaction=nonstopmode -outdir=../$(BUILD) alfm-bem.tex
	cp $(BUILD)/alfm-bem.pdf paper/alfm-bem.pdf

check-paper:
	$(PYTHON) scripts/check_paper.py --compare paper/alfm-bem.pdf

# The pinned image includes the complete toolchain. Typesetting runs offline.
paper-container:
	docker run --rm --network none --platform linux/amd64 \
	  --user "$$(id -u):$$(id -g)" -e HOME=/tmp -e SOURCE_DATE_EPOCH=1794960000 -e FORCE_SOURCE_DATE=1 \
	  -v "$(CURDIR):/work" -w /work $(TEX_IMAGE) sh -ec \
	  'make paper PYTHON=python3 BUILD=build/release-paper LATEXMK_FLAGS=-gg && python3 scripts/check_paper.py --build-dir build/release-paper $(if $(REFERENCE),--compare "$(REFERENCE)",)'

package:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
