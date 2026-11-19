PYTHON ?= python
BUILD := build/paper
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export MPLBACKEND ?= Agg

.PHONY: test reproduce paper check-paper package
test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
	$(PYTHON) examples/feedback.py

reproduce:
	$(PYTHON) -m experiments.controls --check results/controls.json
	$(PYTHON) -m experiments.budget_benchmark --check results/budget.json

paper:
	$(PYTHON) scripts/paper_results.py --check
	mkdir -p $(BUILD)
	cd paper && latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=../$(BUILD) alfm-bem.tex
	cp $(BUILD)/alfm-bem.pdf paper/alfm-bem.pdf

check-paper:
	$(PYTHON) scripts/check_paper.py --compare paper/alfm-bem.pdf

package:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
