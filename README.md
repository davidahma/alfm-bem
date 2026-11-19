# ALFM-BEM

[![CI](https://github.com/davidahma/alfm-bem/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/davidahma/alfm-bem/actions/workflows/ci.yml)
[![Software license](https://img.shields.io/badge/software-MIT-blue.svg)](LICENSE)

**Outcome-conditioned experience memory for frozen predictors.** Store the embedding and outcome of a past case. For a new case, retrieve examples of past successes and failures and score its familiarity. The reference implementation exposes the scores and the cases that produced them.

[Read the paper and appendices](paper/alfm-bem.pdf) · [Browse the results](results/) · [Download the release](https://github.com/davidahma/alfm-bem/releases/tag/v0.2.0)

## What the controls show

**At the same review budget, the simpler policies were competitive.** Across 12 paired replays of handwritten digits, each policy referred 720 of 7,200 future case presentations. There were 454 classifier errors available to catch. BEM risk selected 178, frozen confidence selected 291, and a calibrated blend of BEM risk and confidence also selected 291. Combining risk, success, familiarity and confidence selected 290. The strongest comparison, a local class-label vote blended with confidence, selected 379. These totals reuse one digit corpus across seeds. [Benchmark protocol and paired comparisons](paper/benchmark.tex)

| Five-seed synthetic outcome classification | Failure F1 | Success F1 |
| --- | ---: | ---: |
| BEM | 0.588 | 0.716 |
| Outcome vote among cosine neighbors | **0.744** | **0.734** |

The vote uses the same stored outcomes, including successes. In a separate claims simulation, a failure-only rule selected the same referrals as the BEM filter across 10,000 generated claims. The paper also evaluates review allocation on handwritten digits with a fixed classifier, observed prediction errors and an identical review budget for each policy. [Synthetic and claims controls](results/controls.json) · [Digit benchmark](results/budget.json)

## Run a first example

Use Python 3.11 or later. The following commands install the library from this checkout. The example uses local arrays and runs on CPU.

```sh
git clone https://github.com/davidahma/alfm-bem.git
cd alfm-bem
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/feedback.py
```

```python
import numpy as np
from alfm_bem import BidirectionalExperienceMemory

memory = BidirectionalExperienceMemory(dim=3, similarity_threshold=0.5)
memory.add_experience(np.array([1., 0., 0.]), -1.0, "case-17")
score, cases = memory.risk_signal(np.array([0.8, 0.6, 0.]))
print(round(score, 3), [case.context_hash for case in cases])
```

An outcome ranges from −1 to +1. Supply a case identifier and an embedding from your predictor. `risk_signal` and `success_signal` return evidence scores and supporting records. `coverage_signal` summarizes local familiarity. Score calibration depends on the representation and the observations available.

## Reproduce the research

Run these commands from the cloned repository with the environment active. The core experiments run on CPU and obtain their inputs from seeded generators and the digits data bundled with scikit-learn. The numerical constraints match the reference runs.

```sh
python -m pip install -c constraints/research.txt -e '.[research]'
make test
make reproduce
```

To write fresh result files for inspection:

```sh
python -m experiments.controls --output controls.local.json
python -m experiments.budget_benchmark --output budget.local.json
```

The optional MiniLM text control uses the 20 Newsgroups corpus and a pinned model revision. It measures retrieval of a keyword-defined group. The corpus and model download on the first run. Use `--offline` to reuse the cache. [Reference output and package versions](results/text.json)

```sh
python -m pip install -e '.[text,research]'
python -m experiments.real_embeddings --output text.local.json
```

| Question | Entry point |
| --- | --- |
| How does the public memory API work? | [Runnable example](examples/feedback.py) and [implementation](src/alfm_bem/bem.py) |
| Which cases should a reviewer see when capacity is limited? | [Budget benchmark](experiments/budget_benchmark.py) and [protocol](paper/benchmark.tex) |
| How much do the baseline, input geometry and feedback assumptions matter? | [Controls](experiments/controls.py) and [control appendix](paper/controls.tex) |
| What does the adapter bound cover? | [Methods appendix](paper/methods.tex) and [contract tests](tests/test_core_contracts.py) |

## Build the paper

Install a TeX Live distribution with pdfLaTeX, BibTeX, Latin Modern, and latexmk. The PDF validation also uses Poppler's `pdftotext`. On Ubuntu install `latexmk texlive-latex-extra lmodern poppler-utils`.

```sh
make paper
make check-paper
```

The output is [paper/alfm-bem.pdf](paper/alfm-bem.pdf). All appendix sources and the bibliography live beside the manuscript. CI rebuilds the PDF, compares its text with the distributed copy, reruns the numerical benchmarks, tests two Python versions, and builds the Python distributions.

## Citation and contribution

Use [CITATION.cff](CITATION.cff) to cite this release. The [original archived manuscript](https://doi.org/10.5281/zenodo.17868262) describes the earlier proposal. The current preprint revises its claims and adds matched comparisons.

Software is available under MIT. The paper and the author's result tables are CC BY 4.0. Dataset provenance and attribution are in [NOTICE.md](NOTICE.md). For changes and reproducibility reports, see [CONTRIBUTING.md](CONTRIBUTING.md). For private reports, see [SECURITY.md](SECURITY.md).
