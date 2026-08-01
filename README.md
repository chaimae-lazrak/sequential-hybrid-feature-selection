# Sequential hybrid framework for medical feature selection using population transfer

Reference implementation for the paper:

> C. Lazrak, A. Bouayad, A. Talha. *A sequential hybrid framework for medical
> feature selection using population transfer.* Discover Artificial
> Intelligence, [year]. DOI: [to be added on acceptance]

The framework couples an exploration-oriented metaheuristic (PSO, HHO or BSO)
with the Binary Al-Biruni Earth Radius algorithm (bABER) through a direct
population transfer at a fixed transition point, and evaluates it on six
benchmark medical datasets under a leakage-aware protocol.

---

## Repository structure

```
src/
  algorithms/
    base_algorithm.py      shared BaseMetaheuristic interface
    pso.py  hho.py  bso.py  baber.py
    pso_baber.py  hho_baber.py  bso_baber.py
  classifier/
    rf_classifier.py       Random Forest wrapper (fitness + test evaluation)
  preprocessing/
    data_loader.py         one loader per dataset
    preprocessor.py        stratified split and z-score scaling
  evaluation/
    evaluator.py           metric aggregation
    statistical_tests.py   Friedman, Wilcoxon, effect sizes
  utils/
    binary_utils.py        transfer functions for binarisation
    fitness.py             objective of Eq. 18
experiments/
  config.py                all parameters used in the paper
  run_experiments.py       driver for the 630 optimization runs
data/
  raw/                     datasets (not redistributed, see data/README.md)
results/                   raw per-run results and figures
```

## Requirements

Python 3.11.14 was used for all reported experiments. Install with:

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

```bash
python experiments/run_experiments.py
```

This runs 7 algorithms × 6 datasets × 15 independent runs (630 optimization
runs) and writes the raw results to `results/`. On an Apple M4 with 16 GB of
memory the full grid takes approximately 130 hours; individual runs take between
560 and 1140 seconds depending on the dataset.

The datasets must be placed in `data/raw/` first; see `data/README.md`.

## Reproducing the tables and figures

The statistical tables of Section 6.6 can be regenerated from the aggregated
results without re-running the optimization:

```bash
python src/evaluation/statistical_tests.py
```

## Experimental protocol

Each of the 15 independent runs draws its own stratified 70/30 train-test
partition, using `random_state = 42 + run_number`, that is seeds 42 to 56. The
scaler is fitted on the training partition of each run only. Within a run, every
fitness evaluation is a 5-fold cross-validation internal to the training
partition. Reported metrics are means and standard deviations over the 15 runs,
each run being evaluated with its own selected subset. No selection among runs
is made on the basis of test-set performance.

Standard deviations reported in the paper are population standard deviations
(`numpy.std` with `ddof=0`) over the 15 runs.

## Known limitations of this release

These are stated in the paper and repeated here so that anyone reproducing the
work knows what to expect.

**The stochastic behaviour of the search is not seeded.** Only the train-test
split is seeded. Re-running the pipeline therefore reproduces the reported
distributions but not individual runs. A global seed can be set per run by
adding `numpy.random.seed(1000 + run_number)` before the call to
`algorithm.run()` in `experiments/run_experiments.py`; this was not done for the
reported experiments.

**Per-run feature subsets were not logged.** Only the number of selected
features is stored for every run; the identity of the selected features is
retained for the best-performing run of each algorithm and dataset. Stability
indices across runs (Jaccard, Kuncheva) therefore cannot be computed from the
released results.

**Per-iteration population states were not logged**, which prevents a direct
measurement of population diversity before and after the transfer.

**The Phase 2 schedule differs between variants.** In `hho_baber.py` and
`bso_baber.py` a bABER instance is created with `max_iter` set to the Phase 2
budget and receives a counter reset at the transition, as described in Section
4.2 of the paper. In `pso_baber.py` the bABER update is inlined and receives the
global iteration counter, so its exploration ratio starts at approximately 0.46
rather than 0.70. This is reported in Section 4.3 of the paper.

**Iteration indexing.** The phase test is applied to a zero-based counter, so
Phase 1 performs 61 population updates and Phase 2 performs 39, against the
nominal budgets of 60 and 40 reported in the paper. The offset is identical for
the three hybrid variants.

## Parameters

All parameters are defined in `experiments/config.py` and correspond to Tables 2
and 3 of the paper: population size 15, 100 iterations, transition ratio 0.6,
α = 0.99; PSO with w = 0.7 and c1 = c2 = 1.5; HHO with β = 1.5; BSO with K = 3,
p_replace = 0.2 and p_one = 0.8; bABER with exploration ratio decreasing from
0.7 to 0.3 and a stagnation threshold of 3; Random Forest with 50 trees and a
maximum depth of 10.

## License

Code released under the MIT License (see `LICENSE`). The datasets are
distributed by their original providers under their own terms; see
`data/README.md`.

## Citation

See `CITATION.cff`.
