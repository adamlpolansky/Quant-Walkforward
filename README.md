# Quant Walk-Forward (first project)

Hi! This is my first ever quant project on my journey to become a strong quant researcher/developer.

## Goal
Build a small but solid research framework that:
- downloads market data (initially via `yfinance`)
- runs **walk-forward** evaluation with rolling **train/test splits** (e.g. 9M train -> 1M test or 12M -> 1M)
- includes **anti-leakage guardrails** (strict time indexing, lag-only features, no lookahead)

This is meant to be the foundation for later work with options data, implied volatility surfaces, and eventually **Heston calibration**.

---

## Project layout

```text
Quant-Walkforward/
  pyproject.toml                # package config (src-layout install via `pip install -e .`)
  requirements.txt

  src/
    qwf/
      __init__.py
      data.py                   # data loading + return helpers
      splits.py                 # walk-forward plan generator
      backtest.py               # signal + backtest core (no I/O)
      metrics.py                # fold summary metrics + stitched curves
      reporting/
        __init__.py
        plots.py                # plot saving helpers

  scripts/
    pull_data.py                # (optional) data pull -> saves to scripts/data/
    rolling_splits.py           # generates walk-forward plan CSV (portable, no hardcoded paths)
    backtest_run.py             # runs walk-forward backtest + saves outputs + plots
    param_sweep.py              # grid search / mini experiment runner
    data/                       # input CSVs (e.g., SPY.csv)
    splits_train_test/          # generated plan CSVs

  tests/
    ...                         # leakage / lookahead tests (pytest)

  outputs/
    ...                         # saved CSVs / JSON / plots
```

---

## Current status
- ✅ Day 1: environment + data download + basic preprocessing (SPY)
- ✅ Day 2: walk-forward split planner (calendar months -> last trading day), saved as plan CSV
- ✅ Day 3: Z-score mean reversion signal + backtest core on `ret` (lagged position), basic metrics export
- ✅ Packaging fix: `src/` layout is now installable (`pyproject.toml` + `pip install -e .`)
- ✅ Script path cleanup:
  - `pull_data.py` now saves to `scripts/data/`
  - `rolling_splits.py` uses portable `Path(...)` defaults (no hardcoded Windows paths)
- ✅ Plot reporting added (`qwf.reporting.plots`) and called from `scripts/backtest_run.py`
- ✅ Walk-forward `test_detail` now recomputes fold-local `equity`/`cum_pnl` **after slicing to TEST** (prevents train-period carryover in per-fold test curves)
- ✅ Benchmark comparison (Strategy vs SPY Buy & Hold) saved in report outputs
- ✅ Transaction costs toggle via `--cost-bps-per-turnover`
- ✅ Param sweep / mini experiment runner that saves `results_grid.csv`

---

## Installation (recommended)
This project uses a `src/` layout, so install it in editable mode.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Why `pip install -e .`?
Without this, scripts may fail with:

```text
ModuleNotFoundError: No module named 'qwf'
```

Editable install makes `qwf` importable while you keep editing files in `src/qwf/`.

---

## Quickstart (end-to-end)

From the project root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/pull_data.py
python scripts/rolling_splits.py --train-months 9 --test-months 1
python scripts/backtest_run.py --save-per-fold

pytest -q
```

### What this does
1. Installs dependencies and the local package (`qwf`)
2. Downloads SPY daily data and saves it to `scripts/data/SPY.csv`
3. Builds a walk-forward plan CSV (default path: `scripts/splits_train_test/walkforward_plan_9_1.csv`)
4. Runs the backtest and saves:
   - `*_test_detail.csv`
   - `*_fold_summary.csv`
   - `*_config.json`
   - `*_perf_summary.csv` (strategy vs benchmark summary)
   - plots under `outputs/plots/<run_name>/`
5. Runs tests (if the test file names follow pytest conventions)

---

## Script usage

### 1) Pull data
```bash
python scripts/pull_data.py
```

**Current behavior**
- downloads `SPY`
- computes returns
- saves to:

```text
scripts/data/SPY.csv
```

---

### 2) Generate walk-forward splits
```bash
python scripts/rolling_splits.py --train-months 9 --test-months 1
```

#### Useful options
- `--train-months` (default `9`)
- `--test-months` (default `1`)
- `--step-months` (default `1`)
- `--start-date` (default `2018-01-01`)
- `--date-col` (default `Date`)
- `--input-dir` (default `scripts/data/`)
- `--output-csv` (optional explicit path)

#### Example (custom output path)
```bash
python scripts/rolling_splits.py   --train-months 12   --test-months 1   --output-csv scripts/splits_train_test/walkforward_plan_12_1.csv
```

---

### 3) Run backtest
```bash
python scripts/backtest_run.py --save-per-fold
```

#### Main options
- `--data` (default: `scripts/data/SPY.csv`)
- `--plan` (default: `scripts/splits_train_test/walkforward_plan_9_1.csv`)
- `--out-dir` (default: `outputs/`)
- `--run-name` (default: `wf_9m_1m_zscore_v1`)
- `--date-col` (default: `Date`)
- `--price-col` (default: `Close`)
- `--ret-col` (default: `ret`)
- `--source-file` (optional filter for multi-file plans)
- signal params:
  - `--n` (default `20`)
  - `--K` (default `1.0`)
  - `--step-frac` (default `0.25`)
  - `--ddof` (default `0`)
- realism params:
  - `--cost-bps-per-turnover` (default `0.0`)
- `--save-per-fold` -> saves per-fold **CSV files** into `outputs/folds/`

#### Example (custom run + costs)
```bash
python scripts/backtest_run.py \
  --plan scripts/splits_train_test/walkforward_plan_12_1.csv \
  --run-name wf_12m_1m_zscore_n20_cost10 \
  --n 20 --K 1.0 --step-frac 0.25 --ddof 0 \
  --cost-bps-per-turnover 10 \
  --save-per-fold
```

---

### 4) Param sweep / mini experiment
Run a small grid and save results:

```bash
python scripts/param_sweep.py --n-grid "10,20,40" --step-frac-grid "0.25,0.5"
```

Optional: include transaction costs in the sweep:

```bash
python scripts/param_sweep.py --n-grid "10,20,40" --step-frac-grid "0.25,0.5" --cost-bps-per-turnover 10
```

Outputs:
- `outputs/experiments/results_grid.csv`
- `outputs/experiments/results_grid_best.md` (Markdown table suitable for README)

---

## Outputs

### Core outputs (default `out-dir = outputs/`)
- `outputs/<run_name>_test_detail.csv`
- `outputs/<run_name>_fold_summary.csv`
- `outputs/<run_name>_config.json`
- `outputs/<run_name>_perf_summary.csv` (strategy vs benchmark summary)

### Plots
Plots are saved under:

```text
outputs/plots/<run_name>/
```

Typical files:
- `equity_stitched.png`
- `drawdown_stitched.png`
- `rolling_sharpe.png`
- `sharpe_by_fold.png`
- `total_return_by_fold.png`
- `equity_vs_benchmark.png` (Strategy vs SPY Buy & Hold)

### Per-fold plots (important)
`--save-per-fold` currently saves **CSV files only** (not PNG plots).

Per-fold plot PNGs are controlled in `scripts/backtest_run.py` by:
```python
save_per_fold_equity=False
```

If you want per-fold equity PNGs, change it to:
```python
save_per_fold_equity=True
```

Then rerun:
```bash
python scripts/backtest_run.py --save-per-fold
```

Per-fold plots will be saved to:
```text
outputs/plots/<run_name>/folds/
```

---

## Anti-leakage guardrails (implemented)

### Backtest timing discipline
- Positions are applied with a **strict lag**:
  - `pnl(t) = pos(t-1) * ret(t)`
- Signal is computed on `train_start .. test_end` only so the first test point gets a valid lagged state from train history.

### Fold-local equity semantics (fixed)
In walk-forward output:
- test rows are sliced first (`test_start .. test_end`)
- then `cum_pnl` / `equity` are recomputed on the **TEST slice only**

This prevents `test_detail` fold curves from being "pre-loaded" by train-period cumulative values.

---

## Testing

### Run tests
```bash
pytest -q
```

### Pytest naming note
Pytest auto-discovers files like:
- `test_*.py`
- `*_test.py`

If a test file is named something else (for example `leakage_and_looahead.py`), `pytest -q` may show `0 tests ran`.

Recommended rename:
```bash
git mv tests/leakage_and_looahead.py tests/test_leakage_and_lookahead.py
```

---

## Troubleshooting

### 1) `No module named 'qwf'`
Install the project in editable mode from the repo root:
```bash
python -m pip install -e .
```

---

### 2) "Plots were not saved"
Plots are **not** saved directly in `outputs/`. Check:
```text
outputs/plots/<run_name>/
```

Also note:
- `--save-per-fold` saves per-fold CSVs (not per-fold PNGs)
- per-fold PNGs require `save_per_fold_equity=True` in `scripts/backtest_run.py`

---

### 3) `backtest_run.py` cannot find data
Defaults expect:
```text
scripts/data/SPY.csv
```

Run:
```bash
python scripts/pull_data.py
```
This script now saves to the correct folder (`scripts/data/`).

---

### 4) `rolling_splits.py` path issues on another machine
The script was updated to use `Path(__file__).resolve().parents[1]` and should be portable.

If needed, you can always override paths explicitly:
```bash
python scripts/rolling_splits.py --input-dir scripts/data --output-csv scripts/splits_train_test/walkforward_plan_9_1.csv
```

---

## Planned milestones
- Week 1: Walk-forward engine + metrics + leakage tests
- Week 2: Options chain + implied volatility (IV) panel
- Week 3: Surface smoothing baseline (spline/SVI/kernel)
- Week 4: Heston calibration + comparison vs. baseline

---

## Notes
This project is intentionally small and modular:
- `src/qwf/*` contains reusable research logic
- `scripts/*` contains I/O and command-line entry points
- outputs and plots are saved to disk for inspection / iteration

It is designed as a clean foundation for future work (intraday signals, options data, IV surfaces, and Heston calibration).

---

## End of week one (reflection)

**Week 1 goal:** build a tiny research framework that is *honest* (no lookahead), reproducible, and easy to extend.

### What I built 
- A simple end-to-end pipeline: **data → splits → walk-forward backtest → metrics → plots**
- A walk-forward plan generator that uses **calendar months** and maps to trading days.
- A minimal mean-reversion signal (Z-score) with **lagged execution** (`pos(t-1) * ret(t)`).
- A report layer that saves CSV outputs + stitched equity/drawdown plots.
- Guardrails and tests for typical time-series mistakes (leakage/lookahead).

### “Realism checks” I added 
- **Benchmark comparison:** strategy equity vs **SPY buy & hold** over the same stitched test timeline.
- **Transaction costs (optional):** `--cost-bps-per-turnover` penalizes turnover  
  (`turnover = abs(pos - pos.shift(1))`, `cost = turnover * bps/10000`).
- **Mini experiment runner:** small parameter sweep to avoid tuning-by-feel and to keep a record of experiments.

### What I learned (so far) 
- The hardest part is making the pipeline **correct by construction**:
  - time indexing discipline,
  - lag-only application,
  - fold-local accounting,
  - and tests that fail loudly when something is off.
- I also learned how much *presentation* matters: stitched curves + benchmark + a simple grid table
  seems much more convincing.

### Week 2 direction 
Next step is to keep the same philosophy (small + clean + testable) but move to options:
- pull an options chain dataset (or a small static sample),
- build an IV panel and a baseline smoothing method,
- and only then go for full surface modeling and Heston calibration.

*Note: (This README section is a living log — I’m treating it as my “lab notebook” for the project.)*
