# Quant-Walkforward

Week 1 still exists as the original single-asset daily walk-forward engine. A minimal week-2 baseline now also exists for local multi-asset daily cross-sectional experiments built from per-ticker CSV files.

## Install

Use editable install from the repo root. The scripts assume `qwf` is importable as a normal package.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

After install, these entry points should work without `PYTHONPATH` or `sys.path` hacks:

```bash
python scripts/backtest_run.py --help
python scripts/param_sweep.py --help
python scripts/make_synthetic_xs_data.py --help
python scripts/rolling_splits_global.py --help
python scripts/run_xs_model.py --help
```

## Week-1 workflow

Typical local flow:

```bash
python scripts/pull_data.py
python scripts/rolling_splits.py --train-months 9 --test-months 1
python scripts/backtest_run.py --save-per-fold
pytest -q
```

Key scripts:

- `scripts/pull_data.py`: download daily SPY data by default, or a small multi-ticker ETF universe into per-ticker CSV files
- `scripts/rolling_splits.py`: generate a walk-forward plan CSV from local input files
- `scripts/backtest_run.py`: run the week-1 walk-forward backtest and save reports
- `scripts/param_sweep.py`: run a small parameter grid over the current week-1 engine

## Week-2 baseline

The repo now includes a minimal daily cross-sectional baseline:

- long panel built from local per-ticker CSV files
- shared global calendar from the intersection of selected tickers
- simple per-ticker daily features
- one forward label: next-day close-to-close return
- pooled ridge regression across train rows
- daily top-k / bottom-k equal-weight long-short portfolio
- evaluation with daily IC, long-short spread, and portfolio performance summary

Main week-2 scripts:

- `scripts/make_synthetic_xs_data.py`: generate a deterministic offline demo universe
- `scripts/rolling_splits_global.py`: build a single global split plan from the shared calendar
- `scripts/run_xs_model.py`: load local CSVs, build features and labels, optionally auto-build a global plan, fit the ridge baseline, and save predictions plus portfolio outputs

Preferred offline demo flow:

```bash
python scripts/make_synthetic_xs_data.py
python scripts/run_xs_model.py --input-dir scripts/data/demo_xs --run-name demo_xs --train-months 12 --test-months 3 --step-months 3 --start-date 2020-01-01 --k 2
```

## Week-2 real ETF workflow

First real-data validation universe:

- `XLC`
- `XLY`
- `XLP`
- `XLE`
- `XLF`
- `XLV`
- `XLI`
- `XLB`
- `XLRE`
- `XLK`
- `XLU`
- `SPY`
- `QQQ`

Download the ETF CSV files from yfinance:

```bash
python scripts/pull_data.py --use-default-etf-universe --start-date 2018-01-01 --output-dir scripts/data/real_etf_xs
```

Run the existing week-2 cross-sectional baseline on the downloaded directory:

```bash
python scripts/run_xs_model.py --input-dir scripts/data/real_etf_xs --run-name etf_xs_v1 --train-months 12 --test-months 3 --step-months 3 --start-date 2020-01-01 --k 2
```

Expected outputs remain the same as the synthetic demo: predictions CSV, portfolio detail CSV, portfolio daily CSV, IC daily CSV, spread daily CSV, summary JSON, equity plot, and daily IC plot under `outputs/`.

This is the first real-data validation workflow for week 2. It is intentionally not production data infrastructure.

If `--plan` is omitted, `scripts/run_xs_model.py` builds a global plan automatically from the intersected panel calendar and saves it alongside the other outputs.

Week-2 outputs include:

- `<run_name>_predictions.csv`
- `<run_name>_portfolio_detail.csv`
- `<run_name>_portfolio_daily.csv`
- `<run_name>_ic_daily.csv`
- `<run_name>_spread_daily.csv`
- `<run_name>_xs_summary.json`
- `outputs/plots/<run_name>_xs_equity.png`
- `outputs/plots/<run_name>_daily_ic.png`

Week-2 summary metrics include:

- daily Pearson IC summary (`mean_ic`, `ic_std`, `ic_ir`, `n_ic_days`)
- daily long-short spread summary (`mean_spread`, `spread_std`, `n_spread_days`)
- portfolio summary (`total_return_gross`, `total_return_net`, `cagr_net`, `ann_vol_net`, `sharpe_net`, `max_drawdown_net`, `mean_turnover`, `mean_cost`, `n_traded_days`)

Turnover convention:

- one-way turnover = `0.5 * sum(abs(weight_t - weight_{t-1}))`
- cost = `turnover * cost_bps / 10000`

## Input CSV contract

The local CSV loader for week 1 lives in `src/qwf/data.py`.

Expected columns:

- required: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- optional: `Adj_Close`, `ret`, `log_ret`

Loader behavior:

- parses `Date` into a `DatetimeIndex`
- sorts by date
- removes duplicate timestamps by keeping the first row
- strips timezone info if present
- raises an informative error if required columns are missing
- computes `ret` from `price_col` if `ret` is absent
- computes `log_ret` from `price_col` if `log_ret` is absent

For the current workflow, `price_col` defaults to `Close` and `ret_col` defaults to `ret`.

## Outputs

`scripts/backtest_run.py` writes outputs under `outputs/`, including:

- `<run_name>_test_detail.csv`
- `<run_name>_fold_summary.csv`
- `<run_name>_config.json`
- `<run_name>_perf_summary.csv`
- plots under `outputs/plots/<run_name>/`

If `--save-per-fold` is set, per-fold test CSVs are also written under `outputs/folds/`.

## Testing

Run:

```bash
pytest -q
```

The test suite covers the existing week-1 lookahead/leakage guardrails, split semantics, and the new week-2 baseline modules on synthetic data.

## Scope note

Implemented now:

- week-1 single-asset walk-forward backtest
- week-2 baseline daily cross-sectional ridge workflow

Still intentionally out of scope:

- intraday logic
- options data or surface modeling
- neutralization layers and optimizers
- tree models
- advanced execution simulation
