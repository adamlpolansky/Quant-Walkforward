from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qwf.experiments import run_cross_sectional_walkforward_experiment
from qwf.features import DAILY_FEATURE_COLUMNS, make_daily_features
from qwf.labels import add_forward_return_label

ROOT = Path(__file__).resolve().parents[1]


def _make_panel(n_days: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rows: list[dict[str, object]] = []

    for idx, ticker in enumerate(tickers):
        x = np.arange(n_days, dtype=float)
        drift = 0.05 + 0.01 * idx
        seasonal = 0.20 * np.sin(x / 5.0 + idx)
        close = 100.0 + idx * 5.0 + np.cumsum(drift + seasonal)
        volume = 1_000_000 + idx * 25_000 + 20_000 * (1.0 + np.cos(x / 6.0 + idx))

        for date, c, v in zip(dates, close, volume, strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "Open": c - 0.3,
                    "High": c + 0.6,
                    "Low": c - 0.8,
                    "Close": c,
                    "Volume": v,
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def test_run_cross_sectional_walkforward_experiment_returns_richer_summary() -> None:
    panel = make_daily_features(_make_panel())
    panel = add_forward_return_label(panel, horizon=1, label_col="label_1d_fwd")

    dates = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    plan = pd.DataFrame(
        {
            "fold_id": [0],
            "train_start": [dates.iloc[0]],
            "train_end": [dates.iloc[39]],
            "test_start": [dates.iloc[40]],
            "test_end": [dates.iloc[49]],
        }
    )

    results = run_cross_sectional_walkforward_experiment(
        panel,
        plan,
        feature_cols=DAILY_FEATURE_COLUMNS,
        label_col="label_1d_fwd",
        label_horizon=1,
        alpha=1.0,
        k=1,
        cost_bps_per_turnover=5.0,
    )

    predictions = results["predictions"]
    portfolio_detail = results["portfolio_detail"]
    portfolio_daily = results["portfolio_daily"]
    ic_daily = results["ic_daily"]
    spread_daily = results["spread_daily"]
    summary = results["summary"]

    assert not predictions.empty
    assert predictions["score"].notna().any()
    assert set(["fold_id", "score", "label_1d_fwd"]).issubset(predictions.columns)
    assert predictions["date"].min() == dates.iloc[40]
    assert predictions["date"].max() == dates.iloc[49]

    assert not portfolio_detail.empty
    assert not portfolio_daily.empty
    assert not ic_daily.empty
    assert not spread_daily.empty
    assert set(
        ["portfolio_ret_gross", "turnover", "cost", "portfolio_ret_net", "equity", "is_constant_score_day"]
    ).issubset(portfolio_daily.columns)
    assert set(["ic_pearson", "n_assets"]).issubset(ic_daily.columns)
    assert set(["long_mean_ret", "short_mean_ret", "spread"]).issubset(spread_daily.columns)
    assert summary["turnover_convention"] == "one_way_half_abs_change"
    assert summary["model_name"] == "ridge"
    assert summary["model_params"] == {"alpha": 1.0}
    assert summary["score_normalization"] == "none"
    assert summary["score_smoothing"] == "none"
    assert summary["n_constant_score_days"] == 0
    assert set(
        [
            "mean_ic",
            "ic_std",
            "mean_spread",
            "total_return_net",
            "sharpe_net",
            "sortino_net",
            "mean_turnover",
            "n_traded_days",
        ]
    ).issubset(summary.keys())


def test_week2_demo_scripts_run_end_to_end_without_network(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo_xs"
    out_dir = tmp_path / "outputs"
    run_name = "demo_xs_test"
    run_dir = out_dir / "runs" / run_name

    subprocess.run(
        [
            sys.executable,
            "scripts/make_synthetic_xs_data.py",
            "--output-dir",
            str(data_dir),
            "--num-tickers",
            "6",
            "--years",
            "2",
            "--seed",
            "11",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_xs_model.py",
            "--input-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--run-name",
            run_name,
            "--train-months",
            "12",
            "--test-months",
            "3",
            "--step-months",
            "3",
            "--start-date",
            "2020-01-01",
            "--k",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = run_dir / "xs_summary.json"
    ticker_summary_path = run_dir / "ticker_summary.csv"
    config_path = run_dir / "run_config.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ticker_summary = pd.read_csv(ticker_summary_path)
    registry = pd.read_csv(out_dir / "run_registry.csv")

    assert run_dir.exists()
    assert (run_dir / "plan.csv").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "portfolio_detail.csv").exists()
    assert (run_dir / "portfolio_daily.csv").exists()
    assert (run_dir / "ic_daily.csv").exists()
    assert (run_dir / "spread_daily.csv").exists()
    assert ticker_summary_path.exists()
    assert summary_path.exists()
    assert config_path.exists()
    assert (run_dir / "plots" / "xs_equity.png").exists()
    assert (run_dir / "plots" / "daily_ic.png").exists()
    assert (run_dir / "plots" / "rolling_ic.png").exists()
    assert (run_dir / "plots" / "drawdown.png").exists()
    assert (run_dir / "plots" / "long_short_spread.png").exists()
    assert (run_dir / "plots" / "ticker_contribution.png").exists()
    assert (out_dir / "champions" / "best_by_sharpe.json").exists()
    assert (out_dir / "champions" / "best_by_sortino.json").exists()
    assert (out_dir / "champions" / "best_by_mean_ic.json").exists()
    assert (out_dir / "champions" / "current_etf_xs_champion.json").exists()

    assert summary["n_tickers"] == 6
    assert summary["n_folds"] >= 1
    assert summary["n_traded_days"] >= 1
    assert summary["turnover_convention"] == "one_way_half_abs_change"
    assert set(
        [
            "ticker",
            "n_long_days",
            "n_short_days",
            "total_contribution",
            "selection_rate_long",
            "selection_rate_short",
        ]
    ).issubset(ticker_summary.columns)
    assert summary["score_normalization"] == "none"
    assert summary["score_smoothing"] == "none"
    assert summary["n_constant_score_days"] == 0
    assert set(["run_name", "run_type", "path_to_run_dir"]).issubset(registry.columns)
    assert (registry["run_name"] == run_name).any()


def test_run_cross_sectional_walkforward_experiment_reports_constant_score_days() -> None:
    panel = _make_panel()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        panel[col] = panel.groupby("date", sort=False)[col].transform("mean")

    panel = make_daily_features(panel)
    panel = add_forward_return_label(panel, horizon=1, label_col="label_1d_fwd")
    dates = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    plan = pd.DataFrame(
        {
            "fold_id": [0],
            "train_start": [dates.iloc[0]],
            "train_end": [dates.iloc[39]],
            "test_start": [dates.iloc[40]],
            "test_end": [dates.iloc[49]],
        }
    )

    results = run_cross_sectional_walkforward_experiment(
        panel,
        plan,
        feature_cols=DAILY_FEATURE_COLUMNS,
        label_col="label_1d_fwd",
        label_horizon=1,
        alpha=1.0,
        k=1,
    )

    portfolio_daily = results["portfolio_daily"]
    summary = results["summary"]

    assert summary["n_constant_score_days"] >= 1
    assert portfolio_daily["is_constant_score_day"].any()
    assert portfolio_daily.loc[portfolio_daily["is_constant_score_day"], "gross_exposure"].eq(0.0).all()
