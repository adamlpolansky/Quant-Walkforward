from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_xs_directory(output_dir: Path, *, n_days: int = 504, n_tickers: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    for idx in range(n_tickers):
        ticker = f"TEST{idx + 1:02d}"
        x = np.arange(n_days, dtype=float)
        drift = 0.0003 + 0.00005 * idx
        seasonal = 0.006 * np.sin(x / 9.0 + idx)
        returns = drift + seasonal
        close = 100.0 * np.exp(np.cumsum(returns))
        prev_close = np.r_[close[0], close[:-1]]
        open_ = prev_close * (1.0 + 0.001 * np.cos(x / 11.0 + idx))
        high = np.maximum(open_, close) * 1.003
        low = np.minimum(open_, close) * 0.997
        volume = (800_000 + idx * 75_000 + 25_000 * (1.0 + np.sin(x / 13.0 + idx))).round().astype(int)

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
        df.to_csv(output_dir / f"{ticker}.csv", index=False)


def test_run_xs_strategy_ablation_writes_summary_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo_xs"
    out_dir = tmp_path / "outputs"
    run_name = "strategy_ablation_test"
    _write_synthetic_xs_directory(data_dir)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_xs_strategy_ablation.py",
            "--input-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--run-name",
            run_name,
            "--model-name",
            "linear",
            "--label-horizons",
            "1,3",
            "--k-values",
            "2",
            "--score-normalizations",
            "none,rank_to_minus1_plus1",
            "--score-smoothing-methods",
            "none",
            "--train-months",
            "12",
            "--test-months",
            "3",
            "--step-months",
            "3",
            "--start-date",
            "2020-01-01",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = out_dir / f"{run_name}_strategy_ablation_summary.csv"
    summary_df = pd.read_csv(summary_path)

    assert summary_path.exists()
    assert len(summary_df) == 4
    assert set(summary_df["model_name"]) == {"linear"}
    assert set(summary_df["horizon"]) == {1, 3}
    assert set(summary_df["normalization"]) == {"none", "rank_to_minus1_plus1"}
    assert set(
        [
            "parameters",
            "horizon",
            "k",
            "normalization",
            "smoothing",
            "n_constant_score_days",
            "mean_ic",
            "mean_spread",
            "total_return_net",
            "sharpe_net",
            "sortino_net",
            "max_drawdown_net",
            "mean_turnover",
        ]
    ).issubset(summary_df.columns)
