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


def test_run_xs_ablation_writes_summary_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo_xs"
    out_dir = tmp_path / "outputs"
    run_name = "ablation_test"
    run_dir = out_dir / "runs" / run_name
    _write_synthetic_xs_directory(data_dir)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_xs_ablation.py",
            "--input-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--run-name",
            run_name,
            "--model-names",
            "linear,ridge",
            "--ridge-alphas",
            "1.0",
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

    summary_path = run_dir / "ablation_summary.csv"
    config_path = run_dir / "run_config.json"
    registry_path = out_dir / "run_registry.csv"
    summary_df = pd.read_csv(summary_path)
    registry_df = pd.read_csv(registry_path)

    assert summary_path.exists()
    assert config_path.exists()
    assert (run_dir / "plan.csv").exists()
    assert registry_path.exists()
    assert len(summary_df) == 2
    assert set(summary_df["model_name"]) == {"linear", "ridge"}
    assert (registry_df["run_name"] == run_name).sum() == 2
    assert (registry_df["run_type"] == "ablation").any()
    assert set(
        [
            "model_name",
            "parameters",
            "horizon",
            "k",
            "normalization",
            "smoothing",
            "n_tickers",
            "n_traded_days",
            "n_constant_score_days",
            "mean_ic",
            "ic_ir",
            "mean_spread",
            "total_return_net",
            "cagr_net",
            "sharpe_net",
            "sortino_net",
            "max_drawdown_net",
            "mean_turnover",
        ]
    ).issubset(summary_df.columns)
