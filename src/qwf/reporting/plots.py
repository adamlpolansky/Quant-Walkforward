# src/qwf/reporting/plots.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qwf import metrics


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity(equity: pd.Series, *, title: str, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(equity.index, equity.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True)
    _save(fig, out_path)


def plot_drawdown(dd: pd.Series, *, title: str, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dd.index, dd.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    _save(fig, out_path)


def plot_bar(df: pd.DataFrame, *, x: str, y: str, title: str, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, axis="y")
    _save(fig, out_path)


def save_report_plots(
    test_detail: pd.DataFrame,
    fold_summary: pd.DataFrame,
    *,
    out_dir: str | Path,
    run_name: str = "run",
    rolling_sharpe_window: int | None = 63,
    save_per_fold_equity: bool = False,
    max_folds: int = 24,
    benchmark_stitched: pd.DataFrame | None = None,
    strategy_label: str = "Strategy",
    benchmark_label: str = "Buy & Hold",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stitched = metrics.stitched_curve(
        test_detail,
        rolling_sharpe_window=rolling_sharpe_window,
    )

    if benchmark_stitched is not None:
        plot_equity_compare(
            stitched["equity"], benchmark_stitched["equity"],
            label_a=strategy_label, label_b=benchmark_label,
            title=f"{run_name} - Equity: Strategy vs Benchmark",
            out_path=out_dir / "equity_vs_benchmark.png",
        )
    plot_equity(
        stitched["equity"],
        title=f"{run_name} - Stitched Equity",
        out_path=out_dir / "equity_stitched.png",
    )

    plot_drawdown(
        stitched["drawdown"],
        title=f"{run_name} - Stitched Drawdown",
        out_path=out_dir / "drawdown_stitched.png",
    )

    if "rolling_sharpe" in stitched.columns:
        plot_equity(  # same helper is fine; axis labels are generic
            stitched["rolling_sharpe"],
            title=f"{run_name} - Rolling Sharpe (window={rolling_sharpe_window})",
            out_path=out_dir / "rolling_sharpe.png",
        )

    # fold-level bars (if present)
    if "sharpe" in fold_summary.columns:
        plot_bar(
            fold_summary,
            x="fold_id",
            y="sharpe",
            title=f"{run_name} - Sharpe by Fold",
            out_path=out_dir / "sharpe_by_fold.png",
        )

    if "total_return" in fold_summary.columns:
        plot_bar(
            fold_summary,
            x="fold_id",
            y="total_return",
            title=f"{run_name} - Total Return by Fold",
            out_path=out_dir / "total_return_by_fold.png",
        )

    # Optional: per-fold equity (local reset)
    if save_per_fold_equity:
        td_local = metrics.add_fold_local_curves(test_detail)
        folds = list(td_local["fold_id"].dropna().unique())[:max_folds]
        for fid in folds:
            g = td_local.loc[td_local["fold_id"] == fid]
            if g.empty:
                continue
            plot_equity(
                g["equity_local"],
                title=f"{run_name} - Fold {fid} Equity (local)",
                out_path=out_dir / "folds" / f"equity_fold_{fid}.png",
            )
            plot_drawdown(
                g["dd_local"],
                title=f"{run_name} - Fold {fid} Drawdown (local)",
                out_path=out_dir / "folds" / f"drawdown_fold_{fid}.png",
            )

def plot_equity_compare(
        equity_a: pd.Series,
        equity_b: pd.Series,
        *,
        label_a: str,
        label_b: str,
        title: str,
        out_path: Path,
) -> None:
    # Align to common index (safe if series have slight diffs)
    idx = equity_a.index.intersection(equity_b.index)
    a = equity_a.reindex(idx)
    b = equity_b.reindex(idx)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a.index, a.values, label=label_a)
    ax.plot(b.index, b.values, label=label_b)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True)
    ax.legend()
    _save(fig, out_path)