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


def plot_time_series(
    series_df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: list[str],
    title: str,
    y_label: str,
    out_path: Path,
    labels: list[str] | None = None,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    series = series_df.copy()
    series[x_col] = pd.to_datetime(series[x_col])
    resolved_labels = labels or y_cols

    for y_col, label in zip(y_cols, resolved_labels, strict=True):
        ax.plot(series[x_col], series[y_col], label=label)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True)
    if len(y_cols) > 1:
        ax.legend()
    _save(fig, out_path)


def plot_rolling_ic(
    ic_daily: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    date_col: str = "date",
    ic_col: str = "ic_pearson",
    window: int = 20,
) -> None:
    if date_col not in ic_daily.columns or ic_col not in ic_daily.columns:
        raise ValueError(f"IC daily data must contain '{date_col}' and '{ic_col}'")

    ic_plot = ic_daily[[date_col, ic_col]].copy()
    ic_plot[date_col] = pd.to_datetime(ic_plot[date_col])
    ic_plot[ic_col] = pd.to_numeric(ic_plot[ic_col], errors="coerce")
    ic_plot["rolling_ic"] = ic_plot[ic_col].rolling(window, min_periods=max(3, window // 3)).mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ic_plot[date_col], ic_plot[ic_col], color="0.75", linewidth=1.0, label="Daily IC")
    ax.plot(ic_plot[date_col], ic_plot["rolling_ic"], color="tab:blue", linewidth=1.8, label=f"Rolling mean ({window}d)")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.grid(True)
    ax.legend()
    _save(fig, out_path)


def plot_long_short_spread_cumulative(
    spread_daily: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    date_col: str = "date",
) -> None:
    required_cols = [date_col, "long_mean_ret", "short_mean_ret", "spread"]
    missing = [col for col in required_cols if col not in spread_daily.columns]
    if missing:
        raise ValueError(f"Spread daily data is missing required columns: {missing}")

    plot_df = spread_daily[required_cols].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    for col in ["long_mean_ret", "short_mean_ret", "spread"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        plot_df[col] = (1.0 + plot_df[col]).cumprod()

    plot_time_series(
        plot_df,
        x_col=date_col,
        y_cols=["long_mean_ret", "short_mean_ret", "spread"],
        labels=["Long leg", "Short leg", "Long-short spread"],
        title=title,
        y_label="Cumulative Return",
        out_path=out_path,
    )


def plot_ticker_contribution(
    ticker_summary: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    ticker_col: str = "ticker",
    contribution_col: str = "total_contribution",
) -> None:
    required_cols = [ticker_col, contribution_col]
    missing = [col for col in required_cols if col not in ticker_summary.columns]
    if missing:
        raise ValueError(f"Ticker summary is missing required columns: {missing}")

    plot_df = ticker_summary[[ticker_col, contribution_col]].copy()
    plot_df[contribution_col] = pd.to_numeric(plot_df[contribution_col], errors="coerce").fillna(0.0)
    plot_df = plot_df.sort_values(contribution_col, ascending=False, kind="mergesort").reset_index(drop=True)
    colors = ["tab:green" if value >= 0.0 else "tab:red" for value in plot_df[contribution_col]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(plot_df[ticker_col].astype(str), plot_df[contribution_col], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Total Contribution")
    ax.grid(True, axis="y")
    _save(fig, out_path)


def save_xs_report_plots(
    portfolio_daily: pd.DataFrame,
    ic_daily: pd.DataFrame,
    spread_daily: pd.DataFrame,
    ticker_summary: pd.DataFrame,
    *,
    out_dir: str | Path,
    run_name: str,
    rolling_ic_window: int = 20,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = portfolio_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    equity = pd.Series(daily["equity"].to_numpy(), index=daily["date"], name="equity")
    drawdown = metrics.drawdown_from_equity(equity)

    paths = {
        "equity": out_dir / f"{run_name}_xs_equity.png",
        "daily_ic": out_dir / f"{run_name}_daily_ic.png",
        "rolling_ic": out_dir / f"{run_name}_rolling_ic.png",
        "drawdown": out_dir / f"{run_name}_drawdown.png",
        "long_short_spread": out_dir / f"{run_name}_long_short_spread.png",
        "ticker_contribution": out_dir / f"{run_name}_ticker_contribution.png",
    }

    plot_equity(
        equity,
        title=f"{run_name} - Net Equity",
        out_path=paths["equity"],
    )
    plot_time_series(
        ic_daily.fillna({"ic_pearson": 0.0}),
        x_col="date",
        y_cols=["ic_pearson"],
        title=f"{run_name} - Daily Pearson IC",
        y_label="IC",
        out_path=paths["daily_ic"],
    )
    plot_rolling_ic(
        ic_daily,
        out_path=paths["rolling_ic"],
        title=f"{run_name} - Rolling Pearson IC",
        window=rolling_ic_window,
    )
    plot_drawdown(
        drawdown,
        title=f"{run_name} - Net Drawdown",
        out_path=paths["drawdown"],
    )
    plot_long_short_spread_cumulative(
        spread_daily,
        out_path=paths["long_short_spread"],
        title=f"{run_name} - Long vs Short vs Spread",
    )
    plot_ticker_contribution(
        ticker_summary,
        out_path=paths["ticker_contribution"],
        title=f"{run_name} - Total Contribution by Ticker",
    )

    return paths


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
