from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qwf.data import load_price_panel_from_directory
from qwf.experiments import run_cross_sectional_walkforward_experiment
from qwf.features import DAILY_FEATURE_COLUMNS, make_daily_features
from qwf.labels import add_forward_return_label
from qwf.splits import make_global_walkforward_plan_from_dates

ROOT = Path(__file__).resolve().parents[1]


def _parse_tickers(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tickers = [item.strip() for item in value.split(",") if item.strip()]
    return tickers or None


def _save_line_plot(series_df: pd.DataFrame, *, x_col: str, y_col: str, title: str, y_label: str, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series_df[x_col], series_df[y_col])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_input = ROOT / "scripts" / "data"
    default_out = ROOT / "outputs"

    p = argparse.ArgumentParser(description="Run the week-2 cross-sectional ridge baseline.")
    p.add_argument("--input-dir", type=Path, default=default_input, help="Directory with one daily CSV per ticker")
    p.add_argument("--plan", type=Path, default=None, help="Optional global walk-forward plan CSV")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Output directory")
    p.add_argument("--run-name", type=str, default="xs_ridge_1d_v1", help="Prefix for output files")
    p.add_argument("--date-col", type=str, default="Date")
    p.add_argument("--tickers", type=str, default=None, help="Optional comma-separated subset, for example SPY,QQQ,IWM")
    p.add_argument("--recursive", action="store_true", help="Search for CSV files recursively")
    p.add_argument("--label-horizon", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--train-months", type=int, default=9)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--start-date", type=str, default="2018-01-01")
    p.add_argument(
        "--cost-bps-per-turnover",
        type=float,
        default=0.0,
        help="Transaction cost in bps per one-way turnover. Default 0.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_price_panel_from_directory(
        args.input_dir,
        date_col=args.date_col,
        tickers=_parse_tickers(args.tickers),
        recursive=args.recursive,
        intersect_dates=True,
    )
    aligned_tickers = int(panel["ticker"].nunique())
    shared_dates = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    print(
        "Aligned panel:\n"
        f"- tickers: {aligned_tickers}\n"
        f"- shared dates: {len(shared_dates)}\n"
        f"- date range: {shared_dates.iloc[0].date()} to {shared_dates.iloc[-1].date()}"
    )
    if aligned_tickers < max(2, 2 * args.k):
        print(
            "Warning:\n"
            f"- requested k={args.k}, but only {aligned_tickers} tickers remain after alignment\n"
            "- the daily portfolio builder will clip k to the available cross-section"
        )

    panel = make_daily_features(panel)
    label_col = f"label_{args.label_horizon}d_fwd"
    panel = add_forward_return_label(panel, horizon=args.label_horizon, label_col=label_col)

    plan_path = args.plan
    if plan_path is None:
        plan_path = args.out_dir / f"{args.run_name}_auto_plan.csv"
        plan = make_global_walkforward_plan_from_dates(
            panel["date"].drop_duplicates().sort_values(),
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            start_date=args.start_date,
            output_csv=plan_path,
        )
    else:
        plan = pd.read_csv(
            plan_path,
            parse_dates=["train_start", "train_end", "test_start", "test_end"],
        )

    results = run_cross_sectional_walkforward_experiment(
        panel,
        plan,
        feature_cols=DAILY_FEATURE_COLUMNS,
        label_col=label_col,
        label_horizon=args.label_horizon,
        alpha=args.alpha,
        k=args.k,
        cost_bps_per_turnover=args.cost_bps_per_turnover,
    )

    predictions = results["predictions"]
    portfolio_detail = results["portfolio_detail"]
    portfolio_daily = results["portfolio_daily"]
    ic_daily = results["ic_daily"]
    spread_daily = results["spread_daily"]
    summary = dict(results["summary"])
    summary["date_start"] = str(panel["date"].min().date())
    summary["date_end"] = str(panel["date"].max().date())
    summary["plan_path"] = str(plan_path)

    pred_path = args.out_dir / f"{args.run_name}_predictions.csv"
    detail_path = args.out_dir / f"{args.run_name}_portfolio_detail.csv"
    daily_path = args.out_dir / f"{args.run_name}_portfolio_daily.csv"
    ic_path = args.out_dir / f"{args.run_name}_ic_daily.csv"
    spread_path = args.out_dir / f"{args.run_name}_spread_daily.csv"
    summary_path = args.out_dir / f"{args.run_name}_xs_summary.json"

    predictions.to_csv(pred_path, index=False)
    portfolio_detail.to_csv(detail_path, index=False)
    portfolio_daily.to_csv(daily_path, index=False)
    ic_daily.to_csv(ic_path, index=False)
    spread_daily.to_csv(spread_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots_dir = args.out_dir / "plots"
    equity_plot_path = plots_dir / f"{args.run_name}_xs_equity.png"
    ic_plot_path = plots_dir / f"{args.run_name}_daily_ic.png"

    _save_line_plot(
        portfolio_daily,
        x_col="date",
        y_col="equity",
        title=f"{args.run_name} - Net Equity",
        y_label="Equity",
        out_path=equity_plot_path,
    )
    _save_line_plot(
        ic_daily.fillna({"ic_pearson": 0.0}),
        x_col="date",
        y_col="ic_pearson",
        title=f"{args.run_name} - Daily Pearson IC",
        y_label="IC",
        out_path=ic_plot_path,
    )

    print(f"Saved:\n- {pred_path}\n- {detail_path}\n- {daily_path}\n- {ic_path}\n- {spread_path}\n- {summary_path}")
    print(f"Plots saved:\n- {equity_plot_path}\n- {ic_plot_path}")
    print(
        "Run summary:\n"
        f"- tickers: {summary['n_tickers']}\n"
        f"- date range: {summary['date_start']} to {summary['date_end']}\n"
        f"- folds: {summary['n_folds']}\n"
        f"- traded days: {summary['n_traded_days']}\n"
        f"- mean IC: {summary['mean_ic']:.6f}\n"
        f"- mean spread: {summary['mean_spread']:.6f}\n"
        f"- total return net: {summary['total_return_net']:.6f}\n"
        f"- sharpe net: {summary['sharpe_net']:.6f}\n"
        f"- sortino net: {summary['sortino_net']:.6f}\n"
        f"- max drawdown net: {summary['max_drawdown_net']:.6f}"
    )


if __name__ == "__main__":
    main()
