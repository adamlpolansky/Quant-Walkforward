from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qwf.data import load_price_panel_from_directory
from qwf.experiments import run_cross_sectional_walkforward_experiment
from qwf.features import DAILY_FEATURE_COLUMNS, make_daily_features
from qwf.labels import add_forward_return_label
from qwf.metrics import benchmark_perf_summary, build_xs_benchmark_curves
from qwf.models import SUPPORTED_MODEL_NAMES
from qwf.portfolio import summarize_ticker_selection
from qwf.reporting.plots import save_xs_report_plots
from qwf.run_management import (
    build_registry_row,
    current_utc_timestamp,
    init_run_paths,
    append_run_registry,
    save_plan_snapshot,
    update_champion_files,
    write_run_config,
)
from qwf.signals import SUPPORTED_SCORE_NORMALIZATIONS, SUPPORTED_SCORE_SMOOTHING
from qwf.splits import make_global_walkforward_plan_from_dates

ROOT = Path(__file__).resolve().parents[1]


def _parse_tickers(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tickers = [item.strip() for item in value.split(",") if item.strip()]
    return tickers or None


def _build_model_params(args: argparse.Namespace) -> dict[str, float]:
    params: dict[str, float] = {}
    if args.model_name in {"ridge", "lasso", "elasticnet"}:
        params["alpha"] = float(args.alpha)
    if args.model_name == "elasticnet" and args.l1_ratio is not None:
        params["l1_ratio"] = float(args.l1_ratio)
    return params


def _format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.6f}"


def parse_args() -> argparse.Namespace:
    default_input = ROOT / "scripts" / "data"
    default_out = ROOT / "outputs"

    p = argparse.ArgumentParser(description="Run the week-2 cross-sectional baseline model.")
    p.add_argument("--input-dir", type=Path, default=default_input, help="Directory with one daily CSV per ticker")
    p.add_argument("--plan", type=Path, default=None, help="Optional global walk-forward plan CSV")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Output root directory")
    p.add_argument("--run-name", type=str, default="xs_ridge_1d_v1", help="Prefix for output files")
    p.add_argument("--date-col", type=str, default="Date")
    p.add_argument("--tickers", type=str, default=None, help="Optional comma-separated subset, for example SPY,QQQ,IWM")
    p.add_argument("--recursive", action="store_true", help="Search for CSV files recursively")
    p.add_argument("--label-horizon", type=int, default=1)
    p.add_argument("--model-name", type=str, default="ridge", choices=SUPPORTED_MODEL_NAMES)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--l1-ratio", type=float, default=None, help="ElasticNet l1_ratio. Ignored for other models.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--score-normalization", type=str, default="none", choices=SUPPORTED_SCORE_NORMALIZATIONS)
    p.add_argument("--score-smoothing", type=str, default="none", choices=SUPPORTED_SCORE_SMOOTHING)
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
    model_params = _build_model_params(args)
    timestamp = current_utc_timestamp()
    run_paths = init_run_paths(args.out_dir, args.run_name)

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

    plan_snapshot_path = run_paths.run_dir / "plan.csv"
    if args.plan is None:
        plan = make_global_walkforward_plan_from_dates(
            panel["date"].drop_duplicates().sort_values(),
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            start_date=args.start_date,
            output_csv=plan_snapshot_path,
        )
    else:
        plan = pd.read_csv(
            args.plan,
            parse_dates=["train_start", "train_end", "test_start", "test_end"],
        )
        save_plan_snapshot(plan, plan_snapshot_path)

    results = run_cross_sectional_walkforward_experiment(
        panel,
        plan,
        feature_cols=DAILY_FEATURE_COLUMNS,
        label_col=label_col,
        label_horizon=args.label_horizon,
        model_name=args.model_name,
        model_params=model_params,
        alpha=args.alpha,
        k=args.k,
        score_normalization=args.score_normalization,
        score_smoothing=args.score_smoothing,
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
    summary["plan_path"] = str(plan_snapshot_path)
    ticker_summary = summarize_ticker_selection(portfolio_detail, score_col=summary["score_col"])
    benchmark_curves = build_xs_benchmark_curves(
        predictions,
        portfolio_daily,
        label_col=label_col,
    )
    summary.update(benchmark_perf_summary(benchmark_curves, return_col="spy_return", prefix="spy"))
    summary.update(benchmark_perf_summary(benchmark_curves, return_col="eq_universe_return", prefix="eq_universe"))
    summary["strategy_minus_spy_total_return"] = (
        float(summary["total_return_net"] - summary["spy_total_return"])
        if pd.notna(summary["total_return_net"]) and pd.notna(summary["spy_total_return"])
        else float("nan")
    )
    summary["strategy_minus_eq_universe_total_return"] = (
        float(summary["total_return_net"] - summary["eq_universe_total_return"])
        if pd.notna(summary["total_return_net"]) and pd.notna(summary["eq_universe_total_return"])
        else float("nan")
    )

    pred_path = run_paths.run_dir / "predictions.csv"
    detail_path = run_paths.run_dir / "portfolio_detail.csv"
    daily_path = run_paths.run_dir / "portfolio_daily.csv"
    ic_path = run_paths.run_dir / "ic_daily.csv"
    spread_path = run_paths.run_dir / "spread_daily.csv"
    ticker_summary_path = run_paths.run_dir / "ticker_summary.csv"
    summary_path = run_paths.run_dir / "xs_summary.json"

    predictions.to_csv(pred_path, index=False)
    portfolio_detail.to_csv(detail_path, index=False)
    portfolio_daily.to_csv(daily_path, index=False)
    ic_daily.to_csv(ic_path, index=False)
    spread_daily.to_csv(spread_path, index=False)
    ticker_summary.to_csv(ticker_summary_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path = write_run_config(
        run_paths,
        run_name=args.run_name,
        run_type="single_run",
        timestamp=timestamp,
        input_dir=args.input_dir,
        repo_root=ROOT,
        config={
            "model_name": args.model_name,
            "model_params": model_params,
            "label_horizon": args.label_horizon,
            "k": args.k,
            "score_normalization": args.score_normalization,
            "score_smoothing": args.score_smoothing,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "step_months": args.step_months,
            "start_date": args.start_date,
            "cost_bps_per_turnover": args.cost_bps_per_turnover,
            "date_col": args.date_col,
            "tickers": _parse_tickers(args.tickers),
            "recursive": bool(args.recursive),
            "plan_path": str(plan_snapshot_path),
        },
    )
    registry_path = append_run_registry(
        run_paths.registry_path,
        [
            build_registry_row(
                timestamp=timestamp,
                run_name=args.run_name,
                run_type="single_run",
                path_to_run_dir=run_paths.run_dir,
                model_name=summary["model_name"],
                parameters=summary["model_params"],
                horizon=summary["label_horizon"],
                k=summary["k"],
                normalization=summary["score_normalization"],
                smoothing=summary["score_smoothing"],
                mean_ic=summary["mean_ic"],
                mean_spread=summary["mean_spread"],
                total_return_net=summary["total_return_net"],
                sharpe_net=summary["sharpe_net"],
                sortino_net=summary["sortino_net"],
                max_drawdown_net=summary["max_drawdown_net"],
                n_constant_score_days=summary["n_constant_score_days"],
            )
        ],
    )
    champion_paths = update_champion_files(run_paths.registry_path, run_paths.champions_dir)

    plot_paths = save_xs_report_plots(
        portfolio_daily,
        ic_daily,
        spread_daily,
        ticker_summary,
        benchmark_curves=benchmark_curves,
        out_dir=run_paths.plots_dir,
        run_name=args.run_name,
    )
    best_ticker = ticker_summary.iloc[0]
    worst_ticker = ticker_summary.iloc[-1]
    if summary["n_constant_score_days"] > 0:
        print(
            f"warning: {summary['n_constant_score_days']} dates had constant scores across all tickers and were skipped"
        )

    print(
        "Run saved:\n"
        f"- run dir: {run_paths.run_dir}\n"
        f"- config: {config_path}\n"
        f"- registry: {registry_path}\n"
        f"- summary: {summary_path}"
    )
    print(
        "Files saved:\n"
        f"- {pred_path}\n"
        f"- {detail_path}\n"
        f"- {daily_path}\n"
        f"- {ic_path}\n"
        f"- {spread_path}\n"
        f"- {ticker_summary_path}\n"
        f"- {plan_snapshot_path}"
    )
    print(
        "Plots saved:\n"
        f"- {plot_paths['equity']}\n"
        f"- {plot_paths['daily_ic']}\n"
        f"- {plot_paths['rolling_ic']}\n"
        f"- {plot_paths['drawdown']}\n"
        f"- {plot_paths['long_short_spread']}\n"
        f"- {plot_paths['ticker_contribution']}\n"
        f"- {plot_paths['benchmark_overlay']}"
    )
    if champion_paths:
        print(
            "Champions updated:\n"
            + "\n".join(f"- {path}" for path in champion_paths)
        )
    benchmark_lines = [
        f"- equal-weight universe total return: {_format_metric(summary['eq_universe_total_return'])}",
        f"- strategy minus equal-weight universe total return: {_format_metric(summary['strategy_minus_eq_universe_total_return'])}",
    ]
    if pd.notna(summary["spy_total_return"]):
        benchmark_lines.extend(
            [
                f"- SPY total return: {_format_metric(summary['spy_total_return'])}",
                f"- strategy minus SPY total return: {_format_metric(summary['strategy_minus_spy_total_return'])}",
            ]
        )
    else:
        benchmark_lines.append("- SPY benchmark: unavailable (SPY not in run universe)")
    run_summary_lines = [
        f"- model: {summary['model_name']}",
        f"- model params: {json.dumps(summary['model_params'], sort_keys=True)}",
        f"- label horizon: {summary['label_horizon']}d",
        f"- score normalization: {summary['score_normalization']}",
        f"- score smoothing: {summary['score_smoothing']}",
        f"- tickers: {summary['n_tickers']}",
        f"- date range: {summary['date_start']} to {summary['date_end']}",
        f"- folds: {summary['n_folds']}",
        f"- traded days: {summary['n_traded_days']}",
        f"- constant-score days skipped: {summary['n_constant_score_days']}",
        f"- mean IC: {summary['mean_ic']:.6f}",
        f"- mean spread: {summary['mean_spread']:.6f}",
        f"- total return net: {summary['total_return_net']:.6f}",
        f"- sharpe net: {summary['sharpe_net']:.6f}",
        f"- sortino net: {summary['sortino_net']:.6f}",
        f"- max drawdown net: {summary['max_drawdown_net']:.6f}",
        *benchmark_lines,
        f"- best contributor: {best_ticker['ticker']} ({best_ticker['total_contribution']:.6f})",
        f"- worst contributor: {worst_ticker['ticker']} ({worst_ticker['total_contribution']:.6f})",
    ]
    print("Run summary:\n" + "\n".join(run_summary_lines))


if __name__ == "__main__":
    main()
