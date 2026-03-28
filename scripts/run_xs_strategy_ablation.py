from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from qwf.data import load_price_panel_from_directory
from qwf.experiments import run_cross_sectional_walkforward_experiment
from qwf.features import DAILY_FEATURE_COLUMNS, make_daily_features
from qwf.labels import add_forward_return_labels
from qwf.models import SUPPORTED_MODEL_NAMES
from qwf.run_management import (
    append_run_registry,
    build_registry_row,
    current_utc_timestamp,
    init_run_paths,
    save_plan_snapshot,
    update_champion_files,
    write_run_config,
)
from qwf.signals import SUPPORTED_SCORE_NORMALIZATIONS, SUPPORTED_SCORE_SMOOTHING
from qwf.splits import make_global_walkforward_plan_from_dates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "scripts" / "data" / "real_etf_xs"
DEFAULT_OUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class StrategySpec:
    label_horizon: int
    k: int
    score_normalization: str
    score_smoothing: str


def _parse_csv_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tickers(value: str | None) -> list[str] | None:
    tickers = _parse_csv_tokens(value)
    return tickers or None


def _parse_int_list(value: str) -> list[int]:
    values = _parse_csv_tokens(value)
    if not values:
        raise ValueError("Expected at least one integer value")
    return [int(item) for item in values]


def _parse_choice_list(value: str, *, supported: Sequence[str], label: str) -> list[str]:
    values = [item.lower() for item in _parse_csv_tokens(value)]
    if not values:
        raise ValueError(f"Expected at least one {label}")
    invalid = sorted(set(values) - set(supported))
    if invalid:
        raise ValueError(f"Unsupported {label}: {invalid}. Supported values: {list(supported)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_model_params(args: argparse.Namespace) -> dict[str, float]:
    params: dict[str, float] = {}
    if args.model_name in {"ridge", "lasso", "elasticnet"}:
        params["alpha"] = float(args.alpha)
    if args.model_name == "elasticnet" and args.l1_ratio is not None:
        params["l1_ratio"] = float(args.l1_ratio)
    return params


def _format_experiment_name(spec: StrategySpec) -> str:
    return (
        f"h{spec.label_horizon}"
        f"__k{spec.k}"
        f"__norm_{spec.score_normalization}"
        f"__smooth_{spec.score_smoothing}"
    )


def build_strategy_specs(
    *,
    label_horizons: Sequence[int],
    k_values: Sequence[int],
    score_normalizations: Sequence[str],
    score_smoothing_methods: Sequence[str],
) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for label_horizon in label_horizons:
        for k in k_values:
            for score_normalization in score_normalizations:
                for score_smoothing in score_smoothing_methods:
                    specs.append(
                        StrategySpec(
                            label_horizon=int(label_horizon),
                            k=int(k),
                            score_normalization=str(score_normalization),
                            score_smoothing=str(score_smoothing),
                        )
                    )
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small week-2 strategy-settings ablation for one model configuration.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with one daily CSV per ticker")
    parser.add_argument("--plan", type=Path, default=None, help="Optional global walk-forward plan CSV")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root directory")
    parser.add_argument("--run-name", type=str, default="etf_xs_strategy_ablation_v1", help="Prefix for summary outputs")
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--tickers", type=str, default=None, help="Optional comma-separated subset, for example SPY,QQQ")
    parser.add_argument("--recursive", action="store_true", help="Search for CSV files recursively")
    parser.add_argument("--model-name", type=str, default="elasticnet", choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio. Ignored for other models.")
    parser.add_argument("--label-horizons", type=str, default="1,3,5")
    parser.add_argument("--k-values", type=str, default="2,3,4")
    parser.add_argument("--score-normalizations", type=str, default="none,zscore,rank_to_minus1_plus1")
    parser.add_argument("--score-smoothing-methods", type=str, default="none,ema_3")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument(
        "--cost-bps-per-turnover",
        type=float,
        default=0.0,
        help="Transaction cost in bps per one-way turnover. Default 0.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    timestamp = current_utc_timestamp()
    run_paths = init_run_paths(args.out_dir, args.run_name)
    model_params = _build_model_params(args)

    panel = load_price_panel_from_directory(
        args.input_dir,
        date_col=args.date_col,
        tickers=_parse_tickers(args.tickers),
        recursive=args.recursive,
        intersect_dates=True,
    )
    aligned_tickers = int(panel["ticker"].nunique())
    shared_dates = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    k_values = _parse_int_list(args.k_values)
    print(
        "Aligned panel:\n"
        f"- tickers: {aligned_tickers}\n"
        f"- shared dates: {len(shared_dates)}\n"
        f"- date range: {shared_dates.iloc[0].date()} to {shared_dates.iloc[-1].date()}"
    )
    if aligned_tickers < max(2, 2 * max(k_values)):
        print(
            "Warning:\n"
            f"- requested max k={max(k_values)}, but only {aligned_tickers} tickers remain after alignment\n"
            "- the daily portfolio builder will clip k to the available cross-section"
        )

    feature_panel = make_daily_features(panel)
    label_horizons = _parse_int_list(args.label_horizons)
    panel_with_labels = add_forward_return_labels(feature_panel, horizons=label_horizons)

    plan_snapshot_path = run_paths.run_dir / "plan.csv"
    if args.plan is None:
        plan = make_global_walkforward_plan_from_dates(
            panel_with_labels["date"].drop_duplicates().sort_values(),
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

    specs = build_strategy_specs(
        label_horizons=label_horizons,
        k_values=k_values,
        score_normalizations=_parse_choice_list(
            args.score_normalizations,
            supported=SUPPORTED_SCORE_NORMALIZATIONS,
            label="score normalizations",
        ),
        score_smoothing_methods=_parse_choice_list(
            args.score_smoothing_methods,
            supported=SUPPORTED_SCORE_SMOOTHING,
            label="score smoothing methods",
        ),
    )
    if not specs:
        raise ValueError("No strategy-setting experiments were generated from the supplied grid")

    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(specs, start=1):
        label_col = f"label_{spec.label_horizon}d_fwd"
        experiment_name = _format_experiment_name(spec)
        print(
            f"[{idx}/{len(specs)}] {experiment_name} "
            f"for model={args.model_name} params={json.dumps(model_params, sort_keys=True)}"
        )
        results = run_cross_sectional_walkforward_experiment(
            panel_with_labels,
            plan,
            feature_cols=DAILY_FEATURE_COLUMNS,
            label_col=label_col,
            label_horizon=spec.label_horizon,
            model_name=args.model_name,
            model_params=model_params,
            k=spec.k,
            score_normalization=spec.score_normalization,
            score_smoothing=spec.score_smoothing,
            cost_bps_per_turnover=args.cost_bps_per_turnover,
        )
        summary = dict(results["summary"])
        if summary["n_constant_score_days"] > 0:
            print(
                f"warning: {experiment_name} skipped {summary['n_constant_score_days']} constant-score dates"
            )

        rows.append(
            {
                "experiment_name": experiment_name,
                "model_name": summary["model_name"],
                "parameters": json.dumps(summary["model_params"], sort_keys=True),
                "horizon": summary["label_horizon"],
                "k": summary["k"],
                "normalization": summary["score_normalization"],
                "smoothing": summary["score_smoothing"],
                "n_tickers": summary["n_tickers"],
                "n_traded_days": summary["n_traded_days"],
                "n_constant_score_days": summary["n_constant_score_days"],
                "mean_ic": summary["mean_ic"],
                "mean_spread": summary["mean_spread"],
                "total_return_net": summary["total_return_net"],
                "sharpe_net": summary["sharpe_net"],
                "sortino_net": summary["sortino_net"],
                "max_drawdown_net": summary["max_drawdown_net"],
                "mean_turnover": summary["mean_turnover"],
            }
        )

    summary_df = pd.DataFrame(rows)
    ordered_cols = [
        "experiment_name",
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
        "mean_spread",
        "total_return_net",
        "sharpe_net",
        "sortino_net",
        "max_drawdown_net",
        "mean_turnover",
    ]
    summary_df = summary_df[ordered_cols]

    summary_path = run_paths.run_dir / "strategy_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    config_path = write_run_config(
        run_paths,
        run_name=args.run_name,
        run_type="strategy_ablation",
        timestamp=timestamp,
        input_dir=args.input_dir,
        repo_root=ROOT,
        config={
            "model_name": args.model_name,
            "model_params": model_params,
            "label_horizons": label_horizons,
            "k_values": k_values,
            "score_normalizations": _parse_choice_list(
                args.score_normalizations,
                supported=SUPPORTED_SCORE_NORMALIZATIONS,
                label="score normalizations",
            ),
            "score_smoothing_methods": _parse_choice_list(
                args.score_smoothing_methods,
                supported=SUPPORTED_SCORE_SMOOTHING,
                label="score smoothing methods",
            ),
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
    registry_rows = [
        build_registry_row(
            timestamp=timestamp,
            run_name=args.run_name,
            run_type="strategy_ablation",
            experiment_name=str(row["experiment_name"]),
            path_to_run_dir=run_paths.run_dir,
            model_name=str(row["model_name"]),
            parameters=str(row["parameters"]),
            horizon=int(row["horizon"]),
            k=int(row["k"]),
            normalization=str(row["normalization"]),
            smoothing=str(row["smoothing"]),
            mean_ic=float(row["mean_ic"]),
            mean_spread=float(row["mean_spread"]),
            total_return_net=float(row["total_return_net"]),
            sharpe_net=float(row["sharpe_net"]),
            sortino_net=float(row["sortino_net"]),
            max_drawdown_net=float(row["max_drawdown_net"]),
            n_constant_score_days=int(row["n_constant_score_days"]),
        )
        for row in summary_df.to_dict(orient="records")
    ]
    registry_path = append_run_registry(run_paths.registry_path, registry_rows)
    champion_paths = update_champion_files(run_paths.registry_path, run_paths.champions_dir)

    print(
        "Run saved:\n"
        f"- run dir: {run_paths.run_dir}\n"
        f"- config: {config_path}\n"
        f"- summary: {summary_path}\n"
        f"- plan: {plan_snapshot_path}\n"
        f"- registry: {registry_path}"
    )
    if champion_paths:
        print(
            "Champions updated:\n"
            + "\n".join(f"- {path}" for path in champion_paths)
        )
    print(
        "Top results by sharpe_net:\n"
        f"{summary_df.sort_values('sharpe_net', ascending=False).head(10).to_string(index=False)}"
    )


if __name__ == "__main__":
    main()
