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
from qwf.labels import add_forward_return_label
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
DEFAULT_MODEL_NAMES = ",".join(SUPPORTED_MODEL_NAMES)
DEFAULT_RIDGE_ALPHAS = "0.1,1.0,10.0"
DEFAULT_LASSO_ALPHAS = "0.0001,0.001,0.01"
DEFAULT_ELASTICNET_ALPHAS = "0.0001,0.001"
DEFAULT_ELASTICNET_L1_RATIOS = "0.25,0.5,0.75"


@dataclass(frozen=True)
class AblationSpec:
    model_name: str
    model_params: dict[str, float]


def _parse_csv_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tickers(value: str | None) -> list[str] | None:
    tickers = _parse_csv_tokens(value)
    return tickers or None


def _parse_model_names(value: str) -> list[str]:
    model_names = [item.lower() for item in _parse_csv_tokens(value)]
    if not model_names:
        raise ValueError("At least one model name is required")

    invalid = sorted(set(model_names) - set(SUPPORTED_MODEL_NAMES))
    if invalid:
        raise ValueError(f"Unsupported model names: {invalid}. Supported models: {list(SUPPORTED_MODEL_NAMES)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


def _parse_float_list(value: str) -> list[float]:
    values = _parse_csv_tokens(value)
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(item) for item in values]


def _format_experiment_name(spec: AblationSpec) -> str:
    if not spec.model_params:
        return spec.model_name

    parts = [spec.model_name]
    for key, value in sorted(spec.model_params.items()):
        parts.append(f"{key}_{value:g}")
    return "__".join(parts)


def build_ablation_specs(
    *,
    model_names: Sequence[str],
    ridge_alphas: Sequence[float],
    lasso_alphas: Sequence[float],
    elasticnet_alphas: Sequence[float],
    elasticnet_l1_ratios: Sequence[float],
) -> list[AblationSpec]:
    specs: list[AblationSpec] = []

    for model_name in model_names:
        if model_name == "linear":
            specs.append(AblationSpec(model_name="linear", model_params={}))
        elif model_name == "ridge":
            specs.extend(
                AblationSpec(model_name="ridge", model_params={"alpha": float(alpha)})
                for alpha in ridge_alphas
            )
        elif model_name == "lasso":
            specs.extend(
                AblationSpec(model_name="lasso", model_params={"alpha": float(alpha)})
                for alpha in lasso_alphas
            )
        elif model_name == "elasticnet":
            for alpha in elasticnet_alphas:
                for l1_ratio in elasticnet_l1_ratios:
                    specs.append(
                        AblationSpec(
                            model_name="elasticnet",
                            model_params={"alpha": float(alpha), "l1_ratio": float(l1_ratio)},
                        )
                    )
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small week-2 cross-sectional model ablation grid.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with one daily CSV per ticker")
    parser.add_argument("--plan", type=Path, default=None, help="Optional global walk-forward plan CSV")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root directory")
    parser.add_argument("--run-name", type=str, default="etf_xs_ablation_v1", help="Prefix for summary outputs")
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--tickers", type=str, default=None, help="Optional comma-separated subset, for example SPY,QQQ")
    parser.add_argument("--recursive", action="store_true", help="Search for CSV files recursively")
    parser.add_argument("--label-horizon", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--score-normalization", type=str, default="none", choices=SUPPORTED_SCORE_NORMALIZATIONS)
    parser.add_argument("--score-smoothing", type=str, default="none", choices=SUPPORTED_SCORE_SMOOTHING)
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
    parser.add_argument(
        "--model-names",
        type=str,
        default=DEFAULT_MODEL_NAMES,
        help="Comma-separated model families to include. Supported: linear,ridge,lasso,elasticnet",
    )
    parser.add_argument("--ridge-alphas", type=str, default=DEFAULT_RIDGE_ALPHAS)
    parser.add_argument("--lasso-alphas", type=str, default=DEFAULT_LASSO_ALPHAS)
    parser.add_argument("--elasticnet-alphas", type=str, default=DEFAULT_ELASTICNET_ALPHAS)
    parser.add_argument("--elasticnet-l1-ratios", type=str, default=DEFAULT_ELASTICNET_L1_RATIOS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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

    selected_model_names = _parse_model_names(args.model_names)
    specs = build_ablation_specs(
        model_names=selected_model_names,
        ridge_alphas=_parse_float_list(args.ridge_alphas) if "ridge" in selected_model_names else [],
        lasso_alphas=_parse_float_list(args.lasso_alphas) if "lasso" in selected_model_names else [],
        elasticnet_alphas=_parse_float_list(args.elasticnet_alphas) if "elasticnet" in selected_model_names else [],
        elasticnet_l1_ratios=_parse_float_list(args.elasticnet_l1_ratios) if "elasticnet" in selected_model_names else [],
    )
    if not specs:
        raise ValueError("No ablation experiments were generated from the supplied grid")

    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(specs, start=1):
        experiment_name = _format_experiment_name(spec)
        print(
            f"[{idx}/{len(specs)}] "
            f"{experiment_name} with params {json.dumps(spec.model_params, sort_keys=True)}"
        )
        results = run_cross_sectional_walkforward_experiment(
            panel,
            plan,
            feature_cols=DAILY_FEATURE_COLUMNS,
            label_col=label_col,
            label_horizon=args.label_horizon,
            model_name=spec.model_name,
            model_params=spec.model_params,
            k=args.k,
            score_normalization=args.score_normalization,
            score_smoothing=args.score_smoothing,
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
                "ic_ir": summary["ic_ir"],
                "mean_spread": summary["mean_spread"],
                "total_return_net": summary["total_return_net"],
                "cagr_net": summary["cagr_net"],
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
        "ic_ir",
        "mean_spread",
        "total_return_net",
        "cagr_net",
        "sharpe_net",
        "sortino_net",
        "max_drawdown_net",
        "mean_turnover",
    ]
    summary_df = summary_df[ordered_cols]

    summary_path = run_paths.run_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    config_path = write_run_config(
        run_paths,
        run_name=args.run_name,
        run_type="ablation",
        timestamp=timestamp,
        input_dir=args.input_dir,
        repo_root=ROOT,
        config={
            "model_names": selected_model_names,
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
            "ridge_alphas": _parse_float_list(args.ridge_alphas) if "ridge" in selected_model_names else [],
            "lasso_alphas": _parse_float_list(args.lasso_alphas) if "lasso" in selected_model_names else [],
            "elasticnet_alphas": _parse_float_list(args.elasticnet_alphas) if "elasticnet" in selected_model_names else [],
            "elasticnet_l1_ratios": _parse_float_list(args.elasticnet_l1_ratios) if "elasticnet" in selected_model_names else [],
            "plan_path": str(plan_snapshot_path),
        },
    )
    registry_rows = [
        build_registry_row(
            timestamp=timestamp,
            run_name=args.run_name,
            run_type="ablation",
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
        f"{summary_df.sort_values('sharpe_net', ascending=False).head(5).to_string(index=False)}"
    )


if __name__ == "__main__":
    main()
