from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REGISTRY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "run_name",
    "run_type",
    "experiment_name",
    "model_name",
    "parameters",
    "horizon",
    "k",
    "normalization",
    "smoothing",
    "mean_ic",
    "mean_spread",
    "total_return_net",
    "sharpe_net",
    "sortino_net",
    "max_drawdown_net",
    "n_constant_score_days",
    "path_to_run_dir",
)
CHAMPION_METRICS: tuple[tuple[str, str], ...] = (
    ("best_by_sharpe.json", "sharpe_net"),
    ("best_by_sortino.json", "sortino_net"),
    ("best_by_mean_ic.json", "mean_ic"),
)


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    runs_dir: Path
    run_dir: Path
    plots_dir: Path
    champions_dir: Path
    registry_path: Path
    config_path: Path


def current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_run_paths(output_root: str | Path, run_name: str) -> RunPaths:
    root = Path(output_root)
    runs_dir = root / "runs"
    run_dir = runs_dir / str(run_name)
    plots_dir = run_dir / "plots"
    champions_dir = root / "champions"
    registry_path = root / "run_registry.csv"
    config_path = run_dir / "run_config.json"

    plots_dir.mkdir(parents=True, exist_ok=True)
    champions_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        output_root=root,
        runs_dir=runs_dir,
        run_dir=run_dir,
        plots_dir=plots_dir,
        champions_dir=champions_dir,
        registry_path=registry_path,
        config_path=config_path,
    )


def try_git_commit_hash(repo_root: str | Path) -> str | None:
    root = Path(repo_root)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    commit_hash = result.stdout.strip()
    if result.returncode != 0 or not commit_hash:
        return None
    return commit_hash


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    return out_path


def write_run_config(
    paths: RunPaths,
    *,
    run_name: str,
    run_type: str,
    timestamp: str,
    input_dir: str | Path,
    repo_root: str | Path,
    config: Mapping[str, Any],
) -> Path:
    payload = {
        "run_name": str(run_name),
        "timestamp": str(timestamp),
        "run_type": str(run_type),
        "input_dir": str(Path(input_dir)),
        "run_dir": str(paths.run_dir),
        "git_commit_hash": try_git_commit_hash(repo_root),
        **dict(config),
    }
    return write_json(paths.config_path, payload)


def save_plan_snapshot(plan: pd.DataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(path, index=False)
    return path


def build_registry_row(
    *,
    timestamp: str,
    run_name: str,
    run_type: str,
    path_to_run_dir: str | Path,
    model_name: str | None,
    parameters: Mapping[str, Any] | str | None,
    horizon: int | None,
    k: int | None,
    normalization: str | None,
    smoothing: str | None,
    mean_ic: float | None,
    mean_spread: float | None,
    total_return_net: float | None,
    sharpe_net: float | None,
    sortino_net: float | None,
    max_drawdown_net: float | None,
    n_constant_score_days: int | None,
    experiment_name: str | None = None,
) -> dict[str, object]:
    if parameters is None:
        parameters_value = ""
    elif isinstance(parameters, str):
        parameters_value = parameters
    else:
        parameters_value = json.dumps(dict(parameters), sort_keys=True)

    return {
        "timestamp": str(timestamp),
        "run_name": str(run_name),
        "run_type": str(run_type),
        "experiment_name": "" if experiment_name is None else str(experiment_name),
        "model_name": "" if model_name is None else str(model_name),
        "parameters": parameters_value,
        "horizon": horizon,
        "k": k,
        "normalization": normalization,
        "smoothing": smoothing,
        "mean_ic": mean_ic,
        "mean_spread": mean_spread,
        "total_return_net": total_return_net,
        "sharpe_net": sharpe_net,
        "sortino_net": sortino_net,
        "max_drawdown_net": max_drawdown_net,
        "n_constant_score_days": n_constant_score_days,
        "path_to_run_dir": str(Path(path_to_run_dir)),
    }


def append_run_registry(registry_path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([dict(row) for row in rows])
    if new_df.empty:
        return path

    if path.exists():
        existing_df = pd.read_csv(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    ordered_cols = [col for col in REGISTRY_COLUMNS if col in combined.columns]
    extra_cols = [col for col in combined.columns if col not in ordered_cols]
    combined = combined[ordered_cols + extra_cols]
    combined.to_csv(path, index=False)
    return path


def _load_registry(registry_path: str | Path) -> pd.DataFrame:
    path = Path(registry_path)
    if not path.exists():
        return pd.DataFrame(columns=list(REGISTRY_COLUMNS))
    return pd.read_csv(path)


def update_champion_files(
    registry_path: str | Path,
    champions_dir: str | Path,
) -> list[Path]:
    registry = _load_registry(registry_path)
    if registry.empty:
        return []

    eligible = registry[registry["run_type"].isin(["single_run", "strategy_ablation"])].copy()
    if eligible.empty:
        return []

    eligible["timestamp_dt"] = pd.to_datetime(eligible["timestamp"], errors="coerce", utc=True)
    champions_root = Path(champions_dir)
    champions_root.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    best_by_metric: dict[str, dict[str, Any]] = {}

    for filename, metric_col in CHAMPION_METRICS:
        if metric_col in eligible.columns:
            metric_df = eligible.copy()
            metric_df[metric_col] = pd.to_numeric(metric_df[metric_col], errors="coerce")
            metric_df = metric_df.dropna(subset=[metric_col])
        else:
            metric_df = pd.DataFrame()

        if metric_df.empty:
            best_row = {
                "champion_metric": metric_col,
                "champion_value": None,
                "updated_timestamp": current_utc_timestamp(),
                "note": f"no eligible rows with finite {metric_col}",
            }
        else:
            metric_df = metric_df.sort_values(
                [metric_col, "timestamp_dt"],
                ascending=[False, False],
                kind="mergesort",
            ).reset_index(drop=True)
            best_row = metric_df.iloc[0].to_dict()
            best_row.pop("timestamp_dt", None)
            best_row["champion_metric"] = metric_col
            best_row["champion_value"] = best_row.get(metric_col)
            best_row["updated_timestamp"] = current_utc_timestamp()

        out_path = champions_root / filename
        write_json(out_path, best_row)
        written_paths.append(out_path)
        best_by_metric[metric_col] = best_row

    if "sharpe_net" in best_by_metric:
        current_path = champions_root / "current_etf_xs_champion.json"
        write_json(current_path, best_by_metric["sharpe_net"])
        written_paths.append(current_path)

    return written_paths
