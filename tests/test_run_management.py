from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qwf.run_management import (
    append_run_registry,
    build_registry_row,
    init_run_paths,
    update_champion_files,
    write_run_config,
)


def test_init_run_paths_creates_expected_directories(tmp_path: Path) -> None:
    paths = init_run_paths(tmp_path / "outputs", "demo_run")

    assert paths.output_root == tmp_path / "outputs"
    assert paths.run_dir == tmp_path / "outputs" / "runs" / "demo_run"
    assert paths.plots_dir.exists()
    assert paths.champions_dir.exists()


def test_registry_append_and_champion_update_work(tmp_path: Path) -> None:
    paths = init_run_paths(tmp_path / "outputs", "demo_run")
    config_path = write_run_config(
        paths,
        run_name="demo_run",
        run_type="single_run",
        timestamp="2026-03-28T12:00:00+00:00",
        input_dir=tmp_path / "data",
        repo_root=tmp_path,
        config={"model_name": "elasticnet", "model_params": {"alpha": 0.001, "l1_ratio": 0.5}},
    )

    rows = [
        build_registry_row(
            timestamp="2026-03-28T12:00:00+00:00",
            run_name="demo_run_a",
            run_type="single_run",
            path_to_run_dir=tmp_path / "outputs" / "runs" / "demo_run_a",
            model_name="ridge",
            parameters={"alpha": 1.0},
            horizon=1,
            k=2,
            normalization="none",
            smoothing="none",
            mean_ic=0.01,
            mean_spread=0.001,
            total_return_net=0.02,
            sharpe_net=0.4,
            sortino_net=0.5,
            max_drawdown_net=-0.1,
            n_constant_score_days=0,
        ),
        build_registry_row(
            timestamp="2026-03-28T13:00:00+00:00",
            run_name="demo_run_b",
            run_type="strategy_ablation",
            experiment_name="h3__k3__norm_none__smooth_none",
            path_to_run_dir=tmp_path / "outputs" / "runs" / "demo_run_b",
            model_name="elasticnet",
            parameters={"alpha": 0.001, "l1_ratio": 0.5},
            horizon=3,
            k=3,
            normalization="none",
            smoothing="none",
            mean_ic=0.02,
            mean_spread=0.003,
            total_return_net=0.05,
            sharpe_net=0.9,
            sortino_net=1.1,
            max_drawdown_net=-0.08,
            n_constant_score_days=1,
        ),
    ]

    registry_path = append_run_registry(paths.registry_path, rows)
    champions = update_champion_files(paths.registry_path, paths.champions_dir)
    registry = pd.read_csv(registry_path)
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    champion_payload = json.loads((paths.champions_dir / "current_etf_xs_champion.json").read_text(encoding="utf-8"))

    assert config_path.exists()
    assert config_payload["run_name"] == "demo_run"
    assert config_payload["run_type"] == "single_run"
    assert config_payload["model_name"] == "elasticnet"
    assert registry_path.exists()
    assert len(registry) == 2
    assert (registry["run_name"] == "demo_run_b").any()
    assert champions
    assert champion_payload["run_name"] == "demo_run_b"
    assert champion_payload["champion_metric"] == "sharpe_net"
