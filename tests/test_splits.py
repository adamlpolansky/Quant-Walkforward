from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qwf.splits import make_walkforward_plan_for_directory


def _write_synthetic_price_csv(path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    close = 100.0 + np.arange(len(idx), dtype=float)

    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000,
        }
    )

    # Write a deliberately messy file so the planner exercises sorted unique date loading.
    messy = pd.concat([df.iloc[40:60], df.iloc[[0]], df.iloc[:40], df.iloc[60:]], ignore_index=True)
    messy.to_csv(path, index=False)


def _make_plan(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    csv_path = input_dir / "SYN.csv"
    output_csv = tmp_path / "walkforward_plan.csv"

    _write_synthetic_price_csv(csv_path)

    plan = make_walkforward_plan_for_directory(
        input_dir=input_dir,
        output_csv=output_csv,
        train_months=2,
        test_months=1,
        step_months=1,
        start_date="2020-01-01",
        date_col="Date",
    )
    return plan, output_csv


def test_make_walkforward_plan_is_stable_on_synthetic_calendar(tmp_path: Path) -> None:
    plan, _ = _make_plan(tmp_path)

    expected = pd.DataFrame(
        [
            {
                "source_file": "SYN.csv",
                "fold_id": 0,
                "train_start": "2020-01-01",
                "train_end": "2020-02-28",
                "test_start": "2020-03-02",
                "test_end": "2020-03-31",
            },
            {
                "source_file": "SYN.csv",
                "fold_id": 1,
                "train_start": "2020-02-03",
                "train_end": "2020-03-31",
                "test_start": "2020-04-01",
                "test_end": "2020-04-30",
            },
            {
                "source_file": "SYN.csv",
                "fold_id": 2,
                "train_start": "2020-03-02",
                "train_end": "2020-04-30",
                "test_start": "2020-05-01",
                "test_end": "2020-05-29",
            },
            {
                "source_file": "SYN.csv",
                "fold_id": 3,
                "train_start": "2020-04-01",
                "train_end": "2020-05-29",
                "test_start": "2020-06-01",
                "test_end": "2020-06-30",
            },
        ]
    )

    assert_frame_equal(plan.reset_index(drop=True), expected)


def test_make_walkforward_plan_respects_basic_fold_semantics(tmp_path: Path) -> None:
    plan, output_csv = _make_plan(tmp_path)

    assert output_csv.exists()
    assert not plan.empty

    reloaded = pd.read_csv(output_csv)
    assert_frame_equal(plan, reloaded)

    date_cols = ["train_start", "train_end", "test_start", "test_end"]
    plan_dates = plan.copy()
    for col in date_cols:
        plan_dates[col] = pd.to_datetime(plan_dates[col])

    assert plan_dates["fold_id"].is_monotonic_increasing
    assert (plan_dates["train_start"] <= plan_dates["train_end"]).all()
    assert (plan_dates["test_start"] <= plan_dates["test_end"]).all()
    assert (plan_dates["train_end"] < plan_dates["test_start"]).all()

    train_lengths = (plan_dates["train_end"] - plan_dates["train_start"]).dt.days
    test_lengths = (plan_dates["test_end"] - plan_dates["test_start"]).dt.days
    assert (train_lengths >= 0).all()
    assert (test_lengths >= 0).all()

    assert plan_dates["train_start"].is_monotonic_increasing
    assert plan_dates["test_start"].is_monotonic_increasing
