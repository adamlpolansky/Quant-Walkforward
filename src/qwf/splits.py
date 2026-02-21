from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    return ts.replace(day=1)


def _calendar_end_of_month_window(start_in_month: pd.Timestamp, n_months: int) -> pd.Timestamp:
    """
    Given any date inside a month, return the calendar (inclusive) end date of a window
    spanning n_months starting from that month.
    Example: start_in_month=2018-01-02, n_months=1 -> 2018-01-31
             start_in_month=2018-01-02, n_months=9 -> 2018-09-30
    """
    if n_months <= 0:
        raise ValueError("n_months must be >= 1")
    ms = _month_start(start_in_month)
    return (ms + pd.DateOffset(months=n_months)) - pd.Timedelta(days=1)


def _first_date_on_or_after(dates: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[pd.Timestamp]:
    # dates must be sorted
    pos = dates.searchsorted(t, side="left")
    if pos >= len(dates):
        return None
    return pd.Timestamp(dates[pos])


def _last_date_on_or_before(dates: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[pd.Timestamp]:
    # dates must be sorted
    pos = dates.searchsorted(t, side="right") - 1
    if pos < 0:
        return None
    return pd.Timestamp(dates[pos])


def _load_sorted_unique_dates_from_csv(fp: Path, date_col: str) -> pd.DatetimeIndex:
    """
    Load only date_col from CSV, parse to datetime, return sorted unique DatetimeIndex (tz-naive).
    """
    try:
        s = pd.read_csv(fp, usecols=[date_col])[date_col]
    except ValueError:
        # usecols mismatch -> show available columns
        cols = list(pd.read_csv(fp, nrows=0).columns)
        raise ValueError(f"{fp.name}: missing date_col='{date_col}'. Columns: {cols}")

    dates = pd.to_datetime(s, errors="coerce").dropna()
    if dates.empty:
        return pd.DatetimeIndex([])

    idx = pd.DatetimeIndex(dates.unique()).sort_values()

    # strip timezone if present
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    return idx


def make_walkforward_plan_for_directory(
    input_dir: str | Path,
    output_csv: str | Path,
    train_months: int,
    test_months: int,
    step_months: int = 1,
    start_date: str | pd.Timestamp = "2018-01-01",
    date_col: str = "Date",
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Build a walk-forward split plan for ALL *.csv files in input_dir.

    Plan semantics:
    - Windows are defined in CALENDAR months, but the resulting *end* dates are the last available
      trading day in the data <= the calendar end boundary.
    - 'end' is INCLUSIVE.
    - Fold 0 train starts at 'start_date' (not necessarily month-start).
    - For each fold, the "month counting" begins in the month containing train_start_cal.
    - Next fold moves by step_months (calendar months) from the fold's train month start.

    Output columns:
      source_file, fold_id,
      train_start, train_end, test_start, test_end,
      train_months, test_months, step_months, start_date
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    if train_months < 1 or test_months < 1 or step_months < 1:
        raise ValueError("train_months, test_months, step_months must be >= 1")

    start_ts = pd.Timestamp(start_date).normalize()

    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir.resolve()} (pattern={pattern})")

    rows: List[Dict[str, Any]] = []

    for fp in files:
        dates = _load_sorted_unique_dates_from_csv(fp, date_col=date_col)
        if dates.empty:
            continue

        max_date = pd.Timestamp(dates[-1]).normalize()

        # fold 0 train start = start_date (can be inside a month)
        fold0_train_start_cal = start_ts
        anchor_month_start = _month_start(fold0_train_start_cal)

        fold_id = 0
        k = 0

        while True:
            fold_train_month_start = anchor_month_start + pd.DateOffset(months=k * step_months)

            train_start_cal = fold0_train_start_cal if k == 0 else fold_train_month_start

            # if even train start calendar is beyond data, we're done for this file
            if train_start_cal > max_date:
                break

            train_end_cal = _calendar_end_of_month_window(train_start_cal, train_months)

            test_start_cal = _month_start(train_start_cal) + pd.DateOffset(months=train_months)
            test_end_cal = _calendar_end_of_month_window(test_start_cal, test_months)

            # since k increases monotonically, once test_start_cal is beyond data, we can stop
            if test_start_cal > max_date:
                break

            train_start = _first_date_on_or_after(dates, train_start_cal)
            train_end = _last_date_on_or_before(dates, train_end_cal)
            test_start = _first_date_on_or_after(dates, test_start_cal)
            test_end = _last_date_on_or_before(dates, test_end_cal)

            # If we can't form valid non-empty windows, skip fold but keep trying next k.
            if (
                train_start is None or train_end is None or
                test_start is None or test_end is None or
                train_start > train_end or
                test_start > test_end
            ):
                k += 1
                continue

            rows.append({
                # safer for recursive=True (no filename collisions)
                "source_file": fp.relative_to(input_dir).as_posix(),
                "fold_id": fold_id,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
            })

            fold_id += 1
            k += 1

    plan = pd.DataFrame(rows)
    if plan.empty:
        raise RuntimeError("No valid folds produced. Check start_date / date_col / file contents.")

    plan = plan.sort_values(["source_file", "fold_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_csv, index=False)
    return plan