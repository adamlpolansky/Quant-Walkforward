from __future__ import annotations

import argparse
from pathlib import Path

from qwf.data import load_price_panel_from_directory
from qwf.splits import make_global_walkforward_plan_from_dates

ROOT = Path(__file__).resolve().parents[1]


def _parse_tickers(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tickers = [item.strip() for item in value.split(",") if item.strip()]
    return tickers or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a global walk-forward split plan from a shared calendar.")
    p.add_argument("--train-months", type=int, default=9)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--start-date", type=str, default="2018-01-01")
    p.add_argument("--date-col", type=str, default="Date")
    p.add_argument("--tickers", type=str, default=None, help="Optional comma-separated subset, for example SPY,QQQ,IWM")
    p.add_argument("--recursive", action="store_true", help="Search for CSV files recursively")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "scripts" / "data",
        help="Directory with input CSV files",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output plan CSV path (default: scripts/splits_train_test/walkforward_global_plan_<train>_<test>.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = (
            ROOT / "scripts" / "splits_train_test" /
            f"walkforward_global_plan_{args.train_months}_{args.test_months}.csv"
        )

    panel = load_price_panel_from_directory(
        args.input_dir,
        date_col=args.date_col,
        tickers=_parse_tickers(args.tickers),
        recursive=args.recursive,
        intersect_dates=True,
    )

    calendar = panel["date"].drop_duplicates().sort_values()
    plan = make_global_walkforward_plan_from_dates(
        calendar,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        start_date=args.start_date,
        output_csv=output_csv,
    )

    print(f"Saved global split plan: {output_csv}")
    print(f"Rows: {len(plan)}")


if __name__ == "__main__":
    main()
