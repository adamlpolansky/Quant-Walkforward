from __future__ import annotations

import argparse
from pathlib import Path
from qwf.splits import make_walkforward_plan_for_directory

ROOT = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate walk-forward split plan CSV.")
    p.add_argument("--train-months", type=int, default=9)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--start-date", type=str, default="2018-01-01")
    p.add_argument("--date-col", type=str, default="Date")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "scripts" / "data",
        help="Directory with input CSV files"
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output plan CSV path (default: scripts/splits_train_test/walkforward_plan_<train>_<test>.csv)"
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = (
            ROOT / "scripts" / "splits_train_test" /
            f"walkforward_plan_{args.train_months}_{args.test_months}.csv"
        )

    plan = make_walkforward_plan_for_directory(
        input_dir=args.input_dir,
        output_csv=output_csv,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        start_date=args.start_date,
        date_col=args.date_col,
    )

    print(f"✅ Saved split plan: {output_csv}")
    print(f"Rows: {len(plan)}")

if __name__ == "__main__":
    main()