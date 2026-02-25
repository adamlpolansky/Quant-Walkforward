from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qwf.backtest import ZScoreMRConfig, run_wf_backtest_ret
from qwf import metrics
from qwf.reporting.plots import save_report_plots

ROOT = Path(__file__).resolve().parents[1]  # ✅ FIX: define ROOT


def _load_price_csv(path: Path, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    return df


def _ensure_returns(df: pd.DataFrame, price_col: str, ret_col: str) -> pd.DataFrame:
    out = df.copy()

    if ret_col not in out.columns:
        if price_col not in out.columns:
            raise ValueError(f"Missing price_col='{price_col}' in data columns: {list(out.columns)}")
        close = pd.to_numeric(out[price_col], errors="coerce").astype(float)
        out[ret_col] = close.pct_change()

    if "log_ret" not in out.columns and price_col in out.columns:
        close = pd.to_numeric(out[price_col], errors="coerce").astype(float)
        out["log_ret"] = np.log(close).diff()

    return out


def parse_args() -> argparse.Namespace:
    default_data = ROOT / "scripts" / "data" / "SPY.csv"
    default_plan = ROOT / "scripts" / "splits_train_test" / "walkforward_plan_9_1.csv"
    default_out = ROOT / "outputs"

    p = argparse.ArgumentParser(description="Run walk-forward backtest (I/O only).")
    p.add_argument("--data", type=Path, default=default_data, help="CSV with OHLCV (default: scripts/data/SPY.csv)")
    p.add_argument("--plan", type=Path, default=default_plan, help="WF plan CSV (default: scripts/splits_train_test/...)")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Output directory (default: outputs/)")
    p.add_argument("--run-name", type=str, default="wf_9m_1m_zscore_v1", help="Prefix for output files")

    p.add_argument("--date-col", type=str, default="Date")
    p.add_argument("--price-col", type=str, default="Close")
    p.add_argument("--ret-col", type=str, default="ret")

    p.add_argument("--source-file", type=str, default=None, help="Filter plan by source_file (e.g., SPY.csv)")

    p.add_argument("--n", type=int, default=20)
    p.add_argument("--K", type=float, default=1.0)
    p.add_argument("--step-frac", type=float, default=0.25)
    p.add_argument("--ddof", type=int, default=0)

    p.add_argument("--save-per-fold", action="store_true", help="Save per-fold CSVs into out_dir/folds/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_price_csv(args.data, date_col=args.date_col)
    df = _ensure_returns(df, price_col=args.price_col, ret_col=args.ret_col)

    plan = pd.read_csv(
        args.plan,
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    )

    source_file = args.source_file
    if source_file is None and "source_file" in plan.columns:
        source_file = args.data.name

    cfg = ZScoreMRConfig(
        n=args.n,
        K=args.K,
        step_frac=args.step_frac,
        ddof=args.ddof,
    )

    test_detail = run_wf_backtest_ret(
        df=df,
        plan=plan,
        price_col=args.price_col,
        ret_col=args.ret_col,
        cfg=cfg,
        source_file=source_file,
    )

    fold_summary = metrics.fold_summary(test_detail)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_path = args.out_dir / f"{args.run_name}_test_detail.csv"
    sum_path = args.out_dir / f"{args.run_name}_fold_summary.csv"
    cfg_path = args.out_dir / f"{args.run_name}_config.json"

    test_detail.to_csv(test_path, index=True)
    fold_summary.to_csv(sum_path, index=False)

    if args.save_per_fold:
        folds_dir = args.out_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        for fold_id, g in test_detail.groupby("fold_id", sort=True):
            g.to_csv(folds_dir / f"fold_{fold_id}_test.csv", index=True)

    meta = {
        "run_name": args.run_name,
        "data_path": str(args.data),
        "plan_path": str(args.plan),
        "date_col": args.date_col,
        "price_col": args.price_col,
        "ret_col": args.ret_col,
        "source_file": source_file,
        "cfg": {
            "n": cfg.n,
            "K": cfg.K,
            "step_frac": cfg.step_frac,
            "ddof": cfg.ddof,
            "min_periods": cfg.min_periods,
        },
    }
    cfg_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"✅ Saved:\n- {test_path}\n- {sum_path}\n- {cfg_path}")
    plots_dir = args.out_dir / "plots" / args.run_name
    save_report_plots(
        test_detail=test_detail,
        fold_summary=fold_summary,
        out_dir=plots_dir,
        run_name=args.run_name,
        rolling_sharpe_window=63,
        save_per_fold_equity=True,  # když dáš True, uloží i per-fold pngčka
    )
    print(f"📈 Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
