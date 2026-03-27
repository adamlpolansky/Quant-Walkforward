from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qwf import metrics
from qwf.backtest import ZScoreMRConfig, run_wf_backtest_ret
from qwf.data import ensure_returns, load_price_csv

ROOT = Path(__file__).resolve().parents[1]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    default_data = ROOT / "scripts" / "data" / "SPY.csv"
    default_plan = ROOT / "scripts" / "splits_train_test" / "walkforward_plan_9_1.csv"
    default_out = ROOT / "outputs" / "experiments"

    p = argparse.ArgumentParser(description="Param sweep (mini experiment) for the week-1 WF backtest.")
    p.add_argument("--data", type=Path, default=default_data, help="CSV with week-1 OHLCV columns")
    p.add_argument("--plan", type=Path, default=default_plan, help="Walk-forward plan CSV")
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--date-col", type=str, default="Date")
    p.add_argument("--price-col", type=str, default="Close")
    p.add_argument("--ret-col", type=str, default="ret")
    p.add_argument("--source-file", type=str, default=None, help="Filter plan by source_file (for example SPY.csv)")

    p.add_argument("--K", type=float, default=1.0)
    p.add_argument("--ddof", type=int, default=0)
    p.add_argument(
        "--cost-bps-per-turnover",
        type=float,
        default=0.0,
        help="Transaction cost in bps per 1.0 turnover (abs(pos-pos.shift(1))). Default 0.",
    )

    p.add_argument("--n-grid", type=str, default="10,20,40", help="Comma-separated, for example 10,20,40")
    p.add_argument(
        "--step-frac-grid",
        type=str,
        default="0.25,0.5",
        help="Comma-separated, for example 0.25,0.5",
    )
    p.add_argument("--results-name", type=str, default="results_grid", help="Base name for result files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_price_csv(args.data, date_col=args.date_col)
    df = ensure_returns(df, price_col=args.price_col, ret_col=args.ret_col)

    plan = pd.read_csv(
        args.plan,
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    )

    source_file = args.source_file
    if source_file is None and "source_file" in plan.columns:
        source_file = args.data.name

    n_grid = _parse_int_list(args.n_grid)
    step_grid = _parse_float_list(args.step_frac_grid)
    rows: list[dict[str, float | int]] = []

    for n in n_grid:
        for step_frac in step_grid:
            cfg = ZScoreMRConfig(n=n, K=args.K, step_frac=step_frac, ddof=args.ddof)

            test_detail = run_wf_backtest_ret(
                df=df,
                plan=plan,
                price_col=args.price_col,
                ret_col=args.ret_col,
                cfg=cfg,
                source_file=source_file,
                cost_bps_per_turnover=args.cost_bps_per_turnover,
            )

            fold_summary = metrics.fold_summary(test_detail)
            stitched = metrics.stitched_curve(test_detail, rolling_sharpe_window=None)
            stats = metrics.perf_stats_from_pnl(stitched["pnl"])
            mean_fold_sharpe = float(pd.to_numeric(fold_summary["sharpe"], errors="coerce").mean())

            rows.append(
                {
                    "n": n,
                    "step_frac": step_frac,
                    "K": float(args.K),
                    "ddof": int(args.ddof),
                    "cost_bps_per_turnover": float(args.cost_bps_per_turnover),
                    "mean_fold_sharpe": mean_fold_sharpe,
                    **stats,
                }
            )

    res = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.results_name}.csv"
    md_path = args.out_dir / f"{args.results_name}_best.md"
    res.to_csv(csv_path, index=False)

    best_by_mean_sharpe = res.sort_values("mean_fold_sharpe", ascending=False).head(1)
    best_by_total_return = res.sort_values("total_return", ascending=False).head(1)

    md_lines = [
        f"# Param sweep results ({args.results_name})",
        "",
        f"Saved CSV: `{csv_path}`",
        "",
        "## Best config by mean fold Sharpe",
        _md_table(best_by_mean_sharpe),
        "",
        "## Best config by stitched total return",
        _md_table(best_by_total_return),
        "",
        "## Full grid (sorted by mean fold Sharpe)",
        _md_table(res.sort_values("mean_fold_sharpe", ascending=False)),
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved:\n- {csv_path}\n- {md_path}")


def _md_table(df_: pd.DataFrame) -> str:
    cols = [
        "n",
        "step_frac",
        "K",
        "cost_bps_per_turnover",
        "mean_fold_sharpe",
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
    ]
    df2 = df_[cols].copy()

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]

    def _fmt(v: object) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{float(v):.6g}"
        return str(v)

    for _, row in df2.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in cols) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
