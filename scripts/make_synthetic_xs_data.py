from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_synthetic_xs_data(
    output_dir: str | Path,
    *,
    num_tickers: int = 8,
    years: int = 3,
    start_date: str = "2020-01-01",
    seed: int = 7,
    ticker_prefix: str = "DEMO",
) -> list[Path]:
    if num_tickers < 2:
        raise ValueError("num_tickers must be >= 2")
    if years < 1:
        raise ValueError("years must be >= 1")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=252 * years)
    common_factor = rng.normal(0.0002, 0.0075, size=len(dates))

    saved_paths: list[Path] = []

    for idx in range(num_tickers):
        ticker = f"{ticker_prefix}{idx + 1:02d}"
        drift = 0.00005 + 0.00003 * idx
        idio_scale = 0.006 + 0.0005 * idx
        idio = rng.normal(0.0, idio_scale, size=len(dates))
        returns = drift + 0.35 * common_factor + idio

        close = 100.0 * np.exp(np.cumsum(returns))
        prev_close = np.r_[close[0], close[:-1]]
        open_gap = rng.normal(0.0, 0.0025, size=len(dates))
        open_ = prev_close * (1.0 + open_gap)

        intraday_range = np.abs(rng.normal(0.005 + 0.0004 * idx, 0.0015, size=len(dates)))
        high = np.maximum(open_, close) * (1.0 + intraday_range)
        low = np.minimum(open_, close) * (1.0 - intraday_range)

        vol_base = 900_000 + idx * 120_000
        vol_regime = 1.0 + 0.15 * np.sin(np.arange(len(dates)) / 21.0 + idx)
        volume_noise = np.exp(rng.normal(0.0, 0.18, size=len(dates)))
        volume = np.maximum(vol_base * vol_regime * volume_noise, 25_000).round().astype(int)

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

        out_path = out_dir / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        saved_paths.append(out_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a deterministic synthetic daily cross-sectional demo universe.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "scripts" / "data" / "demo_xs")
    p.add_argument("--num-tickers", type=int, default=8)
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--start-date", type=str, default="2020-01-01")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ticker-prefix", type=str, default="DEMO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = make_synthetic_xs_data(
        args.output_dir,
        num_tickers=args.num_tickers,
        years=args.years,
        start_date=args.start_date,
        seed=args.seed,
        ticker_prefix=args.ticker_prefix,
    )

    print(f"Saved synthetic demo universe to: {args.output_dir}")
    print(f"Tickers: {len(paths)}")
    if paths:
        print(f"First file: {paths[0]}")


if __name__ == "__main__":
    main()
