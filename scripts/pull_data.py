from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from qwf.data import add_returns, load_ohlcv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "scripts" / "data"
DEFAULT_REAL_ETF_DIR = DEFAULT_DATA_DIR / "real_etf_xs"
DEFAULT_START_DATE = "2018-01-01"
LEGACY_DEFAULT_TICKER = "SPY"
DEFAULT_ETF_UNIVERSE: tuple[str, ...] = (
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLB",
    "XLRE",
    "XLK",
    "XLU",
    "SPY",
    "QQQ",
)


@dataclass(frozen=True)
class DownloadResult:
    ticker: str
    out_path: Path
    rows: int
    date_start: str
    date_end: str


def _parse_ticker_tokens(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []

    tickers: list[str] = []
    for raw_value in values:
        for item in str(raw_value).split(","):
            ticker = item.strip().upper()
            if ticker:
                tickers.append(ticker)
    return tickers


def resolve_tickers(
    *,
    ticker_args: Sequence[str] | None = None,
    tickers_csv: str | None = None,
    use_default_etf_universe: bool = False,
) -> list[str]:
    requested: list[str] = []

    if use_default_etf_universe:
        requested.extend(DEFAULT_ETF_UNIVERSE)

    requested.extend(_parse_ticker_tokens(ticker_args))
    requested.extend(_parse_ticker_tokens([tickers_csv] if tickers_csv else None))

    if not requested:
        return [LEGACY_DEFAULT_TICKER]

    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in requested:
        if ticker in seen:
            continue
        seen.add(ticker)
        deduped.append(ticker)
    return deduped


def resolve_output_dir(
    output_dir: Path | None,
    *,
    tickers: Sequence[str],
    use_default_etf_universe: bool,
) -> Path:
    if output_dir is not None:
        return output_dir

    if use_default_etf_universe or len(tickers) > 1:
        return DEFAULT_REAL_ETF_DIR

    return DEFAULT_DATA_DIR


def save_price_csv(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index_label="Date")


def download_one_ticker(
    ticker: str,
    *,
    start_date: str,
    end_date: str | None,
    output_dir: Path,
) -> DownloadResult:
    df = load_ohlcv(ticker, start=start_date, end=end_date)
    df = add_returns(df)

    out_path = output_dir / f"{ticker}.csv"
    save_price_csv(df, out_path)

    return DownloadResult(
        ticker=ticker,
        out_path=out_path,
        rows=int(len(df)),
        date_start=str(df.index.min().date()),
        date_end=str(df.index.max().date()),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download daily OHLCV data from yfinance into one CSV per ticker.")
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        help="Ticker to download. Repeat the flag for multiple tickers. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Optional comma-separated ticker list, for example XLC,XLY,SPY,QQQ",
    )
    parser.add_argument(
        "--use-default-etf-universe",
        action="store_true",
        help="Download XLC,XLY,XLP,XLE,XLF,XLV,XLI,XLB,XLRE,XLK,XLU,SPY,QQQ.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-ticker CSV files. Defaults to scripts/data/real_etf_xs for multi-ticker ETF runs and scripts/data for legacy SPY-only usage.",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Inclusive download start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Optional exclusive end date (YYYY-MM-DD).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tickers = resolve_tickers(
        ticker_args=args.ticker,
        tickers_csv=args.tickers,
        use_default_etf_universe=args.use_default_etf_universe,
    )
    output_dir = resolve_output_dir(
        args.output_dir,
        tickers=tickers,
        use_default_etf_universe=args.use_default_etf_universe,
    )

    results: list[DownloadResult] = []
    failures: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            result = download_one_ticker(
                ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=output_dir,
            )
            results.append(result)
            print(
                f"[ok] {result.ticker}: {result.rows} rows "
                f"({result.date_start} to {result.date_end}) -> {result.out_path}"
            )
        except Exception as exc:
            failures.append((ticker, str(exc)))
            print(f"[failed] {ticker}: {exc}")

    if not results:
        raise SystemExit("No ticker data downloaded successfully.")

    overall_start = min(result.date_start for result in results)
    overall_end = max(result.date_end for result in results)
    print(
        f"Downloaded {len(results)}/{len(tickers)} tickers into {output_dir} "
        f"with overall date range {overall_start} to {overall_end}."
    )

    if failures:
        failed_names = ", ".join(ticker for ticker, _ in failures)
        print(f"Failed tickers: {failed_names}")


if __name__ == "__main__":
    main()
