from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np


def load_ohlcv(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
        threads=False,
    )


    if df is None or df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # ensure "naive" DatetimeIndex if tz-aware
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="first")].sort_index()

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted increasing after cleaning")

    # Standardize column names

    # Flatten MultiIndex columns from yfinance (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # pokud je tam jen jeden ticker (u tebe SPY), nech jen "Price" level
        if df.columns.get_level_values("Ticker").nunique() == 1:
            df.columns = df.columns.get_level_values("Price")
        else:
            # kdybys někdy stahoval více tickerů najednou
            df.columns = ["_".join(map(str, col)) for col in df.columns.to_list()]

    # Standardize column names (now safe)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in df.columns]
    return df[keep]


def add_returns(df: pd.DataFrame, price_col: str | None = None) -> pd.DataFrame:
    out = df.copy()

    if price_col is None:
        # If auto_adjust=False, you might prefer Adj_Close when available
        price_col = "Adj_Close" if "Adj_Close" in out.columns else "Close"

    if price_col not in out.columns:
        raise ValueError(f"{price_col} column is required to compute returns")

    close = pd.to_numeric(out[price_col], errors="coerce")

    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close).diff()

    return out
