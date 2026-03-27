from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_DATE_COL = "Date"
DEFAULT_PANEL_DATE_COL = "date"
DEFAULT_TICKER_COL = "ticker"
WEEK1_REQUIRED_PRICE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
OPTIONAL_PRICE_COLUMNS: tuple[str, ...] = ("Adj_Close", "ret", "log_ret")
WEEK2_REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    DEFAULT_PANEL_DATE_COL,
    DEFAULT_TICKER_COL,
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)


def _format_columns(columns: Sequence[str]) -> str:
    return ", ".join(columns)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    source_name: str = "data",
    date_col: str = DEFAULT_DATE_COL,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if not missing:
        return

    expected = _format_columns(required_columns)
    optional = _format_columns(OPTIONAL_PRICE_COLUMNS)
    available = _format_columns([str(col) for col in df.columns])
    raise ValueError(
        f"{source_name}: missing required columns {missing}. "
        f"Week-1 CSV contract expects [{expected}] plus a '{date_col}' column. "
        f"Optional derived columns are [{optional}]. "
        f"Available columns: [{available}]"
    )


def _coerce_datetime_index(
    values: pd.Series | pd.Index | Iterable[object],
    *,
    source_name: str,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    invalid_count = int(dt_index.isna().sum())
    if invalid_count:
        raise ValueError(f"{source_name}: found {invalid_count} invalid values in '{date_col}'")

    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_localize(None)

    return dt_index


def set_sorted_date_index(
    df: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    source_name: str = "data",
) -> pd.DataFrame:
    if date_col not in df.columns:
        available = _format_columns([str(col) for col in df.columns])
        raise ValueError(
            f"{source_name}: missing date column '{date_col}'. Available columns: [{available}]"
        )

    out = df.copy()
    out.index = _coerce_datetime_index(out.pop(date_col), source_name=source_name, date_col=date_col)
    out = out[~out.index.duplicated(keep="first")].sort_index()

    if out.empty:
        raise ValueError(f"{source_name}: no rows remaining after date parsing and de-duplication")

    return out


def load_price_csv(
    path: str | Path,
    *,
    date_col: str = DEFAULT_DATE_COL,
    required_columns: Sequence[str] = WEEK1_REQUIRED_PRICE_COLUMNS,
) -> pd.DataFrame:
    """
    Load a local daily price CSV for the current week-1 workflow.

    Expected contract:
    - required columns: Open, High, Low, Close, Volume
    - optional derived columns: ret, log_ret
    - date column: Date
    """
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df = set_sorted_date_index(df, date_col=date_col, source_name=csv_path.name)
    validate_required_columns(df, required_columns, source_name=csv_path.name, date_col=date_col)
    return df


def intersect_calendar(calendars: Sequence[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    if not calendars:
        raise ValueError("At least one calendar is required")

    common = _coerce_datetime_index(calendars[0], source_name="calendar", date_col=DEFAULT_PANEL_DATE_COL)
    common = common.unique().sort_values()

    for calendar in calendars[1:]:
        idx = _coerce_datetime_index(calendar, source_name="calendar", date_col=DEFAULT_PANEL_DATE_COL)
        common = common.intersection(idx.unique().sort_values())

    if common.empty:
        raise ValueError("No shared dates across the selected universe")

    return pd.DatetimeIndex(common).sort_values()


def _select_price_files(
    input_dir: str | Path,
    *,
    tickers: Sequence[str] | None = None,
    recursive: bool = False,
) -> list[Path]:
    root = Path(input_dir)
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {root.resolve()} (pattern={pattern})")

    if tickers is None:
        return files

    requested = {str(ticker) for ticker in tickers}
    selected = [fp for fp in files if fp.stem in requested]
    missing = sorted(requested - {fp.stem for fp in selected})
    if missing:
        raise FileNotFoundError(
            f"Missing CSV files for tickers {missing} in: {root.resolve()}"
        )

    return selected


def load_price_panel_from_directory(
    input_dir: str | Path,
    *,
    date_col: str = DEFAULT_DATE_COL,
    tickers: Sequence[str] | None = None,
    recursive: bool = False,
    intersect_dates: bool = True,
    required_columns: Sequence[str] = WEEK1_REQUIRED_PRICE_COLUMNS,
) -> pd.DataFrame:
    """
    Load multiple local price CSV files into a long daily panel.

    Assumptions:
    - one file per ticker
    - ticker is inferred from the file stem
    - by default, dates are intersected across the selected universe
    """
    files = _select_price_files(input_dir, tickers=tickers, recursive=recursive)

    frames: list[tuple[str, pd.DataFrame]] = []
    for fp in files:
        frames.append((fp.stem, load_price_csv(fp, date_col=date_col, required_columns=required_columns)))

    if not frames:
        raise RuntimeError("No valid price files were loaded")

    common_dates: pd.DatetimeIndex | None = None
    if intersect_dates:
        common_dates = intersect_calendar([df.index for _, df in frames])

    panel_frames: list[pd.DataFrame] = []
    for ticker, df in frames:
        aligned = df.loc[common_dates] if common_dates is not None else df.copy()
        out = aligned.copy()
        out[DEFAULT_PANEL_DATE_COL] = out.index
        out[DEFAULT_TICKER_COL] = ticker
        panel_frames.append(out.reset_index(drop=True))

    panel = pd.concat(panel_frames, ignore_index=True, sort=False)
    ordered_cols = [
        DEFAULT_PANEL_DATE_COL,
        DEFAULT_TICKER_COL,
        *WEEK1_REQUIRED_PRICE_COLUMNS,
    ]
    extra_cols = [col for col in panel.columns if col not in ordered_cols]
    panel = panel[ordered_cols + extra_cols]
    panel = panel.sort_values([DEFAULT_PANEL_DATE_COL, DEFAULT_TICKER_COL], kind="mergesort").reset_index(drop=True)

    validate_required_columns(
        panel,
        WEEK2_REQUIRED_PANEL_COLUMNS[2:],
        source_name="panel",
        date_col=DEFAULT_PANEL_DATE_COL,
    )
    return panel


def _resolve_price_col(df: pd.DataFrame, price_col: str | None) -> str:
    if price_col is None:
        price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"

    if price_col not in df.columns:
        available = _format_columns([str(col) for col in df.columns])
        raise ValueError(
            f"Missing price column '{price_col}'. Available columns: [{available}]"
        )

    return price_col


def _price_series(df: pd.DataFrame, price_col: str) -> pd.Series:
    return pd.to_numeric(df[price_col], errors="coerce").astype(float)


def add_returns(df: pd.DataFrame, price_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    resolved_price_col = _resolve_price_col(out, price_col)
    close = _price_series(out, resolved_price_col)

    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close).diff()
    return out


def ensure_returns(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    ret_col: str = "ret",
    log_ret_col: str = "log_ret",
) -> pd.DataFrame:
    out = df.copy()
    need_ret = ret_col not in out.columns
    need_log_ret = log_ret_col not in out.columns

    if not need_ret and not need_log_ret:
        return out

    resolved_price_col = _resolve_price_col(out, price_col)
    close = _price_series(out, resolved_price_col)

    if need_ret:
        out[ret_col] = close.pct_change()

    if need_log_ret:
        out[log_ret_col] = np.log(close).diff()

    return out


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
    df.index = _coerce_datetime_index(df.index, source_name=f"ticker={ticker}")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.get_level_values("Ticker").nunique() == 1:
            df.columns = df.columns.get_level_values("Price")
        else:
            df.columns = ["_".join(map(str, col)) for col in df.columns.to_list()]

    df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]

    keep = [col for col in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] if col in df.columns]
    df = df[keep]
    validate_required_columns(df, WEEK1_REQUIRED_PRICE_COLUMNS, source_name=f"ticker={ticker}")
    return df
