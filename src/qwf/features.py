from __future__ import annotations

import pandas as pd

REQUIRED_FEATURE_INPUT_COLUMNS: tuple[str, ...] = ("date", "ticker", "Close", "Volume")
DAILY_FEATURE_COLUMNS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "vol_5d",
    "vol_20d",
    "dist_ma_5",
    "dist_ma_20",
    "vol_ratio_20",
]


def _validate_panel_columns(panel: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [col for col in required_columns if col not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")


def make_daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple backward-looking daily features per ticker on a long panel.

    The returned panel is sorted by date, ticker for deterministic downstream use.
    """
    _validate_panel_columns(panel, REQUIRED_FEATURE_INPUT_COLUMNS)

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    grouped = out.groupby("ticker", sort=False)
    close = grouped["Close"]
    volume = grouped["Volume"]

    out["ret_1d"] = close.pct_change()
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)

    daily_ret = out.groupby("ticker", sort=False)["ret_1d"]
    out["vol_5d"] = daily_ret.transform(lambda s: s.rolling(5, min_periods=5).std())
    out["vol_20d"] = daily_ret.transform(lambda s: s.rolling(20, min_periods=20).std())

    ma_5 = close.transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma_20 = close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol_ma_20 = volume.transform(lambda s: s.rolling(20, min_periods=20).mean())

    out["dist_ma_5"] = out["Close"] / ma_5 - 1.0
    out["dist_ma_20"] = out["Close"] / ma_20 - 1.0
    out["vol_ratio_20"] = out["Volume"] / vol_ma_20

    return out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
