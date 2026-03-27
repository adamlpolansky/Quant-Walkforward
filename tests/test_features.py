from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qwf.features import DAILY_FEATURE_COLUMNS, make_daily_features


def _make_panel(n_days: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows: list[dict[str, object]] = []

    for ticker, offset in [("AAA", 0.0), ("BBB", 20.0)]:
        x = np.arange(n_days, dtype=float)
        close = 100.0 + offset + np.cumsum(0.2 + 0.3 * np.sin(x / 4.0 + offset))
        volume = 1_000_000 + offset * 1000 + 10_000 * (1.0 + np.cos(x / 5.0))

        for date, c, v in zip(dates, close, volume, strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "Open": c - 0.5,
                    "High": c + 0.5,
                    "Low": c - 1.0,
                    "Close": c,
                    "Volume": v,
                }
            )

    panel = pd.DataFrame(rows)
    return panel.sort_values(["ticker", "date"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def test_make_daily_features_creates_expected_columns_and_sorts_output() -> None:
    panel = _make_panel()

    out = make_daily_features(panel)

    for col in DAILY_FEATURE_COLUMNS:
        assert col in out.columns

    expected_order = out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    assert_frame_equal(out[["date", "ticker"]], expected_order[["date", "ticker"]])


def test_make_daily_features_are_computed_per_ticker_independently() -> None:
    panel_1 = _make_panel()
    panel_2 = panel_1.copy()
    panel_2.loc[panel_2["ticker"] == "BBB", "Close"] *= 10.0
    panel_2.loc[panel_2["ticker"] == "BBB", "Volume"] *= 5.0

    out_1 = make_daily_features(panel_1)
    out_2 = make_daily_features(panel_2)

    aaa_1 = out_1[out_1["ticker"] == "AAA"].reset_index(drop=True)
    aaa_2 = out_2[out_2["ticker"] == "AAA"].reset_index(drop=True)
    assert_frame_equal(aaa_1[DAILY_FEATURE_COLUMNS], aaa_2[DAILY_FEATURE_COLUMNS], check_exact=False, rtol=1e-12, atol=1e-12)


def test_make_daily_features_do_not_use_future_rows() -> None:
    panel_1 = _make_panel()
    panel_2 = panel_1.copy()
    cutoff = pd.Timestamp("2020-01-28")
    future_mask = (panel_2["ticker"] == "AAA") & (panel_2["date"] > cutoff)
    panel_2.loc[future_mask, "Close"] *= 3.0
    panel_2.loc[future_mask, "Volume"] *= 2.0

    out_1 = make_daily_features(panel_1)
    out_2 = make_daily_features(panel_2)

    left = out_1[(out_1["ticker"] == "AAA") & (out_1["date"] <= cutoff)].reset_index(drop=True)
    right = out_2[(out_2["ticker"] == "AAA") & (out_2["date"] <= cutoff)].reset_index(drop=True)
    assert_frame_equal(left[DAILY_FEATURE_COLUMNS], right[DAILY_FEATURE_COLUMNS], check_exact=False, rtol=1e-12, atol=1e-12)
