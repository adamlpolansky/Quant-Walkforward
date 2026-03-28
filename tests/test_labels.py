from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from qwf.labels import add_forward_return_label, add_forward_return_labels


def _make_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = [
        {"date": dates[0], "ticker": "AAA", "Close": 100.0},
        {"date": dates[1], "ticker": "AAA", "Close": 110.0},
        {"date": dates[2], "ticker": "AAA", "Close": 121.0},
        {"date": dates[3], "ticker": "AAA", "Close": 133.1},
        {"date": dates[0], "ticker": "BBB", "Close": 50.0},
        {"date": dates[1], "ticker": "BBB", "Close": 45.0},
        {"date": dates[2], "ticker": "BBB", "Close": 54.0},
        {"date": dates[3], "ticker": "BBB", "Close": 48.6},
    ]
    panel = pd.DataFrame(rows)
    return panel.sort_values(["ticker", "date"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def _make_long_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    growth = [1.0, 1.1, 1.21, 1.331, 1.4641, 1.61051]
    rows = []
    for ticker, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for date, factor in zip(dates, growth, strict=True):
            rows.append({"date": date, "ticker": ticker, "Close": base * factor})
    return pd.DataFrame(rows).sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def test_add_forward_return_label_is_correct_and_last_row_is_nan() -> None:
    out = add_forward_return_label(_make_panel(), horizon=1)

    aaa = out[out["ticker"] == "AAA"].reset_index(drop=True)
    bbb = out[out["ticker"] == "BBB"].reset_index(drop=True)

    expected_aaa = pd.Series([0.10, 0.10, 0.10, float("nan")], name="label_1d_fwd")
    expected_bbb = pd.Series([-0.10, 0.20, -0.10, float("nan")], name="label_1d_fwd")

    assert_series_equal(aaa["label_1d_fwd"], expected_aaa, check_exact=False, rtol=1e-12, atol=1e-12)
    assert_series_equal(bbb["label_1d_fwd"], expected_bbb, check_exact=False, rtol=1e-12, atol=1e-12)


def test_add_forward_return_label_is_independent_per_ticker() -> None:
    panel_1 = _make_panel()
    panel_2 = panel_1.copy()
    panel_2.loc[panel_2["ticker"] == "BBB", "Close"] *= 10.0

    out_1 = add_forward_return_label(panel_1, horizon=1)
    out_2 = add_forward_return_label(panel_2, horizon=1)

    aaa_1 = out_1[out_1["ticker"] == "AAA"]["label_1d_fwd"].reset_index(drop=True)
    aaa_2 = out_2[out_2["ticker"] == "AAA"]["label_1d_fwd"].reset_index(drop=True)
    assert_series_equal(aaa_1, aaa_2, check_exact=False, rtol=1e-12, atol=1e-12)


def test_add_forward_return_label_only_uses_intended_forward_shift() -> None:
    panel_1 = _make_panel()
    panel_2 = panel_1.copy()
    panel_2.loc[(panel_2["ticker"] == "AAA") & (panel_2["date"] == pd.Timestamp("2024-01-04")), "Close"] = 200.0

    out_1 = add_forward_return_label(panel_1, horizon=1)
    out_2 = add_forward_return_label(panel_2, horizon=1)

    aaa_1 = out_1[out_1["ticker"] == "AAA"]["label_1d_fwd"].reset_index(drop=True)
    aaa_2 = out_2[out_2["ticker"] == "AAA"]["label_1d_fwd"].reset_index(drop=True)

    assert_series_equal(aaa_1.iloc[:2], aaa_2.iloc[:2], check_exact=False, rtol=1e-12, atol=1e-12)


def test_add_forward_return_labels_supports_multiple_horizons() -> None:
    out = add_forward_return_labels(_make_long_panel(), horizons=[1, 3, 5])
    aaa = out[out["ticker"] == "AAA"].reset_index(drop=True)

    expected_1d = pd.Series([0.10, 0.10, 0.10, 0.10, 0.10, float("nan")], name="label_1d_fwd")
    expected_3d = pd.Series([0.331, 0.331, 0.331, float("nan"), float("nan"), float("nan")], name="label_3d_fwd")
    expected_5d = pd.Series([0.61051, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")], name="label_5d_fwd")

    assert_series_equal(aaa["label_1d_fwd"], expected_1d, check_exact=False, rtol=1e-12, atol=1e-12)
    assert_series_equal(aaa["label_3d_fwd"], expected_3d, check_exact=False, rtol=1e-12, atol=1e-12)
    assert_series_equal(aaa["label_5d_fwd"], expected_5d, check_exact=False, rtol=1e-12, atol=1e-12)
