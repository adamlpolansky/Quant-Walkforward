from __future__ import annotations

import numpy as np
import pandas as pd

from qwf.portfolio import build_daily_top_bottom_portfolio


def _make_predictions() -> pd.DataFrame:
    date_1 = pd.Timestamp("2024-01-02")
    date_2 = pd.Timestamp("2024-01-03")

    rows = [
        {"date": date_1, "ticker": "A", "score": 0.90, "label_1d_fwd": 0.02},
        {"date": date_1, "ticker": "B", "score": 0.80, "label_1d_fwd": 0.01},
        {"date": date_1, "ticker": "C", "score": 0.10, "label_1d_fwd": 0.00},
        {"date": date_1, "ticker": "D", "score": -0.10, "label_1d_fwd": -0.01},
        {"date": date_1, "ticker": "E", "score": -0.80, "label_1d_fwd": -0.02},
        {"date": date_1, "ticker": "F", "score": -0.90, "label_1d_fwd": -0.03},
        {"date": date_2, "ticker": "A", "score": 0.20, "label_1d_fwd": 0.00},
        {"date": date_2, "ticker": "B", "score": 0.85, "label_1d_fwd": 0.015},
        {"date": date_2, "ticker": "C", "score": 0.75, "label_1d_fwd": 0.01},
        {"date": date_2, "ticker": "D", "score": -0.70, "label_1d_fwd": -0.02},
        {"date": date_2, "ticker": "E", "score": -0.60, "label_1d_fwd": -0.01},
        {"date": date_2, "ticker": "F", "score": -0.10, "label_1d_fwd": 0.005},
    ]
    return pd.DataFrame(rows)


def test_build_daily_top_bottom_portfolio_selects_expected_assets_and_weights() -> None:
    detail, daily = build_daily_top_bottom_portfolio(
        _make_predictions(),
        score_col="score",
        forward_ret_col="label_1d_fwd",
        k=2,
    )

    day_1 = detail[detail["date"] == pd.Timestamp("2024-01-02")].set_index("ticker")
    day_2 = detail[detail["date"] == pd.Timestamp("2024-01-03")].set_index("ticker")

    assert np.isclose(day_1["weight"].sum(), 0.0)
    assert np.isclose(day_2["weight"].sum(), 0.0)

    assert np.isclose(day_1.loc["A", "weight"], 0.5)
    assert np.isclose(day_1.loc["B", "weight"], 0.5)
    assert np.isclose(day_1.loc["E", "weight"], -0.5)
    assert np.isclose(day_1.loc["F", "weight"], -0.5)

    assert np.isclose(day_2.loc["B", "weight"], 0.5)
    assert np.isclose(day_2.loc["C", "weight"], 0.5)
    assert np.isclose(day_2.loc["D", "weight"], -0.5)
    assert np.isclose(day_2.loc["E", "weight"], -0.5)

    assert np.isclose(day_1.loc[["A", "B"], "weight"].nunique(), 1.0)
    assert np.isclose(day_1.loc[["E", "F"], "weight"].abs().nunique(), 1.0)
    assert np.isclose(day_2.loc[["B", "C"], "weight"].nunique(), 1.0)
    assert np.isclose(day_2.loc[["D", "E"], "weight"].abs().nunique(), 1.0)

    assert list(daily["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_build_daily_top_bottom_portfolio_returns_and_turnover_are_deterministic() -> None:
    _, daily = build_daily_top_bottom_portfolio(
        _make_predictions(),
        score_col="score",
        forward_ret_col="label_1d_fwd",
        k=2,
    )

    expected_gross = pd.Series([0.04, 0.0275], name="portfolio_ret_gross")
    expected_turnover = pd.Series([1.0, 1.0], name="turnover")

    pd.testing.assert_series_equal(
        daily["portfolio_ret_gross"].reset_index(drop=True),
        expected_gross,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        daily["turnover"].reset_index(drop=True),
        expected_turnover,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert (daily["gross_exposure"] == 2.0).all()
    assert (daily["n_longs"] == 2).all()
    assert (daily["n_shorts"] == 2).all()
