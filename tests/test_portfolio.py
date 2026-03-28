from __future__ import annotations

import numpy as np
import pandas as pd

from qwf.portfolio import build_daily_top_bottom_portfolio, summarize_ticker_selection


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


def _make_constant_score_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "ticker": "A", "score": 0.0, "label_1d_fwd": 0.01},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "B", "score": 0.0, "label_1d_fwd": -0.01},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "C", "score": 0.0, "label_1d_fwd": 0.02},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "A", "score": 2.0, "label_1d_fwd": 0.03},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "B", "score": 0.0, "label_1d_fwd": -0.01},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "C", "score": -2.0, "label_1d_fwd": -0.02},
        ]
    )


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


def test_summarize_ticker_selection_returns_expected_columns_and_values() -> None:
    detail, _ = build_daily_top_bottom_portfolio(
        _make_predictions(),
        score_col="score",
        forward_ret_col="label_1d_fwd",
        k=2,
    )

    summary = summarize_ticker_selection(detail)

    assert set(
        [
            "ticker",
            "n_long_days",
            "n_short_days",
            "avg_score",
            "avg_forward_return_when_long",
            "avg_forward_return_when_short",
            "mean_contribution",
            "total_contribution",
            "avg_weight_when_selected",
            "selection_rate_long",
            "selection_rate_short",
        ]
    ).issubset(summary.columns)

    ticker_a = summary.set_index("ticker").loc["A"]
    ticker_e = summary.set_index("ticker").loc["E"]

    assert ticker_a["n_long_days"] == 1
    assert ticker_a["n_short_days"] == 0
    assert np.isclose(ticker_a["avg_score"], 0.55)
    assert np.isclose(ticker_a["avg_forward_return_when_long"], 0.02)
    assert np.isclose(ticker_a["mean_contribution"], 0.005)
    assert np.isclose(ticker_a["total_contribution"], 0.01)
    assert np.isclose(ticker_a["avg_weight_when_selected"], 0.5)
    assert np.isclose(ticker_a["selection_rate_long"], 0.5)

    assert ticker_e["n_short_days"] == 2
    assert np.isclose(ticker_e["avg_forward_return_when_short"], -0.015)
    assert np.isclose(ticker_e["total_contribution"], 0.015)
    assert np.isclose(ticker_e["selection_rate_short"], 1.0)


def test_build_daily_top_bottom_portfolio_skips_constant_score_days() -> None:
    detail, daily = build_daily_top_bottom_portfolio(
        _make_constant_score_predictions(),
        score_col="score",
        forward_ret_col="label_1d_fwd",
        k=1,
    )

    first_day = daily.loc[daily["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    second_day = daily.loc[daily["date"] == pd.Timestamp("2024-01-03")].iloc[0]

    assert bool(first_day["is_constant_score_day"]) is True
    assert first_day["n_longs"] == 0
    assert first_day["n_shorts"] == 0
    assert np.isclose(first_day["gross_exposure"], 0.0)
    assert np.isclose(first_day["portfolio_ret_gross"], 0.0)

    assert bool(second_day["is_constant_score_day"]) is False
    assert second_day["n_longs"] == 1
    assert second_day["n_shorts"] == 1
    assert detail.loc[detail["date"] == pd.Timestamp("2024-01-02"), "weight"].eq(0.0).all()
