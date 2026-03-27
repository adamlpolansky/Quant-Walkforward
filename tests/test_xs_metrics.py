from __future__ import annotations

import numpy as np
import pandas as pd

from qwf import metrics
from qwf.portfolio import build_daily_top_bottom_portfolio


def _make_ic_predictions() -> pd.DataFrame:
    date_1 = pd.Timestamp("2024-01-02")
    date_2 = pd.Timestamp("2024-01-03")
    return pd.DataFrame(
        [
            {"date": date_1, "ticker": "A", "score": 1.0, "label_1d_fwd": 1.0},
            {"date": date_1, "ticker": "B", "score": 2.0, "label_1d_fwd": 2.0},
            {"date": date_1, "ticker": "C", "score": 3.0, "label_1d_fwd": 3.0},
            {"date": date_2, "ticker": "A", "score": 1.0, "label_1d_fwd": 3.0},
            {"date": date_2, "ticker": "B", "score": 2.0, "label_1d_fwd": 2.0},
            {"date": date_2, "ticker": "C", "score": 3.0, "label_1d_fwd": 1.0},
        ]
    )


def _make_portfolio_predictions() -> pd.DataFrame:
    date_1 = pd.Timestamp("2024-01-02")
    date_2 = pd.Timestamp("2024-01-03")
    return pd.DataFrame(
        [
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
    )


def test_daily_cross_sectional_ic_and_summary() -> None:
    ic_daily = metrics.daily_cross_sectional_ic(
        _make_ic_predictions(),
        score_col="score",
        forward_ret_col="label_1d_fwd",
    )
    summary = metrics.summarize_ic(ic_daily)

    expected_ic = pd.Series([1.0, -1.0], name="ic_pearson")
    pd.testing.assert_series_equal(
        ic_daily["ic_pearson"].reset_index(drop=True),
        expected_ic,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(summary["mean_ic"], 0.0)
    assert np.isclose(summary["ic_std"], 1.0)
    assert np.isclose(summary["ic_ir"], 0.0)
    assert summary["n_ic_days"] == 2


def test_daily_spread_and_portfolio_summary() -> None:
    detail, daily = build_daily_top_bottom_portfolio(
        _make_portfolio_predictions(),
        score_col="score",
        forward_ret_col="label_1d_fwd",
        k=2,
        cost_bps_per_turnover=10.0,
    )

    spread_daily = metrics.daily_long_short_spread(detail)
    spread_summary = metrics.summarize_spread(spread_daily)
    perf_summary = metrics.portfolio_perf_summary(daily)

    expected_spread = pd.Series([0.04, 0.0275], name="spread")
    pd.testing.assert_series_equal(
        spread_daily["spread"].reset_index(drop=True),
        expected_spread,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(spread_summary["mean_spread"], expected_spread.mean())
    assert spread_summary["n_spread_days"] == 2

    assert perf_summary["n_traded_days"] == 2
    assert np.isclose(perf_summary["mean_turnover"], 1.0)
    assert np.isclose(perf_summary["mean_cost"], 0.001)
    assert "sortino_net" in perf_summary
    assert perf_summary["total_return_net"] < perf_summary["total_return_gross"]
