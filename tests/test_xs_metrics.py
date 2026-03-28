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


def _make_benchmark_predictions(include_spy: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"date": pd.Timestamp("2024-01-01"), "ticker": "AAA", "label_1d_fwd": 0.50},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "BBB", "label_1d_fwd": 0.40},
        {"date": pd.Timestamp("2024-01-02"), "ticker": "AAA", "label_1d_fwd": 0.01},
        {"date": pd.Timestamp("2024-01-02"), "ticker": "BBB", "label_1d_fwd": 0.02},
        {"date": pd.Timestamp("2024-01-03"), "ticker": "AAA", "label_1d_fwd": -0.02},
        {"date": pd.Timestamp("2024-01-03"), "ticker": "BBB", "label_1d_fwd": 0.01},
    ]
    if include_spy:
        rows.extend(
            [
                {"date": pd.Timestamp("2024-01-01"), "ticker": "SPY", "label_1d_fwd": 0.30},
                {"date": pd.Timestamp("2024-01-02"), "ticker": "SPY", "label_1d_fwd": 0.03},
                {"date": pd.Timestamp("2024-01-03"), "ticker": "SPY", "label_1d_fwd": 0.00},
            ]
        )
    return pd.DataFrame(rows)


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


def test_build_xs_benchmark_curves_aligns_to_strategy_dates() -> None:
    portfolio_daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "equity": [1.01, 1.02],
        }
    )

    curves = metrics.build_xs_benchmark_curves(
        _make_benchmark_predictions(include_spy=True),
        portfolio_daily,
        label_col="label_1d_fwd",
    )

    assert list(curves["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pd.testing.assert_series_equal(
        curves["strategy_equity"],
        pd.Series([1.01, 1.02], name="strategy_equity"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        curves["eq_universe_return"],
        pd.Series([0.02, -0.0033333333333333335], name="eq_universe_return"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        curves["spy_return"],
        pd.Series([0.03, 0.0], name="spy_return"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(curves.loc[0, "eq_universe_equity"], 1.02)
    assert np.isclose(curves.loc[1, "eq_universe_equity"], 1.0166, atol=1e-4)
    assert np.isclose(curves.loc[0, "spy_equity"], 1.03)
    assert np.isclose(curves.loc[1, "spy_equity"], 1.03)


def test_benchmark_perf_summary_handles_missing_spy_gracefully() -> None:
    portfolio_daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "equity": [1.01, 1.02],
        }
    )
    curves = metrics.build_xs_benchmark_curves(
        _make_benchmark_predictions(include_spy=False),
        portfolio_daily,
        label_col="label_1d_fwd",
    )

    spy_summary = metrics.benchmark_perf_summary(curves, return_col="spy_return", prefix="spy")
    eq_summary = metrics.benchmark_perf_summary(curves, return_col="eq_universe_return", prefix="eq_universe")

    assert curves["spy_equity"].isna().all()
    assert set(spy_summary) == {
        "spy_total_return",
        "spy_cagr",
        "spy_ann_vol",
        "spy_sharpe",
        "spy_max_drawdown",
    }
    assert np.isnan(spy_summary["spy_total_return"])
    assert np.isnan(spy_summary["spy_cagr"])
    assert np.isnan(spy_summary["spy_ann_vol"])
    assert np.isnan(spy_summary["spy_sharpe"])
    assert np.isnan(spy_summary["spy_max_drawdown"])
    assert np.isfinite(eq_summary["eq_universe_total_return"])
