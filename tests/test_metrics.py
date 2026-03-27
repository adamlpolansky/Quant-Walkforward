from __future__ import annotations

import numpy as np
import pandas as pd

from qwf import metrics


def test_sortino_ratio_matches_expected_value() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])

    got = metrics.sortino_ratio(returns, target=0.0, periods_per_year=4)

    assert np.isclose(got, 1.0)


def test_sortino_ratio_returns_nan_for_edge_cases() -> None:
    no_downside = pd.Series([0.01, 0.02, 0.03])
    zero_downside_dev = pd.Series([-0.01, 0.02, -0.01])

    assert np.isnan(metrics.sortino_ratio(no_downside, target=0.0, periods_per_year=252))
    assert np.isnan(metrics.sortino_ratio(zero_downside_dev, target=0.0, periods_per_year=252))


def test_perf_stats_from_pnl_includes_sortino() -> None:
    pnl = pd.Series([0.01, -0.02, 0.03, -0.01])

    stats = metrics.perf_stats_from_pnl(pnl, periods_per_year=4)

    assert "sortino" in stats
    assert np.isclose(stats["sortino"], 1.0)
