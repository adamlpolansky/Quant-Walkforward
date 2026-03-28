from __future__ import annotations

import pandas as pd

from qwf.signals import normalize_scores_cross_sectionally, prepare_cross_sectional_scores


def _make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "A", "score": 1.0},
            {"date": "2024-01-02", "ticker": "B", "score": 2.0},
            {"date": "2024-01-02", "ticker": "C", "score": 3.0},
            {"date": "2024-01-03", "ticker": "A", "score": 2.0},
            {"date": "2024-01-03", "ticker": "B", "score": 2.0},
            {"date": "2024-01-03", "ticker": "C", "score": 2.0},
        ]
    )


def test_normalize_scores_cross_sectionally_supports_zscore_and_rank() -> None:
    preds = _make_predictions()

    zscore = normalize_scores_cross_sectionally(preds, score_col="score", normalization="zscore", out_col="score_z")
    rank = normalize_scores_cross_sectionally(
        preds,
        score_col="score",
        normalization="rank_to_minus1_plus1",
        out_col="score_rank",
    )

    day_1_z = zscore[zscore["date"] == pd.Timestamp("2024-01-02")]["score_z"].reset_index(drop=True)
    day_1_rank = rank[rank["date"] == pd.Timestamp("2024-01-02")]["score_rank"].reset_index(drop=True)
    day_2_rank = rank[rank["date"] == pd.Timestamp("2024-01-03")]["score_rank"].reset_index(drop=True)

    expected_z = pd.Series([-1.224744871391589, 0.0, 1.224744871391589], name="score_z", dtype=float)
    expected_rank = pd.Series([-1.0, 0.0, 1.0], name="score_rank", dtype=float)
    expected_rank_constant = pd.Series([0.0, 0.0, 0.0], name="score_rank", dtype=float)

    pd.testing.assert_series_equal(day_1_z, expected_z, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(day_1_rank, expected_rank, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(day_2_rank, expected_rank_constant, check_exact=False, rtol=1e-12, atol=1e-12)


def test_prepare_cross_sectional_scores_ema_smoothing_does_not_look_ahead() -> None:
    predictions_1 = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "AAA", "score": 1.0},
            {"date": "2024-01-03", "ticker": "AAA", "score": 2.0},
            {"date": "2024-01-04", "ticker": "AAA", "score": 10.0},
            {"date": "2024-01-02", "ticker": "BBB", "score": -1.0},
            {"date": "2024-01-03", "ticker": "BBB", "score": -2.0},
            {"date": "2024-01-04", "ticker": "BBB", "score": -3.0},
        ]
    )
    predictions_2 = predictions_1.copy()
    predictions_2.loc[
        (predictions_2["ticker"] == "AAA") & (predictions_2["date"] == "2024-01-04"),
        "score",
    ] = 100.0

    out_1 = prepare_cross_sectional_scores(
        predictions_1,
        raw_score_col="score",
        score_smoothing="ema_3",
        score_normalization="none",
    )
    out_2 = prepare_cross_sectional_scores(
        predictions_2,
        raw_score_col="score",
        score_smoothing="ema_3",
        score_normalization="none",
    )

    aaa_1 = out_1[out_1["ticker"] == "AAA"].reset_index(drop=True)
    aaa_2 = out_2[out_2["ticker"] == "AAA"].reset_index(drop=True)

    pd.testing.assert_series_equal(
        aaa_1["score_signal"].iloc[:2],
        aaa_2["score_signal"].iloc[:2],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_prepare_cross_sectional_scores_preserves_missing_raw_scores() -> None:
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "AAA", "score": 1.0},
            {"date": "2024-01-02", "ticker": "BBB", "score": None},
            {"date": "2024-01-03", "ticker": "AAA", "score": 2.0},
            {"date": "2024-01-03", "ticker": "BBB", "score": 3.0},
        ]
    )

    out = prepare_cross_sectional_scores(
        predictions,
        raw_score_col="score",
        score_smoothing="ema_3",
        score_normalization="rank_to_minus1_plus1",
    )

    missing_row = out[(out["ticker"] == "BBB") & (out["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert pd.isna(missing_row["score_signal"])
