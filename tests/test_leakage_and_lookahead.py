import numpy as np
import pandas as pd
import pytest

from qwf.backtest import (
    ZScoreMRConfig,
    compute_zscore_mr_position,
    backtest_from_position_ret,
    run_wf_backtest_ret,
)
from qwf import metrics


def _make_df(n: int = 220) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # deterministická syntetická price path (bez randomness)
    x = np.arange(n, dtype=float)
    close = 100.0 + np.cumsum(0.05 + 0.5 * np.sin(x / 7.0))

    df = pd.DataFrame({"Close": close}, index=idx)
    df["ret"] = pd.Series(close, index=idx).pct_change().fillna(0.0)
    return df


def _make_plan(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    # 2 foldy, contiguous train/test
    return pd.DataFrame(
        {
            "fold_id": [1, 2],
            "train_start": [idx[0], idx[80]],
            "train_end": [idx[99], idx[179]],
            "test_start": [idx[100], idx[180]],
            "test_end": [idx[119], idx[199]],
        }
    )


def _common_cols(df1: pd.DataFrame, df2: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df1.columns and c in df2.columns]


def test_signal_no_lookahead_future_perturbation():
    """
    Když změním ceny až po cutoff, signál do cutoff se NESMÍ změnit.
    """
    df1 = _make_df()
    df2 = df1.copy()

    cutoff = df1.index[150]

    # Změníme pouze budoucnost (po cutoff)
    future_mask = df2.index > cutoff
    df2.loc[future_mask, "Close"] = df2.loc[future_mask, "Close"] * 10.0 + 123.0
    df2["ret"] = df2["Close"].pct_change().fillna(0.0)

    cfg = ZScoreMRConfig(n=20, K=1.0, step_frac=0.25, ddof=0)

    s1 = compute_zscore_mr_position(df1, price_col="Close", cfg=cfg)
    s2 = compute_zscore_mr_position(df2, price_col="Close", cfg=cfg)

    cols = _common_cols(s1, s2, ["z", "target_pos", "pos"])
    assert cols, "compute_zscore_mr_position nevrátil žádný z očekávaných sloupců ['z', 'target_pos', 'pos']"

    left = s1.loc[:cutoff, cols]
    right = s2.loc[:cutoff, cols]

    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as e:
        raise AssertionError(
            f"LOOK-AHEAD BUG: Signál do cutoff={cutoff} se změnil po úpravě budoucích dat.\n"
            f"Porovnávané sloupce: {cols}\n"
            f"Left shape={left.shape}, Right shape={right.shape}\n"
            f"Detail:\n{e}"
        )


def test_backtest_uses_lagged_position_for_pnl():
    """
    Core anti-lookahead test:
      pnl[t] == pos[t-1] * ret[t]
    """
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame({"ret": [0.0, 0.10, -0.05, 0.02, -0.03, 0.04]}, index=idx)
    pos = pd.Series([0.0, 1.0, 1.0, -1.0, 0.5, 0.0], index=idx)

    bt = backtest_from_position_ret(df, pos=pos, ret_col="ret")

    assert "pnl" in bt.columns, "backtest_from_position_ret nevrátil sloupec 'pnl'"
    expected = pos.shift(1) * df["ret"]

    mask = expected.notna() & bt["pnl"].notna()
    assert int(mask.sum()) >= 1, "Není co porovnat (mask.sum() == 0), zkontroluj výstup backtestu"

    got = bt.loc[mask, "pnl"].to_numpy(dtype=float)
    exp = expected.loc[mask].to_numpy(dtype=float)

    try:
        np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-12)
    except AssertionError as e:
        raise AssertionError(
            "LOOK-AHEAD BUG: pnl není počítané z lagged position.\n"
            f"Expected (pos.shift(1) * ret): {exp}\n"
            f"Got: {got}\n"
            f"Detail:\n{e}"
        )


def test_run_wf_backtest_fold1_independent_of_future_changes():
    """
    Změna dat po fold1 test_end nesmí změnit fold1 výstup.
    """
    df1 = _make_df()
    plan = _make_plan(df1)
    cfg = ZScoreMRConfig(n=20, K=1.0, step_frac=0.25, ddof=0)

    out1 = run_wf_backtest_ret(df=df1, plan=plan, price_col="Close", ret_col="ret", cfg=cfg)

    assert "fold_id" in out1.columns, "run_wf_backtest_ret musí vracet sloupec 'fold_id'"

    fold1_end = pd.Timestamp(plan.loc[plan["fold_id"] == 1, "test_end"].iloc[0])

    # Perturbace pouze po fold1_end
    df2 = df1.copy()
    m = df2.index > fold1_end
    df2.loc[m, "Close"] = df2.loc[m, "Close"] + 5000.0
    df2["ret"] = df2["Close"].pct_change().fillna(0.0)

    out2 = run_wf_backtest_ret(df=df2, plan=plan, price_col="Close", ret_col="ret", cfg=cfg)

    g1 = out1[out1["fold_id"] == 1].copy()
    g2 = out2[out2["fold_id"] == 1].copy()

    cols = _common_cols(g1, g2, ["z", "target_pos", "pos", "pos_lag", "ret", "pnl"])
    assert cols, "Nenašly se žádné společné sloupce pro porovnání fold1"

    try:
        pd.testing.assert_frame_equal(g1[cols], g2[cols], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as e:
        raise AssertionError(
            "LEAKAGE BUG: Fold 1 output se změnil po perturbaci dat až po fold1 test_end.\n"
            f"Fold1 end: {fold1_end}\n"
            f"Porovnávané sloupce: {cols}\n"
            f"Detail:\n{e}"
        )


def test_run_wf_backtest_returns_only_test_rows():
    """
    run_wf_backtest_ret má vracet jen test rows (ne train rows).
    """
    df = _make_df()
    plan = _make_plan(df)
    cfg = ZScoreMRConfig(n=20, K=1.0, step_frac=0.25, ddof=0)

    out = run_wf_backtest_ret(df=df, plan=plan, price_col="Close", ret_col="ret", cfg=cfg)

    assert "fold_id" in out.columns, "Chybí sloupec fold_id"
    assert isinstance(out.index, pd.DatetimeIndex), "Výstup musí mít DatetimeIndex"

    for _, row in plan.iterrows():
        fid = row["fold_id"]
        ts = pd.Timestamp(row["test_start"])
        te = pd.Timestamp(row["test_end"])

        g = out[out["fold_id"] == fid]
        assert not g.empty, f"Fold {fid} je prázdný (čekal jsem test rows {ts}..{te})"

        got_min = g.index.min()
        got_max = g.index.max()

        assert got_min >= ts, f"Fold {fid} obsahuje řádky před test_start: got_min={got_min}, test_start={ts}"
        assert got_max <= te, f"Fold {fid} obsahuje řádky po test_end: got_max={got_max}, test_end={te}"


def test_fold_summary_resets_equity_per_fold():
    """
    Dva foldy se stejnou pnl path musí mít stejný total_return,
    pokud fold_summary resetuje equity na 1.0 v každém foldu.
    """
    idx1 = pd.date_range("2024-01-01", periods=3, freq="D")
    idx2 = pd.date_range("2024-02-01", periods=3, freq="D")

    pnl_path = [0.01, -0.02, 0.03]

    td = pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 2, 2, 2],
            "pnl": pnl_path + pnl_path,
            "train_start": [idx1[0]] * 3 + [idx2[0]] * 3,
            "train_end": [idx1[0]] * 3 + [idx2[0]] * 3,
            "test_start": [idx1[0]] * 3 + [idx2[0]] * 3,
            "test_end": [idx1[-1]] * 3 + [idx2[-1]] * 3,
        },
        index=idx1.append(idx2),
    )

    fs = metrics.fold_summary(td, pnl_col="pnl")
    assert len(fs) == 2, f"Čekal jsem 2 foldy ve summary, dostal jsem {len(fs)}"

    tr1 = float(fs.loc[fs["fold_id"] == 1, "total_return"].iloc[0])
    tr2 = float(fs.loc[fs["fold_id"] == 2, "total_return"].iloc[0])

    try:
        np.testing.assert_allclose(tr1, tr2, rtol=0.0, atol=1e-12)
    except AssertionError as e:
        raise AssertionError(
            "BUG: fold_summary zřejmě neresetuje equity per fold (foldy se stejnou pnl path mají jiný total_return).\n"
            f"fold1 total_return={tr1}\n"
            f"fold2 total_return={tr2}\n"
            f"Detail:\n{e}"
        )