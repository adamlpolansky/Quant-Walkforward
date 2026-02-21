from __future__ import annotations

import numpy as np
import pandas as pd


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def compute_pnl_series(
    detail: pd.DataFrame,
    *,
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    fill_missing_ret_with_zero: bool = True,
) -> pd.Series:
    """
    Return a pnl Series (strategy period return):
      pnl = pos_lag * ret

    Preference:
      - if pnl_col exists -> use it
      - else compute from pos_lag (or lag(pos)) and ret
    """
    if pnl_col in detail.columns:
        pnl = _to_float(detail[pnl_col])
        return pnl.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if ret_col not in detail.columns:
        raise ValueError(f"Missing '{ret_col}' and '{pnl_col}' -> cannot compute pnl.")

    ret = _to_float(detail[ret_col])
    if fill_missing_ret_with_zero:
        ret = ret.fillna(0.0)

    if pos_lag_col in detail.columns:
        pos_lag = _to_float(detail[pos_lag_col]).fillna(0.0)
    elif pos_col in detail.columns:
        pos_lag = _to_float(detail[pos_col]).shift(1).fillna(0.0)
    else:
        raise ValueError(f"Missing '{pos_col}'/'{pos_lag_col}' -> cannot compute pnl.")

    pnl = pos_lag * ret
    return pnl.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def equity_from_pnl(
    pnl: pd.Series,
    *,
    start_equity: float = 1.0,
) -> pd.Series:
    """Compounded equity curve from pnl (period returns)."""
    one_plus = (1.0 + pnl).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return float(start_equity) * one_plus.cumprod()


def max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown (negative number, e.g. -0.23)."""
    eq = _to_float(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return np.nan
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def add_pnl(
    detail: pd.DataFrame,
    *,
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    equity_col: str = "equity",
    cum_pnl_col: str = "cum_pnl",
    start_equity: float = 1.0,
    fill_missing_ret_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Add pnl/equity columns to a COPY of detail.

    Note:
      - equity here is compounded from pnl
      - cum_pnl here is set to (equity - start_equity) = cumulative compounded return,
        which is more consistent than pnl.cumsum() for most reporting.
        (If you want additive cum_pnl, we can add another column.)
    """
    out = detail.copy()

    # ensure pos_lag exists if we might need it later
    if pos_lag_col not in out.columns and pos_col in out.columns:
        out[pos_lag_col] = _to_float(out[pos_col]).shift(1).fillna(0.0)

    pnl = compute_pnl_series(
        out,
        ret_col=ret_col,
        pos_col=pos_col,
        pos_lag_col=pos_lag_col,
        pnl_col=pnl_col,
        fill_missing_ret_with_zero=fill_missing_ret_with_zero,
    )
    out[pnl_col] = pnl

    eq = equity_from_pnl(pnl, start_equity=start_equity)
    out[equity_col] = eq
    out[cum_pnl_col] = eq - float(start_equity)

    return out


def fold_summary(
    test_detail: pd.DataFrame,
    *,
    fold_col: str = "fold_id",
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Per-fold metrics computed ONLY on test rows, with equity RESET to 1.0 in each fold.
    This avoids leakage from any equity computed on train+test ranges.
    """
    if fold_col not in test_detail.columns:
        raise ValueError(f"Missing fold column '{fold_col}' in test_detail.columns")

    rows: list[dict] = []

    for fold_id, g in test_detail.groupby(fold_col, sort=True):
        pnl = compute_pnl_series(
            g,
            ret_col=ret_col,
            pos_col=pos_col,
            pos_lag_col=pos_lag_col,
            pnl_col=pnl_col,
            fill_missing_ret_with_zero=True,
        )

        eq = equity_from_pnl(pnl, start_equity=1.0)  # ✅ reset every fold

        n = int(eq.shape[0])
        total_ret = float(eq.iloc[-1] - 1.0) if n else np.nan
        mdd = max_drawdown(eq)

        ann_ret = (1.0 + total_ret) ** (periods_per_year / n) - 1.0 if n and np.isfinite(total_ret) else np.nan

        pnl_clean = pnl.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pnl_clean) > 1:
            mu = float(pnl_clean.mean())
            sd = float(pnl_clean.std(ddof=0))
            ann_vol = sd * np.sqrt(periods_per_year)
            sharpe = (mu / sd) * np.sqrt(periods_per_year) if sd > 0 else np.nan
            win_rate = float((pnl_clean > 0).mean())
        else:
            ann_vol = np.nan
            sharpe = np.nan
            win_rate = np.nan

        row = {
            "fold_id": fold_id,
            "n_test_rows": int(len(g)),
            "total_return": total_ret,
            "max_drawdown": mdd,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "win_rate": win_rate,
        }

        # metadata (if present)
        for c in ["train_start", "train_end", "test_start", "test_end", "source_file"]:
            if c in g.columns:
                row[c] = g[c].iloc[0]

        rows.append(row)

    out = pd.DataFrame(rows)

    preferred = [
        "source_file",
        "fold_id",
        "train_start", "train_end", "test_start", "test_end",
        "n_test_rows",
        "total_return", "max_drawdown",
        "ann_return", "ann_vol", "sharpe", "win_rate",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols].sort_values("fold_id").reset_index(drop=True)