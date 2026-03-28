from __future__ import annotations

from typing import Sequence

import pandas as pd

SUPPORTED_LABEL_HORIZONS: tuple[int, ...] = (1, 3, 5)


def _validate_panel_for_labels(panel: pd.DataFrame, *, price_col: str) -> None:
    if "date" not in panel.columns or "ticker" not in panel.columns:
        raise ValueError("Panel must contain 'date' and 'ticker' columns")
    if price_col not in panel.columns:
        raise ValueError(f"Panel is missing price column '{price_col}'")


def add_forward_return_label(
    panel: pd.DataFrame,
    horizon: int = 1,
    price_col: str = "Close",
    label_col: str | None = None,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    _validate_panel_for_labels(panel, price_col=price_col)
    target_col = label_col or f"label_{horizon}d_fwd"

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    grouped = out.groupby("ticker", sort=False)[price_col]
    future_price = grouped.shift(-horizon)
    out[target_col] = future_price / out[price_col] - 1.0

    return out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def add_forward_return_labels(
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = SUPPORTED_LABEL_HORIZONS,
    price_col: str = "Close",
) -> pd.DataFrame:
    if not horizons:
        raise ValueError("At least one horizon is required")

    out = panel.copy()
    seen: set[int] = set()
    for horizon in horizons:
        if horizon in seen:
            continue
        seen.add(int(horizon))
        out = add_forward_return_label(
            out,
            horizon=int(horizon),
            price_col=price_col,
            label_col=f"label_{int(horizon)}d_fwd",
        )
    return out
