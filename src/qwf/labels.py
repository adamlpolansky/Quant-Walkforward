from __future__ import annotations

import pandas as pd


def add_forward_return_label(
    panel: pd.DataFrame,
    horizon: int = 1,
    price_col: str = "Close",
    label_col: str | None = None,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if "date" not in panel.columns or "ticker" not in panel.columns:
        raise ValueError("Panel must contain 'date' and 'ticker' columns")
    if price_col not in panel.columns:
        raise ValueError(f"Panel is missing price column '{price_col}'")

    target_col = label_col or f"label_{horizon}d_fwd"

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    grouped = out.groupby("ticker", sort=False)[price_col]
    future_price = grouped.shift(-horizon)
    out[target_col] = future_price / out[price_col] - 1.0

    return out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
