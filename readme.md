# Quant Walk-Forward (first project)

Hi! This is my first ever quant project on my journey to become a strong quant researcher/developer.

## Goal
Build a small but solid research framework that:
- downloads market data (initially via `yfinance`)
- runs **walk-forward** evaluation with rolling **train/test splits** (e.g. 12M train → 1M test)
- includes **anti-leakage guardrails** (strict time indexing, lag-only features, no lookahead)

This is meant to be the foundation for later work with options data, implied volatility surfaces, and eventually **Heston calibration**.

## Current status
- ✅ Day 1: environment + data download + basic preprocessing (SPY)

## Planned milestones
- Week 1: Walk-forward engine + metrics + leakage tests
- Week 2: Options chain + implied volatility (IV) panel
- Week 3: Surface smoothing baseline (spline/SVI/kernel)
- Week 4: Heston calibration + comparison vs. baseline

## Quickstart
```bash
pip install -r requirements.txt
python scripts/pull_data.py
