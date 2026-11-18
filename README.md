# MTF Liquidity & Reload Deterioration

Streamlit app that turns multiscale swing-point hierarchies into a multi-timeframe map of liquidity and price levels, then measures how much price deteriorates between child-order reloads in each volatility regime.

The structure engine (`ATRDirectionalChange`, `HierarchicalExtremes`) is a port of [neurotrader888/market-structure](https://github.com/neurotrader888/market-structure) (MIT). The liquidity zones, regime labelling and reload simulator are additions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (`app.py`, `engine.py`, `requirements.txt`, `.streamlit/config.toml`).
2. On share.streamlit.io choose the repo, branch and `app.py` as the main file.

## Tabs

| Tab | What it shows |
| --- | --- |
| Overview | Headline numbers, key reads, drift vs own-impact split by regime |
| Structure | Candlesticks with swing markers per hierarchy level and step-lines of the latest level high/low |
| Liquidity map | Unswept swing levels from all timeframes clustered into weighted zones, beside a volume profile |
| Reload study | Per-reload deterioration by regime, slippage build-up, regime x interval and distance-to-level heatmaps |
| Method & guide | Definitions, formulas, assumptions |

## Data

- **Synthetic**: 1-minute tape with Markov volatility regimes, U-shaped intraday profile, auction volume. Works offline.
- **CSV**: columns `timestamp, open, high, low, close, volume` (1-minute bars).
- **yfinance**: recent 1-minute history (a few days only).

## Reload deterioration

For a parent order split into children placed every *interval* bars:

`deterioration_k = side * (exec_k - exec_{k-1}) / exec_{k-1} * 10,000` bps (positive = worse)

Split into market drift and own impact (square-root law, permanent + decaying temporary part). Regime is read when the previous child was placed, and higher-timeframe levels are used only after their bar has closed, so there is no look-ahead.

## Limits

Bar-data fills only (no queue or intrabar path), overlapping samples understate standard errors, and regime cut-offs are full-sample quantiles. Research and education only.
