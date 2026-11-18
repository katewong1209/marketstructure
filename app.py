"""
Multi-Timeframe Liquidity & Reload Deterioration - Streamlit app.

Market-structure engine ported from neurotrader888/market-structure (MIT).
Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine as E

st.set_page_config(
    page_title="MTF Liquidity & Reload Deterioration",
    page_icon="◩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme (dark slate, GitHub-profile inspired)
# --------------------------------------------------------------------------
BG, SURFACE, BORDER = "#1e2327", "#262d32", "#3a424a"
TEXT, MUTED, ACCENT = "#e6edf3", "#8b949e", "#58a6ff"
GREEN, AMBER, RED, PURPLE = "#3fb950", "#d29922", "#f85149", "#bc8cff"
GRID = "#2f373e"
LEVEL_COLORS = ["#6e7681", "#58a6ff", "#bc8cff", "#f778ba", "#ffa657", "#ff7b72", "#e3b341"]
REGIME_COLORS = {
    2: [ACCENT, RED],
    3: [ACCENT, "#7d8590", AMBER],
    4: [ACCENT, "#7d8590", AMBER, RED],
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hdr {display:flex; align-items:center; gap:10px; font-size:15px; color:§MUTED§; margin-bottom:6px;}
.hdr b {color:§TEXT§;}
.avatar {width:26px; height:26px; border-radius:50%; background:linear-gradient(135deg,§ACCENT§,§PURPLE§);
         display:inline-block;}
.pill {border:1px solid §BORDER§; border-radius:999px; padding:1px 9px; font-size:12px; color:§MUTED§;}
.hero {font-size:16px; color:§MUTED§; line-height:1.55; max-width:980px; margin:2px 0 14px 0;}
.kcard {background:§SURFACE§; border:1px solid §BORDER§; border-radius:8px; padding:12px 14px; height:100%;}
.kcard .t {font-size:12px; color:§MUTED§; text-transform:uppercase; letter-spacing:.04em;}
.kcard .v {font-size:26px; font-weight:600; color:§TEXT§; line-height:1.25;}
.kcard .s {font-size:12.5px; color:§MUTED§; margin-top:2px;}
.pinned {background:§SURFACE§; border:1px solid §BORDER§; border-radius:8px; padding:14px 16px; height:100%;}
.pinned .n {font-weight:600; color:§ACCENT§; font-size:15px;}
.pinned .d {color:§MUTED§; font-size:13.5px; margin:4px 0 8px 0;}
.pinned .l {font-size:12.5px; color:§MUTED§;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle;}
.read {border-left:3px solid §ACCENT§; background:§SURFACE§; padding:10px 14px; border-radius:0 8px 8px 0;
       margin:6px 0; color:§TEXT§; font-size:14.5px;}
.foot {border-top:1px solid §BORDER§; margin-top:28px; padding-top:12px; font-size:12.5px; color:§MUTED§;}
.foot a {color:§ACCENT§;}
</style>
"""
for k, v in {"§BG§": BG, "§SURFACE§": SURFACE, "§BORDER§": BORDER, "§TEXT§": TEXT, "§MUTED§": MUTED,
             "§ACCENT§": ACCENT, "§PURPLE§": PURPLE}.items():
    CSS = CSS.replace(k, v)
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def show(fig, **kw):
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def style_fig(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=10, r=10, t=44 if title else 24, b=10),
        font=dict(color=TEXT, size=12),
        legend=dict(orientation="h", y=1.06, x=0),
        title=dict(text=title, x=0, font=dict(size=14)) if title else None,
        hoverlabel=dict(bgcolor=SURFACE),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def kcard(title, value, sub=""):
    return (f'<div class="kcard"><div class="t">{title}</div><div class="v">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def session_rangebreaks(idx: pd.DatetimeIndex):
    """Hide weekends and overnight gaps for session-based data."""
    hours = idx.hour + idx.minute / 60
    start, end = float(hours.min()), float(hours.max()) + 1 / 60
    brks = [dict(bounds=["sat", "mon"])]
    if end - start < 20:
        brks.append(dict(bounds=[end, start], pattern="hour"))
    return brks


def regime_colors(n):
    return REGIME_COLORS.get(n, [ACCENT] * n)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating synthetic tape...")
def load_synthetic(days, seed, base_vol_bps):
    return E.synthetic_ohlcv(days=days, seed=seed, base_vol_bps=base_vol_bps)


@st.cache_data(show_spinner="Fetching from yfinance...", ttl=900)
def load_yf(ticker, period):
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval="1m", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return E.clean_ohlcv(raw)


@st.cache_data(show_spinner="Reading file...")
def load_csv(raw_bytes):
    return E.clean_ohlcv(pd.read_csv(io.BytesIO(raw_bytes)))


@st.cache_data(show_spinner="Building market structure...")
def get_structure(df, minutes, levels, atr_lb, mult):
    return E.run_structure(E.resample_ohlcv(df, minutes), minutes, levels, atr_lb, mult)


@st.cache_data(show_spinner="Running reload simulation...")
def run_study(df, anchor_minutes, levels, atr_lb, mult, anchor_level, reg_kind, n_reg, reg_lb,
              side, n_children, intervals, stride, style, patience, hs, through, pct_adv, y, perm, hl):
    anchor = get_structure(df, anchor_minutes, levels, atr_lb, mult)
    codes, names, sig = E.regime_arrays(df, reg_kind, n_reg, reg_lb)
    arr = E.prepare_sim_arrays(df, codes, sig, anchor, anchor_level)
    parts = [
        E.simulate_reloads(arr, side, n_children, iv, stride, style, patience, hs, through, pct_adv, y, perm, hl)
        for iv in intervals
    ]
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cover = float(np.isfinite(arr["lvl_lo"]).mean()), float(np.isfinite(arr["lvl_hi"]).mean())
    return res, names, cover, codes


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    st.caption("Everything recomputes from these settings.")

    st.markdown("**Data**")
    src = st.radio("Source", ["Synthetic (regime-switching)", "Upload CSV", "yfinance (1m)"], label_visibility="collapsed")
    if src.startswith("Synthetic"):
        days = st.slider("Trading days", 5, 30, 15)
        seed = st.number_input("Seed", 0, 9999, 7)
        base_vol = st.slider("Base 1-min vol (bps)", 1.5, 8.0, 3.5, 0.5)
    elif src.startswith("Upload"):
        up = st.file_uploader("OHLCV CSV (timestamp, open, high, low, close, volume)", type=["csv"])
    else:
        ticker = st.text_input("Ticker", "SPY")
        period = st.selectbox("Period", ["1d", "5d", "7d"], index=1)

    st.markdown("**Market structure**")
    levels = st.slider("Hierarchy levels", 3, 7, 5)
    atr_lb = st.slider("ATR lookback (bars)", 10, 120, 30)
    mult = st.slider("Reversal = x ATR", 0.5, 4.0, 1.0, 0.25)
    tfs = st.multiselect("Timeframes", list(E.TIMEFRAMES), default=["1min", "5min", "15min", "60min"])
    if not tfs:
        tfs = ["1min"]

    st.markdown("**Volatility regimes**")
    reg_kind = st.radio("Regime variable", ["Realized vol", "ATR %"], horizontal=True)
    n_reg = st.slider("Regimes", 2, 4, 3)
    reg_lb = st.slider("Vol lookback (bars)", 20, 240, 60, 10)

    st.markdown("**Reload study**")
    side_lbl = st.radio("Parent order", ["Buy", "Sell"], horizontal=True)
    n_children = st.slider("Child orders per parent", 3, 30, 10)
    intervals = st.multiselect("Reload interval (1-min bars)", [1, 2, 3, 5, 10, 15, 20, 30], default=[2, 5, 10])
    if not intervals:
        intervals = [5]
    intervals = sorted(intervals)
    detail_iv = st.selectbox("Interval for detail charts", intervals, index=min(1, len(intervals) - 1))
    style = st.radio("Execution style", ["Aggressive", "Passive at level"])
    anchor_tf = st.selectbox("Anchor timeframe", tfs, index=min(1, len(tfs) - 1))
    anchor_level = st.slider("Anchor level", 0, levels - 1, min(2, levels - 1))
    patience = st.slider("Passive patience (bars)", 1, 30, 10)
    with st.expander("Cost model"):
        half_spread = st.number_input("Half spread (bps)", 0.0, 20.0, 1.0, 0.25)
        through = st.number_input("Trade-through to fill passive (bps)", 0.0, 10.0, 1.0, 0.25)
        pct_adv = st.number_input("Parent size (% of ADV)", 0.1, 50.0, 5.0, 0.5)
        impact_y = st.number_input("Impact coefficient Y", 0.0, 3.0, 0.5, 0.1)
        perm_share = st.slider("Permanent share of impact", 0.0, 1.0, 0.33, 0.01)
        halflife = st.number_input("Temporary impact half-life (bars)", 1.0, 120.0, 10.0, 1.0)
    stride = st.slider("Sample every N bars", 5, 120, 30, 5)

# --------------------------------------------------------------------------
# Load + compute
# --------------------------------------------------------------------------
try:
    if src.startswith("Synthetic"):
        df = load_synthetic(days, int(seed), base_vol)
    elif src.startswith("Upload"):
        if up is None:
            st.info("Upload a 1-minute OHLCV CSV in the sidebar, or switch to the synthetic tape.")
            st.stop()
        df = load_csv(up.getvalue())
    else:
        df = load_yf(ticker, period)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load data: {exc}")
    st.stop()

if len(df) < max(300, atr_lb * 5):
    st.error("Not enough bars. Need a few hundred 1-minute bars at least.")
    st.stop()

sessions = E.tag_sessions(df)
day_vol = df["volume"].groupby(df.index.normalize()).sum()
adv = float(day_vol.mean())
n_days = int(day_vol.shape[0])
last_px = float(df["close"].iloc[-1])

structs = {tf: get_structure(df, E.TIMEFRAMES[tf], levels, atr_lb, mult) for tf in tfs}
if anchor_tf not in structs:
    structs[anchor_tf] = get_structure(df, E.TIMEFRAMES[anchor_tf], levels, atr_lb, mult)
total_swings = sum(len(e) for s in structs.values() for e in s.ext)

side = 1 if side_lbl == "Buy" else -1
res, reg_names, cover, reg_codes = run_study(
    df, E.TIMEFRAMES[anchor_tf], levels, atr_lb, mult, anchor_level, reg_kind, n_reg, reg_lb,
    side, n_children, tuple(intervals), stride, style, patience, half_spread, through, pct_adv,
    impact_y, perm_share, halflife,
)
rcol = regime_colors(len(reg_names))
res_d = res[res["interval"] == detail_iv] if not res.empty else res
summ = E.regime_summary(res_d, reg_names) if not res.empty else pd.DataFrame()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="hdr"><span class="avatar"></span><span>katewong1209 / <b>mtf-liquidity-reload</b></span>'
    '<span class="pill">Public</span></div>',
    unsafe_allow_html=True,
)
st.title("Multi-timeframe liquidity & reload deterioration")
st.markdown(
    '<div class="hero">Multiscale swing-point hierarchies across timeframes are turned into a map of resting '
    "liquidity and price levels, then used to measure how much price deteriorates between child-order reloads in "
    "each volatility regime. Use it to size the reload interval and anchor levels so a parent order avoids "
    "implementation shortfall, adverse selection and excessive slippage during volatile breakouts.</div>",
    unsafe_allow_html=True,
)

n_struct_tab = sum(len(e) for e in structs[anchor_tf].ext)
tab_over, tab_struct, tab_liq, tab_study, tab_guide = st.tabs(
    ["Overview", f"Structure ({n_struct_tab})", "Liquidity map", f"Reload study ({len(res_d[res_d['k'] >= 1]) if not res_d.empty else 0})", "Method & guide"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kcard("Bars", f"{len(df):,}", f"{n_days} sessions, {df.index[0]:%d %b} to {df.index[-1]:%d %b}"), unsafe_allow_html=True)
    c2.markdown(kcard("ADV", f"{adv:,.0f}", "mean daily volume, drives impact sizing"), unsafe_allow_html=True)
    c3.markdown(kcard("Swings confirmed", f"{total_swings:,}", f"across {len(structs)} timeframes, {levels} levels"), unsafe_allow_html=True)
    if not summ.empty:
        c4.markdown(kcard("Mean reload deterioration", f"{res_d[res_d['k'] >= 1]['det_bps'].mean():.2f} bps",
                          f"{style.lower()}, {n_children} children every {detail_iv} bar(s)"), unsafe_allow_html=True)
    else:
        c4.markdown(kcard("Mean reload deterioration", "n/a", "not enough bars"), unsafe_allow_html=True)

    st.write("")
    p1, p2, p3 = st.columns(3)
    p1.markdown(
        '<div class="pinned"><div class="n">Multiscale structure</div>'
        '<div class="d">ATR directional-change swings, promoted level by level into a swing-point hierarchy.</div>'
        f'<div class="l"><span class="dot" style="background:{LEVEL_COLORS[1]}"></span>{n_struct_tab} swings on {anchor_tf}, '
        f'{len(structs[anchor_tf].ext[-1])} at level {levels - 1}</div></div>', unsafe_allow_html=True)
    p2.markdown(
        '<div class="pinned"><div class="n">Liquidity map</div>'
        '<div class="d">Unswept highs and lows from every timeframe, clustered into weighted price zones.</div>'
        f'<div class="l"><span class="dot" style="background:{AMBER}"></span>Stops above swing highs, below swing lows</div></div>',
        unsafe_allow_html=True)
    p3.markdown(
        '<div class="pinned"><div class="n">Reload study</div>'
        '<div class="d">Adverse price move between child reloads, split into market drift and own impact, by regime.</div>'
        f'<div class="l"><span class="dot" style="background:{RED}"></span>{n_reg} regimes on {reg_kind.lower()}</div></div>',
        unsafe_allow_html=True)

    st.write("")
    if not summ.empty:
        st.markdown("#### Key reads")
        lo_r, hi_r = summ.index[0], summ.index[-1]
        lo_v, hi_v = summ["mean det (bps)"].iloc[0], summ["mean det (bps)"].iloc[-1]
        ratio = hi_v / lo_v if lo_v > 0 else np.nan
        reads = [f"Reloads placed in the <b>{hi_r}</b> regime deteriorate <b>{hi_v:.2f} bps</b> on average versus "
                 f"<b>{lo_v:.2f} bps</b> in <b>{lo_r}</b>" + (f" ({ratio:.1f}x)." if np.isfinite(ratio) else ".")]
        reads.append(f"Own market impact is {summ['own impact (bps)'].iloc[-1]:.2f} bps of that in {hi_r}; the rest is market drift "
                     f"({summ['market (bps)'].iloc[-1]:.2f} bps).")
        best_iv = res[(res["k"] >= 1) & (res["regime"] == len(reg_names) - 1)].groupby("interval")["det_bps"].mean()
        if len(best_iv) > 1:
            reads.append(f"In {hi_r}, the interval with the least deterioration per reload is <b>{int(best_iv.idxmin())} bar(s)</b> "
                         f"({best_iv.min():.2f} bps); the worst is {int(best_iv.idxmax())} bar(s) ({best_iv.max():.2f} bps).")
        if style == "Passive at level":
            reads.append(f"{summ['chased %'].iloc[-1]:.0f}% of reloads in {hi_r} were not filled at the anchor level and had to chase.")
        for r in reads:
            st.markdown(f'<div class="read">{r}</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_bar(x=summ.index, y=summ["market (bps)"], name="Market drift", marker_color=ACCENT)
        fig.add_bar(x=summ.index, y=summ["own impact (bps)"], name="Own impact", marker_color=AMBER)
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title="bps per reload")
        show(style_fig(fig, 320, "Mean deterioration per reload by volatility regime"))
    st.caption("Results use the synthetic tape unless you switch the data source. They show how the method behaves, not a forecast.")

# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
with tab_struct:
    sc1, sc2, sc3 = st.columns([1, 1, 1])
    s_tf = sc1.selectbox("Timeframe", list(structs), index=list(structs).index(anchor_tf), key="s_tf")
    s_levels = sc2.multiselect("Levels shown", list(range(levels)), default=list(range(1, min(levels, 4))), key="s_lv")
    max_days = max(1, n_days)
    win = sc3.slider("Window (days)", 1, max_days, min(3, max_days), key="s_win")
    st_ = structs[s_tf]
    d = E.resample_ohlcv(df, E.TIMEFRAMES[s_tf])
    mask = d.index >= d.index[-1] - pd.Timedelta(days=win)
    dw = d[mask]
    line_lv = st.selectbox("Level step-lines", ["none"] + list(range(levels)), index=min(3, levels), key="s_line")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dw.index, open=dw["open"], high=dw["high"], low=dw["low"], close=dw["close"],
                                 name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
                                 showlegend=False))
    if line_lv != "none":
        lv = int(line_lv)
        fig.add_trace(go.Scatter(x=dw.index, y=st_.hi[mask, lv], mode="lines", line=dict(color=AMBER, width=1.4, shape="hv"),
                                 name=f"L{lv} high"))
        fig.add_trace(go.Scatter(x=dw.index, y=st_.lo[mask, lv], mode="lines", line=dict(color=ACCENT, width=1.4, shape="hv"),
                                 name=f"L{lv} low"))
    for lv in s_levels:
        e = st_.ext[lv]
        if e.empty:
            continue
        e = e[e["timestamp"] >= dw.index[0]]
        for t, sym, nm in ((1, "triangle-down", "high"), (-1, "triangle-up", "low")):
            ee = e[e["ext_type"] == t]
            fig.add_trace(go.Scatter(
                x=ee["timestamp"], y=ee["price"], mode="markers", name=f"L{lv} {nm}",
                marker=dict(symbol=sym, size=7 + 3 * lv, color=LEVEL_COLORS[lv % len(LEVEL_COLORS)],
                            line=dict(width=1, color=BG)),
            ))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=session_rangebreaks(d.index))
    show(style_fig(fig, 520, f"{s_tf} swing-point hierarchy"))

    cL, cR = st.columns([1, 1])
    ls = E.level_stats(st_)
    with cL:
        st.markdown("**Swings per level**")
        figb = go.Figure(go.Bar(x=[f"L{i}" for i in ls["level"]], y=ls["swings"],
                                marker_color=[LEVEL_COLORS[i % len(LEVEL_COLORS)] for i in ls["level"]]))
        show(style_fig(figb, 260))
    with cR:
        st.markdown("**Scale of each level**")
        table(ls.round(1), hide_index=True)
    checks = {lv: E.extremes_sanity_checks(e) for lv, e in enumerate(st_.ext)}
    ok = all(all(v.values()) for v in checks.values())
    st.caption(("Integrity checks passed" if ok else "Integrity check failed") +
               ": highs and lows alternate, indices are ordered and confirmation never precedes the swing, at every level.")

# --------------------------------------------------------------------------
# Liquidity map
# --------------------------------------------------------------------------
with tab_liq:
    lc1, lc2, lc3, lc4 = st.columns(4)
    m_levels = lc1.multiselect("Levels included", list(range(levels)), default=list(range(1, min(levels, 4))), key="m_lv")
    only_un = lc2.checkbox("Only unswept levels", True)
    tol = lc3.slider("Cluster tolerance (bps)", 1, 40, 8)
    top_n = lc4.slider("Zones shown", 3, 25, 10)
    lv_df = E.collect_levels(structs, df, m_levels, only_un)
    zones = E.cluster_levels(lv_df, tol, last_px)
    if zones.empty:
        st.info("No levels match these settings. Add timeframes or lower the minimum level.")
    else:
        zt = zones.head(top_n)
        m_tf = st.selectbox("Chart timeframe", list(structs), index=list(structs).index(anchor_tf), key="m_tf")
        mw = st.slider("Window (days)", 1, max(1, n_days), min(3, max(1, n_days)), key="m_win")
        dm = E.resample_ohlcv(df, E.TIMEFRAMES[m_tf])
        dm = dm[dm.index >= dm.index[-1] - pd.Timedelta(days=mw)]
        vp_mid, vp_vol = E.volume_profile(df[df.index >= dm.index[0]])
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, column_widths=[0.70, 0.15, 0.15],
                            horizontal_spacing=0.01)
        fig.add_trace(go.Candlestick(x=dm.index, open=dm["open"], high=dm["high"], low=dm["low"], close=dm["close"],
                                     increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False), 1, 1)
        for _, z in zt.iterrows():
            colr = AMBER if z["position"] == "Above" else ACCENT
            fig.add_hline(y=z["price"], line=dict(color=colr, width=1, dash="dot"), opacity=0.35 + 0.5 * z["score"] / zt["score"].max(),
                          row=1, col=1)
        fig.add_trace(go.Bar(x=zt["score"], y=zt["price"], orientation="h", width=last_px * 0.0006,
                             marker_color=[AMBER if p == "Above" else ACCENT for p in zt["position"]],
                             showlegend=False, name="Zone score"), 1, 2)
        fig.add_trace(go.Bar(x=vp_vol, y=vp_mid, orientation="h", marker_color="#5b6670", showlegend=False,
                             name="Volume"), 1, 3)
        fig.update_xaxes(rangeslider_visible=False, rangebreaks=session_rangebreaks(dm.index), row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(showticklabels=False, row=1, col=3)
        show(style_fig(fig, 560, "Price levels with unswept liquidity (zone score | volume profile)"))

        above = zt[zt["position"] == "Above"].sort_values("dist (bps)").head(1)
        below = zt[zt["position"] == "Below"].sort_values("dist (bps)", ascending=False).head(1)
        k1, k2, k3 = st.columns(3)
        k1.markdown(kcard("Last price", f"{last_px:,.2f}", f"{len(lv_df)} levels, {len(zones)} zones"), unsafe_allow_html=True)
        if not above.empty:
            k2.markdown(kcard("Nearest zone above", f"{above['price'].iloc[0]:,.2f}", f"+{above['dist (bps)'].iloc[0]:.0f} bps, {above['side'].iloc[0]}"), unsafe_allow_html=True)
        if not below.empty:
            k3.markdown(kcard("Nearest zone below", f"{below['price'].iloc[0]:,.2f}", f"{below['dist (bps)'].iloc[0]:.0f} bps, {below['side'].iloc[0]}"), unsafe_allow_html=True)
        st.write("")
        table(zt.assign(**{"last confirmed": zt["last confirmed"].dt.strftime("%d %b %H:%M")}).round(2), hide_index=True)
        st.download_button("Download zones (CSV)", zones.to_csv(index=False).encode(), "liquidity_zones.csv", "text/csv")
        st.caption("Zone score = sum over swings of (level + 1) x sqrt(timeframe minutes). A swing is swept once price later trades through it.")

# --------------------------------------------------------------------------
# Reload study
# --------------------------------------------------------------------------
with tab_study:
    if res_d.empty or summ.empty:
        st.warning("Not enough data for the reload simulation. Reduce children / interval or add days.")
    else:
        if style == "Passive at level" and min(cover) < 0.2:
            st.warning(f"Anchor level L{anchor_level} on {anchor_tf} exists for only {min(cover) * 100:.0f}% of bars, so most passive "
                       "orders fall back to the touch. Lower the anchor level or timeframe.")
        rd = res_d[res_d["k"] >= 1]
        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(kcard("Mean per reload", f"{rd['det_bps'].mean():.2f} bps", f"p90 {rd['det_bps'].quantile(0.9):.1f} bps"), unsafe_allow_html=True)
        a2.markdown(kcard("Adverse reloads", f"{(rd['det_bps'] > 0).mean() * 100:.0f}%", "price worse than previous child"), unsafe_allow_html=True)
        fin = res_d[res_d["k"] == n_children - 1]["slip_bps"]
        a3.markdown(kcard("Final slippage vs arrival", f"{fin.mean():.1f} bps", f"{len(fin)} simulated parents"), unsafe_allow_html=True)
        a4.markdown(kcard("Structure breaks", f"{rd['struct_break'].mean() * 100:.0f}%" if rd["struct_break"].notna().any() else "n/a",
                          f"price broke the L{anchor_level} {'high' if side == 1 else 'low'} between reloads"), unsafe_allow_html=True)
        st.write("")

        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            for i, nm in enumerate(reg_names):
                v = rd[rd["regime"] == i]["det_bps"].dropna()
                if len(v):
                    fig.add_trace(go.Box(y=v, name=nm, marker_color=rcol[i], boxmean=True, showlegend=False))
            fig.update_yaxes(title="deterioration (bps)")
            show(style_fig(fig, 360, "Distribution per reload, by regime"))
        with g2:
            fig = go.Figure()
            for i, nm in enumerate(reg_names):
                cur = res_d[res_d["regime_start"] == i].groupby("k")["slip_bps"].mean()
                if len(cur):
                    fig.add_trace(go.Scatter(x=cur.index + 1, y=cur.values, mode="lines+markers", name=nm,
                                             line=dict(color=rcol[i], width=2)))
            fig.update_xaxes(title="child order #")
            fig.update_yaxes(title="cumulative slippage vs arrival (bps)")
            show(style_fig(fig, 360, "Slippage build-up through the parent order"))

        st.markdown("**Regime summary**")
        table(summ.round(2))

        h1, h2 = st.columns(2)
        with h1:
            r_all = res[(res["k"] >= 1) & (res["regime"] >= 0)]
            piv = r_all.pivot_table(index="regime", columns="interval", values="det_bps", aggfunc="mean")
            piv.index = [reg_names[i] if i < len(reg_names) else f"R{i}" for i in piv.index]
            fig = go.Figure(go.Heatmap(z=piv.values, x=[f"{c} bar" for c in piv.columns], y=piv.index,
                                       colorscale=[[0, "#1f3b57"], [0.5, "#3a4a5a"], [1, "#f0883e"]],
                                       text=np.round(piv.values, 2), texttemplate="%{text}",
                                       colorbar=dict(title="bps")))
            show(style_fig(fig, 320, "Mean deterioration: regime x reload interval"))
        with h2:
            bins = [-np.inf, 0, 0.5, 1.5, 3, np.inf]
            labels = ["through level", "0-0.5 ATR", "0.5-1.5 ATR", "1.5-3 ATR", ">3 ATR"]
            rb = rd.dropna(subset=["dist_atr"]).copy()
            if rb.empty:
                st.info("No anchor-level distances available for this selection.")
            else:
                rb["bucket"] = pd.cut(rb["dist_atr"], bins=bins, labels=labels)
                pv = rb.pivot_table(index="regime", columns="bucket", values="det_bps", aggfunc="mean", observed=False)
                pv.index = [reg_names[i] if i < len(reg_names) else f"R{i}" for i in pv.index]
                fig = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns.astype(str)), y=pv.index,
                                           colorscale=[[0, "#1f3b57"], [0.5, "#3a4a5a"], [1, "#f0883e"]],
                                           text=np.round(pv.values, 2), texttemplate="%{text}",
                                           colorbar=dict(title="bps")))
                show(style_fig(fig, 320, f"Deterioration by distance to L{anchor_level} {'support' if side == 1 else 'resistance'}"))

        st.download_button("Download simulated reloads (CSV)", res.to_csv(index=False).encode(), "reload_samples.csv", "text/csv")
        st.caption("Regime is read at the moment the previous child was placed, so there is no look-ahead. "
                   "Regime cut-offs use full-sample quantiles, so they describe the tape rather than forecast it.")

# --------------------------------------------------------------------------
# Method & guide
# --------------------------------------------------------------------------
with tab_guide:
    with st.expander("1. Multiscale structure", expanded=True):
        st.markdown(
            "- **Directional change.** A swing high is confirmed once price falls `mult x ATR` from the running maximum, a swing low once it rises that far from the running minimum. Confirmation lags the swing, which is what a live system would see.\n"
            "- **Hierarchy.** Level 0 holds every directional-change swing. A swing is promoted to level k+1 when it is the most extreme among the neighbouring same-type swings, and an opposite-type swing is promoted with it so highs and lows keep alternating.\n"
            "- **Timeframes.** The same engine runs on 1, 5, 15, 30 and 60 minute bars. A higher-timeframe level only becomes usable on the 1-minute tape after that bar has closed."
        )
    with st.expander("2. Liquidity and price levels"):
        st.markdown(
            "Unswept swing highs usually carry resting buy stops and breakout orders above them, unswept swing lows carry sell stops below. Levels from every timeframe are pooled, weighted by `(level + 1) x sqrt(timeframe minutes)`, and merged into zones within the chosen tolerance. A volume profile is shown alongside as a rough view of where the order book has traded most."
        )
    with st.expander("3. Reload deterioration"):
        st.markdown(
            "A parent order is split into child orders placed every *interval* bars. For child k:\n\n"
            "`deterioration_k = side x (exec_k - exec_{k-1}) / exec_{k-1} x 10,000`  (bps, positive = worse)\n\n"
            "It is split into **market drift** (move in the raw fill price) and **own impact** (the model component). "
            "**Aggressive** children cross the spread at the next open. **Passive at level** children rest at the anchor-level swing and chase across the spread if not filled within the patience window. "
            "Impact follows a square-root law, `Y x daily vol x sqrt(child size / ADV)`, with a permanent part that accumulates and a temporary part that decays with the chosen half-life."
        )
    with st.expander("4. Execution concepts used here"):
        st.markdown(
            "- **Transaction cost analysis (TCA).** Measuring execution price against benchmarks such as arrival price, which is what slippage vs arrival reports.\n"
            "- **Implementation shortfall.** The gap between the decision price and the final average fill, including delay, spread and impact.\n"
            "- **Information leakage.** Regular reloads at the same level show the parent order to other participants, who can trade ahead of the next child. Shorter intervals and visible levels raise the risk.\n"
            "- **Market impact, temporary and permanent.** Temporary impact fades after you stop trading, permanent impact stays in the price. The cost model keeps them separate.\n"
            "- **ADV.** Average daily volume. Parent and child sizes are expressed as a share of it.\n"
            "- **Open and close auctions vs continuous trading.** The first and last bar of each day are tagged as auction proxies. Auction prints concentrate volume, so structure formed there is less reliable for placing continuous-session reloads.\n"
            "- **Order book.** Passive fills assume you trade only when price trades through your level by the chosen number of basis points, a rough stand-in for queue position.\n"
            "- **Liquidity events.** Sweeps of swing highs or lows, shown as swept levels dropping out of the map, mark where resting liquidity was taken."
        )
    with st.expander("5. Assumptions and limits"):
        st.markdown(
            "- Fills use bar data only: no queue, no hidden liquidity, no intrabar path. Passive fills are optimistic.\n"
            "- Overlapping samples are correlated, so standard errors are understated. Raise the sampling stride for cleaner statistics.\n"
            "- The synthetic tape has Markov volatility regimes by construction, so regime effects are cleaner than on real data.\n"
            "- yfinance 1-minute history is short (a few days), which limits the higher levels and the 60-minute timeframe."
        )

st.markdown(
    '<div class="foot">Structure engine ported from <a href="https://github.com/neurotrader888/market-structure">neurotrader888/market-structure</a> '
    "(MIT licence). For research and education only, not investment advice.</div>",
    unsafe_allow_html=True,
)
