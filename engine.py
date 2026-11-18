"""
engine.py - multiscale market structure + child-order reload deterioration.

Pure numpy / pandas (no Streamlit, no Plotly) so it can be tested on its own.

Credit: the ATR directional-change and hierarchical-extremes algorithms are a
port of neurotrader888/market-structure (MIT licence). The liquidity-zone
clustering, volatility-regime labelling and reload-deterioration simulator are
additions built on top of it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIMEFRAMES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "60min": 60}
REGIME_NAMES = {
    2: ["Low", "High"],
    3: ["Low", "Normal", "High"],
    4: ["Low", "Normal", "Elevated", "Stress"],
}


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def clean_ohlcv(df: pd.DataFrame, max_bars: int = 60_000) -> pd.DataFrame:
    """Normalise column names / index and keep the most recent `max_bars`."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("timestamp", "datetime", "date", "time"):
            if cand in df.columns:
                df.index = pd.to_datetime(df[cand])
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Need a datetime index or a timestamp/datetime/date column.")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df[need + ["volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    return df.iloc[-max_bars:]


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return df.copy()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(f"{minutes}min", origin="start").agg(agg)
    return out.dropna(subset=["open"])


def tag_sessions(df: pd.DataFrame) -> pd.Series:
    """First / last bar of each calendar day = opening / closing auction proxy."""
    day = pd.Series(df.index.normalize(), index=df.index)
    first = day.ne(day.shift(1))
    last = day.ne(day.shift(-1))
    out = pd.Series("continuous", index=df.index)
    out[first] = "open auction"
    out[last] = "close auction"
    return out


def synthetic_ohlcv(
    days: int = 15,
    bars_per_day: int = 390,
    seed: int = 7,
    start: str = "2026-09-01",
    base_vol_bps: float = 3.5,
    regime_stay: float = 0.997,
    adv_shares: float = 5_000_000,
) -> pd.DataFrame:
    """1-minute bars with Markov volatility regimes, U-shaped intraday
    volatility/volume, fat tails, slow drift, overnight gaps and auction volume."""
    rng = np.random.default_rng(seed)
    n = days * bars_per_day
    vol_levels = np.array([0.6, 1.0, 2.6])
    state = np.empty(n, dtype=int)
    s = 1
    for i in range(n):
        if rng.random() > regime_stay:
            if s == 1:
                s = 0 if rng.random() < 0.5 else 2
            else:
                s = 1
        state[i] = s
    m = np.tile(np.arange(bars_per_day), days)
    u_vol = 1 + 0.8 * np.exp(-m / 30) + 0.6 * np.exp(-(bars_per_day - 1 - m) / 30)
    sigma = base_vol_bps * 1e-4 * vol_levels[state] * u_vol
    shocks = rng.standard_t(5, n) / np.sqrt(5 / 3)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.995 * drift[i - 1] + rng.normal(0, 0.012) * sigma[i]
    r = sigma * shocks + drift
    gap = np.zeros(n)
    gap[::bars_per_day] = rng.normal(0, 25e-4, days)
    days_idx = pd.bdate_range(start, periods=days)
    times = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=9, minutes=30 + k) for d in days_idx for k in range(bars_per_day)]
    )
    price0 = 100.0
    close = np.empty(n)
    open_ = np.empty(n)
    prev = price0
    for i in range(n):
        open_[i] = prev * np.exp(gap[i])
        close[i] = open_[i] * np.exp(r[i])
        prev = close[i]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1, n)) * sigma * 0.6)
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1, n)) * sigma * 0.6)
    u_volu = 1 + 1.2 * np.exp(-m / 25) + 1.0 * np.exp(-(bars_per_day - 1 - m) / 25)
    vol = (adv_shares / bars_per_day) * u_volu * rng.lognormal(0, 0.4, n) * vol_levels[state] ** 0.6
    vol[m == 0] *= 7
    vol[m == bars_per_day - 1] *= 5
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=times
    )


# --------------------------------------------------------------------------
# Market structure (port of neurotrader888/market-structure)
# --------------------------------------------------------------------------
@dataclass
class LocalExtreme:
    ext_type: int  # +1 top, -1 bottom
    index: int
    price: float
    timestamp: pd.Timestamp
    conf_index: int
    conf_price: float
    conf_timestamp: pd.Timestamp


def atr_series(h: np.ndarray, l: np.ndarray, c: np.ndarray, lb: int) -> np.ndarray:
    """Simple rolling-mean true range; first valid value at i == lb (as in the repo)."""
    prev = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    atr = pd.Series(tr).rolling(lb).sum().to_numpy() / lb
    atr[:lb] = np.nan
    return atr


class ATRDirectionalChange:
    """Confirms a swing high/low once price reverses by `mult` x ATR from the
    pending extreme."""

    def __init__(self, atr: np.ndarray, mult: float = 1.0):
        self.atr = atr
        self.mult = mult
        self.up_move = True
        self.pend_max = np.nan
        self.pend_min = np.nan
        self.pend_max_i = 0
        self.pend_min_i = 0
        self.extremes: list[LocalExtreme] = []

    def _create(self, kind, ext_i, conf_i, idx, h, l, c):
        t, arr = (1, h) if kind == "high" else (-1, l)
        self.extremes.append(
            LocalExtreme(t, ext_i, float(arr[ext_i]), idx[ext_i], conf_i, float(c[conf_i]), idx[conf_i])
        )

    def update(self, i, idx, h, l, c):
        a = self.atr[i]
        if np.isnan(a):
            return
        thr = a * self.mult
        if np.isnan(self.pend_max):
            self.pend_max, self.pend_min = h[i], l[i]
            self.pend_max_i = self.pend_min_i = i
        if self.up_move:
            if h[i] > self.pend_max:
                self.pend_max, self.pend_max_i = h[i], i
            elif l[i] < self.pend_max - thr:
                self._create("high", self.pend_max_i, i, idx, h, l, c)
                self.up_move = False
                self.pend_min, self.pend_min_i = l[i], i
        else:
            if l[i] < self.pend_min:
                self.pend_min, self.pend_min_i = l[i], i
            elif h[i] > self.pend_min + thr:
                self._create("low", self.pend_min_i, i, idx, h, l, c)
                self.up_move = True
                self.pend_max, self.pend_max_i = h[i], i


class HierarchicalExtremes:
    """Level 0 = ATR directional-change swings. A level-k swing is promoted to
    level k+1 when it is the most extreme of the surrounding same-type swings,
    keeping highs and lows alternating at every level."""

    def __init__(self, levels: int, atr: np.ndarray, mult: float = 1.0):
        self._dc = ATRDirectionalChange(atr, mult)
        self._levels = levels
        self.extremes: list[list[LocalExtreme]] = [[] for _ in range(levels)]

    @staticmethod
    def _cmp(x, y, ext_type):
        return x > y if ext_type == 1 else x < y

    def _new_ext(self, level, conf_i, conf_price, conf_time, ext_type):
        if level >= self._levels - 1:
            return
        ext_i = len(self.extremes[level]) - 1
        new_ext = self.extremes[level][ext_i]
        if ext_i < 4:
            return
        prev_ext = self.extremes[level][ext_i - 2]
        if not self._cmp(prev_ext.price, new_ext.price, ext_type):
            return
        prev_next = None
        if len(self.extremes[level + 1]) > 0:
            prev_next = self.extremes[level + 1][-1]
            if prev_next.ext_type != ext_type:
                if not self._cmp(prev_ext.price, prev_next.price, ext_type):
                    return
        for prior_i in range(ext_i - 4, -1, -2):
            prior = self.extremes[level][prior_i]
            if self._cmp(prior.price, prev_ext.price, ext_type):
                return
            if prev_next is not None and prior.index <= prev_next.index:
                break
            elif prior.price == prev_ext.price:
                prev_ext = prior
            elif self._cmp(prior.price, prev_ext.price, -ext_type):
                break
        promoted = copy.copy(prev_ext)
        promoted.conf_index, promoted.conf_price, promoted.conf_timestamp = conf_i, conf_price, conf_time
        if prev_next is not None and prev_next.ext_type == ext_type:
            upgrade = None
            for j in range(ext_i - 1, -1, -2):
                prior = self.extremes[level][j]
                if prior.index >= promoted.index:
                    continue
                if prior.index <= prev_next.index:
                    break
                if upgrade is None or not self._cmp(prior.price, upgrade.price, ext_type):
                    upgrade = prior
            if upgrade is None:  # defensive: cannot keep alternation, skip promotion
                return
            up = copy.copy(upgrade)
            up.conf_index, up.conf_price, up.conf_timestamp = conf_i, conf_price, conf_time
            self.extremes[level + 1].append(up)
            self._new_ext(level + 1, conf_i, conf_price, conf_time, -ext_type)
        self.extremes[level + 1].append(promoted)
        self._new_ext(level + 1, conf_i, conf_price, conf_time, ext_type)

    def update(self, i, idx, h, l, c):
        before = len(self._dc.extremes)
        self._dc.update(i, idx, h, l, c)
        if len(self._dc.extremes) > before:
            e = self._dc.extremes[-1]
            self.extremes[0].append(e)
            self._new_ext(0, i, float(c[i]), idx[i], e.ext_type)

    def _level_extreme(self, level, ext_type, lag=0):
        lst = self.extremes[level]
        if not lst:
            return None
        offset = 0 if lst[-1].ext_type == ext_type else 1
        k = lag * 2 + offset
        if k >= len(lst):
            return None
        return lst[-(k + 1)]

    def level_price(self, level, ext_type, lag=0) -> float:
        e = self._level_extreme(level, ext_type, lag)
        return np.nan if e is None else e.price


@dataclass
class Structure:
    minutes: int
    index: pd.DatetimeIndex
    atr: np.ndarray
    hi: np.ndarray  # (n, levels) latest confirmed level high, NaN if none
    lo: np.ndarray
    ext: list = field(default_factory=list)  # list of DataFrames, one per level


def run_structure(df: pd.DataFrame, minutes: int, levels: int, atr_lb: int, mult: float) -> Structure:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    atr = atr_series(h, l, c, atr_lb) if n > atr_lb + 2 else np.full(n, np.nan)
    he = HierarchicalExtremes(levels, atr, mult)
    hi = np.full((n, levels), np.nan)
    lo = np.full((n, levels), np.nan)
    idx = df.index
    last_count = -1
    for i in range(n):
        he.update(i, idx, h, l, c)
        cnt = sum(len(x) for x in he.extremes)
        if cnt != last_count:
            for lv in range(levels):
                hi[i, lv] = he.level_price(lv, 1)
                lo[i, lv] = he.level_price(lv, -1)
            last_count = cnt
        elif i > 0:
            hi[i], lo[i] = hi[i - 1], lo[i - 1]
    frames = []
    for lv in range(levels):
        rows = he.extremes[lv]
        frames.append(
            pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=["ext_type", "index", "price", "timestamp", "conf_index", "conf_price", "conf_timestamp"]
            )
        )
    return Structure(minutes, idx, atr, hi, lo, frames)


def extremes_sanity_checks(ext: pd.DataFrame) -> dict:
    """Alternation, ordering and confirmation-lag checks for one level."""
    if len(ext) < 2:
        return {"alternating": True, "ordered": True, "conf_after": True}
    t = ext["ext_type"].to_numpy()
    return {
        "alternating": bool(np.all(t[1:] != t[:-1])),
        "ordered": bool(np.all(np.diff(ext["index"].to_numpy()) >= 0)),
        "conf_after": bool(np.all(ext["conf_index"].to_numpy() >= ext["index"].to_numpy())),
    }


def level_stats(st: Structure) -> pd.DataFrame:
    rows = []
    for lv, e in enumerate(st.ext):
        if len(e) < 2:
            rows.append({"level": lv, "swings": len(e), "median swing (bps)": np.nan, "median duration (bars)": np.nan})
            continue
        p = e["price"].to_numpy()
        swing = np.abs(np.diff(p)) / p[:-1] * 1e4
        dur = np.diff(e["index"].to_numpy())
        rows.append(
            {
                "level": lv,
                "swings": len(e),
                "median swing (bps)": float(np.median(swing)),
                "median duration (bars)": float(np.median(dur)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Volatility regimes
# --------------------------------------------------------------------------
def regime_arrays(df: pd.DataFrame, kind: str, n_regimes: int, lb: int):
    """Returns (codes int array, -1 = undefined; names; sigma_day_bps array)."""
    c = df["close"].to_numpy()
    ret = pd.Series(np.log(c)).diff()
    rv = ret.rolling(lb).std().to_numpy()
    bars_per_day = max(1.0, df.groupby(df.index.normalize()).size().median())
    sigma_day_bps = rv * np.sqrt(bars_per_day) * 1e4
    if kind == "ATR %":
        a = atr_series(df["high"].to_numpy(), df["low"].to_numpy(), c, lb)
        x = a / c * 1e4
    else:
        x = rv * 1e4
    xs = pd.Series(x)
    codes = pd.qcut(xs, n_regimes, labels=False, duplicates="drop")
    out = np.where(codes.isna(), -1, codes.fillna(-1)).astype(int)
    k = int(out.max()) + 1 if (out >= 0).any() else 0
    names = REGIME_NAMES.get(n_regimes, [f"R{i}" for i in range(n_regimes)])[:k]
    return out, names, sigma_day_bps


# --------------------------------------------------------------------------
# Liquidity map
# --------------------------------------------------------------------------
def collect_levels(structs: dict, base: pd.DataFrame, levels_sel, only_unswept: bool) -> pd.DataFrame:
    h, l = base["high"].to_numpy(), base["low"].to_numpy()
    suf_max = np.concatenate((np.maximum.accumulate(h[::-1])[::-1], [-np.inf]))
    suf_min = np.concatenate((np.minimum.accumulate(l[::-1])[::-1], [np.inf]))
    frames = []
    for tf, st in structs.items():
        for lv in levels_sel:
            if lv >= len(st.ext) or len(st.ext[lv]) == 0:
                continue
            e = st.ext[lv].copy()
            e["tf"], e["tf_min"], e["level"] = tf, st.minutes, lv
            frames.append(e)
    if not frames:
        return pd.DataFrame()
    lv = pd.concat(frames, ignore_index=True)
    end = lv["timestamp"] + pd.to_timedelta(lv["tf_min"], unit="min")
    pos = base.index.searchsorted(end.to_numpy(), side="left")
    swept = np.where(
        lv["ext_type"].to_numpy() == 1,
        suf_max[pos] > lv["price"].to_numpy(),
        suf_min[pos] < lv["price"].to_numpy(),
    )
    lv["swept"] = swept
    if only_unswept:
        lv = lv[~lv["swept"]]
    return lv.reset_index(drop=True)


def cluster_levels(lv: pd.DataFrame, tol_bps: float, ref_price: float) -> pd.DataFrame:
    if lv.empty:
        return pd.DataFrame()
    lv = lv.sort_values("price").reset_index(drop=True)
    lv["w"] = (lv["level"] + 1) * np.sqrt(lv["tf_min"])
    tol = ref_price * tol_bps / 1e4
    grp = (lv["price"].diff() > tol).cumsum()
    out = []
    for _, g in lv.groupby(grp):
        px = float(np.average(g["price"], weights=g["w"]))
        share_hi = float((g["ext_type"] == 1).mean())
        side = "Buy-side (stops above)" if share_hi > 0.7 else "Sell-side (stops below)" if share_hi < 0.3 else "Mixed"
        out.append(
            {
                "price": px,
                "score": float(g["w"].sum()),
                "swings": len(g),
                "max level": int(g["level"].max()),
                "timeframes": ", ".join(sorted(g["tf"].unique(), key=lambda s: TIMEFRAMES[s])),
                "side": side,
                "last confirmed": g["conf_timestamp"].max(),
            }
        )
    z = pd.DataFrame(out)
    z["dist (bps)"] = (z["price"] - ref_price) / ref_price * 1e4
    z["position"] = np.where(z["price"] >= ref_price, "Above", "Below")
    return z.sort_values("score", ascending=False).reset_index(drop=True)


def volume_profile(df: pd.DataFrame, bins: int = 60):
    lo, hi = df["low"].min(), df["high"].max()
    edges = np.linspace(lo, hi, bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol, _ = np.histogram(tp, bins=edges, weights=df["volume"])
    return mid, vol


# --------------------------------------------------------------------------
# Child-order reload simulator
# --------------------------------------------------------------------------
def map_to_base(values: np.ndarray, st: Structure, base_index: pd.DatetimeIndex) -> np.ndarray:
    """Forward-fill a higher-timeframe series onto base bars, available only
    once the higher-timeframe bar has closed (no look-ahead)."""
    end = st.index + pd.Timedelta(minutes=st.minutes)
    ser = pd.Series(values, index=end)
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser.reindex(base_index, method="ffill").to_numpy()


def prepare_sim_arrays(base, regimes, sigma_day_bps, anchor: Structure, level: int) -> dict:
    return {
        "open": base["open"].to_numpy(),
        "high": base["high"].to_numpy(),
        "low": base["low"].to_numpy(),
        "close": base["close"].to_numpy(),
        "index": base.index,
        "regime": regimes,
        "sigma": sigma_day_bps,
        "lvl_hi": map_to_base(anchor.hi[:, level], anchor, base.index),
        "lvl_lo": map_to_base(anchor.lo[:, level], anchor, base.index),
        "anchor_atr": map_to_base(anchor.atr, anchor, base.index),
    }


def simulate_reloads(
    a: dict,
    side: int,
    n_children: int,
    interval: int,
    stride: int,
    style: str,
    patience: int,
    half_spread_bps: float,
    through_bps: float,
    parent_pct_adv: float,
    impact_y: float,
    perm_share: float,
    halflife_bars: float,
) -> pd.DataFrame:
    """A parent order is sliced into `n_children`; child k is placed every
    `interval` bars. Deterioration of reload k = adverse move in execution price
    versus child k-1 (bps). `side` = +1 buy, -1 sell.

    style 'Aggressive' : cross the spread at the next open.
    style 'Passive at level': rest at the anchor-level swing (support for buys,
        resistance for sells) for `patience` bars, then chase across the spread.
    """
    o, h, l, c = a["open"], a["high"], a["low"], a["close"]
    reg, sig, lh, ll, aatr = a["regime"], a["sigma"], a["lvl_hi"], a["lvl_lo"], a["anchor_atr"]
    idx = a["index"]
    N = len(c)
    pat = patience if style == "Passive at level" else 1
    horizon = (n_children - 1) * interval + pat + 3
    q = parent_pct_adv / 100.0 / n_children
    decay = 0.5 ** (interval / halflife_bars) if halflife_bars > 0 else 0.0
    hs = half_spread_bps / 1e4
    rows = []
    sid = 0
    for s in range(0, N - horizon, stride):
        if reg[s] < 0 or np.isnan(sig[s]):
            continue
        sid += 1
        arrival = c[s]
        raw = np.zeros(n_children)
        exe = np.zeros(n_children)
        imp = np.zeros(n_children)
        placed = np.zeros(n_children, dtype=int)
        wait = np.zeros(n_children)
        chased = np.zeros(n_children)
        perm, resid = 0.0, 0.0
        for k in range(n_children):
            p = s + k * interval
            placed[k] = p
            if style == "Passive at level":
                if side == 1:
                    lim = min(ll[p], c[p]) if np.isfinite(ll[p]) else c[p]
                    hit = l[p + 1: p + 1 + pat] <= lim * (1 - through_bps / 1e4)
                else:
                    lim = max(lh[p], c[p]) if np.isfinite(lh[p]) else c[p]
                    hit = h[p + 1: p + 1 + pat] >= lim * (1 + through_bps / 1e4)
                if hit.any():
                    j = p + 1 + int(np.argmax(hit))
                    raw[k] = lim
                    wait[k] = j - p
                else:
                    j = p + pat + 1
                    raw[k] = o[j] * (1 + side * hs)
                    wait[k] = j - p
                    chased[k] = 1
            else:
                j = p + 1
                raw[k] = o[j] * (1 + side * hs)
                wait[k] = 1
            big_i = impact_y * sig[p] * np.sqrt(q)
            temp = (1 - perm_share) * big_i
            perm += perm_share * big_i
            imp[k] = perm + temp + resid
            resid = decay * (resid + temp)
            exe[k] = raw[k] * (1 + side * imp[k] / 1e4)
        for k in range(n_children):
            rec = {
                "sample": sid,
                "k": k,
                "interval": interval,
                "time": idx[placed[k]],
                "regime_start": int(reg[s]),
                "slip_bps": side * (exe[k] - arrival) / arrival * 1e4,
                "chased": chased[k],
                "wait": wait[k],
            }
            if k == 0:
                rec.update(regime=int(reg[s]), det_bps=np.nan, mkt_bps=np.nan, imp_bps=np.nan,
                           dist_atr=np.nan, struct_break=np.nan)
            else:
                pp = placed[k - 1]
                det = side * (exe[k] - exe[k - 1]) / exe[k - 1] * 1e4
                mkt = side * (raw[k] - raw[k - 1]) / raw[k - 1] * 1e4
                if side == 1:
                    dist = (c[pp] - ll[pp]) / aatr[pp] if np.isfinite(ll[pp]) and aatr[pp] > 0 else np.nan
                    brk = float(h[pp + 1: placed[k] + 1].max() > lh[pp]) if np.isfinite(lh[pp]) else np.nan
                else:
                    dist = (lh[pp] - c[pp]) / aatr[pp] if np.isfinite(lh[pp]) and aatr[pp] > 0 else np.nan
                    brk = float(l[pp + 1: placed[k] + 1].min() < ll[pp]) if np.isfinite(ll[pp]) else np.nan
                rec.update(regime=int(reg[pp]), det_bps=det, mkt_bps=mkt, imp_bps=det - mkt,
                           dist_atr=dist, struct_break=brk)
            rows.append(rec)
    return pd.DataFrame(rows)


def regime_summary(res: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    r = res[(res["k"] >= 1) & (res["regime"] >= 0)]
    if r.empty:
        return pd.DataFrame()
    g = r.groupby("regime")
    out = pd.DataFrame(
        {
            "reloads": g.size(),
            "mean det (bps)": g["det_bps"].mean(),
            "median (bps)": g["det_bps"].median(),
            "p90 (bps)": g["det_bps"].quantile(0.9),
            "adverse %": g["det_bps"].apply(lambda s: (s > 0).mean() * 100),
            "market (bps)": g["mkt_bps"].mean(),
            "own impact (bps)": g["imp_bps"].mean(),
            "chased %": g["chased"].mean() * 100,
            "structure break %": g["struct_break"].mean() * 100,
            "std err (bps)": g["det_bps"].apply(lambda s: s.std(ddof=1) / np.sqrt(max(len(s), 1))),
        }
    )
    out.index = [names[i] if i < len(names) else f"R{i}" for i in out.index]
    out.index.name = "regime"
    return out
