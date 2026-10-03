"""
Build Synthetic EMXC.py

Builds a daily total-return proxy for MSCI Emerging Markets ex China, priced
at the US close like an ETF, so the backtest engine can backfill EMXC
(launched July 26, 2017) with ex-China history instead of EEM. Before October
2026, the engine's EMXC history before launch was EEM, which holds China.

METHOD
    Emerging markets = China + everything else, so on each day

        r_exChina = (r_EEM - w * r_China) / (1 - w)  +  fee adjustment / 252

    r_EEM      iShares MSCI Emerging Markets (EEM), Yahoo adjusted close.
    r_China    MCHI (iShares MSCI China, from 2011-03), return-spliced to GXC
               (SPDR S&P China, from 2007-03) and FXI (iShares China Large-Cap,
               from 2004-10). GXC and FXI track different indexes from MSCI
               China; the checks below measure what that costs.
    w          China's weight in MSCI EM at the start of the day.
    fee        The subtraction inherits (f_EEM - w f_China) / (1 - w) of fund
               fees; EMXC charges 0.25%. The adjustment swaps one for the
               other: (f_EEM - w f_China) / (1 - w) - f_EMXC, about 0.46%/yr.
               Current expense ratios are used for every year.

    China's weight. A cap-weighted index's China weight drifts with China's
    return relative to the whole index and steps when the index adds or drops
    securities. The model starts from each published weight (ANCHORS), drifts
    it with MCHI/GXC/FXI relative to EEM, and adds a log-odds adjustment that
    grows evenly through the gap so the path lands on the next published
    weight. That adjustment absorbs composition changes and any drift error
    from using ETF total returns; it is a smoothing convention, not a
    reconstruction of index events. Before the first anchor the weight drifts
    backward from it; after the last it drifts forward.

    The first version of these anchors (October 3, 2026) used three sparse
    citations and two assumptions. An outside review (ChatGPT through Codex,
    October 3, 2026) found dated benchmark weights for 2004-2010, verified
    against the source documents and used below.

CHECKS (written to "Synthetic EMXC Validation Stats.txt")
    1. Calendar-year (EM ex China minus EM) spreads against MSCI's published
       net USD returns, 2012-2016, using a zero-fee-adjustment version so the
       comparison is exact. 2010 and 2011 are shown separately as reported by
       the reviewer from an older MSCI factsheet that could not be re-opened.
    2. Absolute tracking: proxy against the published EM ex China return, with
       EEM against MSCI EM and real EMXC against its index for scale.
    3. Leave-one-out: drop each published weight in turn and see how close the
       model gets to it from the others.
    4. Post-launch overlap against real EMXC, with the China sleeve built from
       MCHI, from GXC alone and from FXI alone. Anchored to 2020-09-30; after
       that the weight only drifts, so later years are a looser test.
    5. Daily volatility: trailing 12-month daily vol of proxy against EMXC at
       every month-end, because the engine weights by daily vol.

OUTPUT
    EMXC SYNTH Daily Prices.csv         column EMXC_SYNTH, US trading days,
                                        2004-10-08 to 2017-12-29. The engine
                                        uses it only before EMXC's first close.
    Synthetic EMXC Components.csv       ingredients, weight paths and every
                                        sensitivity series through 2026-08
                                        (deliberately not named
                                        "<TICKER> Daily Prices.csv")
    Synthetic EMXC Validation Stats.txt

Engine backfill chain for EMXC: EMXC_SYNTH, EEM, VEIEX.
Checked in the strategy by "EMXC Backfill Check.py".

Run from the Backtest Engine folder:
    python -u "Build Synthetic EMXC.py"
"""
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

OUT_CSV = "EMXC SYNTH Daily Prices.csv"
OUT_COMP = "Synthetic EMXC Components.csv"
OUT_STATS = "Synthetic EMXC Validation Stats.txt"
SYNTH_START = "2004-10-08"
SYNTH_END = "2017-12-29"
COMP_END = "2026-08-31"
EMXC_START = "2017-07-26"
ANCHORED_END = "2020-09-30"
SHIFT = 0.04

# Current expense ratios, annual
F_EEM, F_EMXC = 0.0070, 0.0025
F_CHINA = {"MCHI": 0.0059, "GXC": 0.0059, "FXI": 0.0074}

# China's weight in MSCI EM: (date, weight, source). All published observations.
ANCHORS = [
    ("2004-12-31", 0.077, "Russell Research (2012), Table 4, MSCI EM country weights, Dec 2004"),
    ("2006-12-29", 0.118, "Russell Research (2012), Table 4, Dec 2006"),
    ("2007-12-31", 0.159, "Harding Loevner EM report Q4 2007, MSCI EM benchmark column"),
    ("2008-03-31", 0.139, "Harding Loevner EM report Q1 2008 (also Callan 2018, '13.9% in 2008')"),
    ("2009-12-31", 0.179, "Harding Loevner EM report Q4 2009"),
    ("2010-12-31", 0.173, "Harding Loevner EM report Q4 2010 (KraneShares 2020: 17%)"),
    ("2014-04-30", 0.170, "KraneShares (2020): 17% in 2014, April 2014 context"),
    ("2018-03-29", 0.299, "Morningstar: 29.9% as of March 2018"),
    ("2020-09-30", 0.410, "KraneShares (2020): 41% as of Sept 30, 2020"),
]
# MSCI EM ex China (USD, net) factsheet as of 2026-08-31: calendar years, %
MSCI_OFFICIAL = {  # year: (EM ex China, EM)
    2012: (17.24, 18.22), 2013: (-4.02, -2.60), 2014: (-4.62, -2.19),
    2015: (-16.89, -14.92), 2016: (15.02, 11.19), 2017: (31.25, 37.28),
    2018: (-12.42, -14.57), 2019: (16.21, 18.42), 2020: (12.55, 18.31),
    2021: (10.03, -2.54), 2022: (-19.26, -20.09), 2023: (20.03, 9.83),
    2024: (3.56, 7.50), 2025: (34.61, 33.57),
}
# Reported by the reviewer from an older MSCI factsheet reproduced in a New
# Mexico PERA packet; the link returned 404 when re-checked. The EM figures
# match MSCI EM's published net returns for those years.
MSCI_REPORTED = {2010: (22.28, 18.88), 2011: (-18.44, -18.42)}

_lines = []


def log(s=""):
    print(s)
    _lines.append(str(s))


# ---------------------------------------------------------------- data
TICKERS = ["EEM", "MCHI", "GXC", "FXI", "EMXC", "SPY"]
print("Downloading from Yahoo ...")
raw = yf.download(TICKERS, start="2003-01-01", auto_adjust=True, progress=False)["Close"]
for t in TICKERS:
    if t not in raw.columns or raw[t].dropna().empty:
        sys.exit(f"ERROR: {t} did not download.")
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)


def splice(primary, chain):
    """Return-splice; also return which fund supplies each date."""
    out = px[primary].copy()
    src = pd.Series(np.where(out.notna(), primary, None), index=out.index, dtype=object)
    for bf in chain:
        have = out.dropna()
        first = have.index.min()
        b = px[bf].dropna()
        b = b[b.index <= first]
        if len(b) < 2:
            continue
        scaled = b * (have.iloc[0] / b.iloc[-1])
        fill = scaled.index[scaled.index < first]
        out.loc[fill] = scaled.loc[fill]
        src.loc[fill] = bf
    return out, src


CHAINS = {"MCHI": ("MCHI", ["GXC", "FXI"]), "GXC": ("GXC", ["FXI"]), "FXI": ("FXI", [])}
china, china_src = {}, {}
for k, (p, ch) in CHAINS.items():
    china[k], china_src[k] = splice(p, ch)

idx = px.index[(px.index >= SYNTH_START) & (px.index <= COMP_END)]
eem = px["EEM"].reindex(idx)


def logit(w):
    return np.log(w / (1.0 - w))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def weight_path(cn, anchors=ANCHORS, shift=0.0):
    ts = [pd.Timestamp(a) for a, *_ in anchors]
    ws = [float(np.clip(a[1] + shift, 0.01, 0.95)) for a in anchors]
    rel = cn.reindex(idx) / eem
    w = pd.Series(np.nan, index=idx)
    head = idx[idx < ts[0]]
    if len(head):
        w.loc[head] = ws[0] * rel.loc[head] / rel.asof(ts[0])
    for i in range(len(ts) - 1):
        seg = idx[(idx >= ts[i]) & (idx <= ts[i + 1])]
        if len(seg) == 0:
            continue
        drift = ws[i] * rel.loc[seg] / rel.asof(ts[i])
        corr = logit(ws[i + 1]) - logit(drift.iloc[-1])
        frac = np.asarray((seg - ts[i]).days / (ts[i + 1] - ts[i]).days)
        w.loc[seg] = inv_logit(logit(drift) + corr * frac)
    tail = idx[idx > ts[-1]]
    if len(tail):
        w.loc[tail] = ws[-1] * rel.loc[tail] / rel.asof(ts[-1])
    return w.ffill().bfill()


def synth(k, w, fee=True):
    cn = china[k].reindex(idx)
    r_e, r_c = eem.pct_change(), cn.pct_change()
    wp = w.shift(1).fillna(w.iloc[0])
    r = (r_e - wp * r_c) / (1.0 - wp)
    if fee:
        fc = china_src[k].reindex(idx).map(F_CHINA).astype(float)
        r = r + ((F_EEM - wp * fc) / (1.0 - wp) - F_EMXC) / 252.0
    r.iloc[0] = 0.0
    return 100.0 * (1.0 + r.fillna(0.0)).cumprod()


W = {
    "base": weight_path(china["MCHI"]),
    "lo": weight_path(china["MCHI"], shift=-SHIFT),
    "hi": weight_path(china["MCHI"], shift=+SHIFT),
    "gxc": weight_path(china["GXC"]),
    "fxi": weight_path(china["FXI"]),
}
S = {
    "base": synth("MCHI", W["base"]),
    "lo": synth("MCHI", W["lo"]),
    "hi": synth("MCHI", W["hi"]),
    "gxc": synth("GXC", W["gxc"]),
    "fxi": synth("FXI", W["fxi"]),
}
S0 = {k: synth({"base": "MCHI", "lo": "MCHI", "hi": "MCHI", "gxc": "GXC", "fxi": "FXI"}[k], W[k], fee=False)
      for k in S}


def cal_year(p):
    y = p.dropna().resample("YE").last()
    out = y.pct_change() * 100
    out.index = out.index.year
    return out


eem_y = cal_year(px["EEM"])
emxc_y = cal_year(px["EMXC"])

log("Synthetic EMXC validation   " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
log("Proxy for MSCI EM ex China priced at the US close. 'base' = China sleeve MCHI <- GXC <- FXI.")
log("lo / hi = every China weight 4 points lower / higher. gxc / fxi = China sleeve from GXC")
log("(FXI before 2007-03) or FXI alone, each with its own weight drift.")
log("")
log("China sleeve segments (base): " + "; ".join(
    f"{s} {g.index.min().date()}->{g.index.max().date()}"
    for s, g in china_src["MCHI"].reindex(idx).dropna().groupby(china_src["MCHI"].reindex(idx).dropna(), sort=False)))
log("")
log("China weight anchors (published observations):")
for a, w, src in ANCHORS:
    log(f"  {a}  {w:6.1%}  {src}")
log("")
log("Modeled China weight at each year-end, %:")
ye = pd.DataFrame(W).resample("YE").last()
ye.index = ye.index.year
log((ye * 100).to_string(float_format=lambda v: f"{v:5.1f}"))

# ---- Check 1: spreads, exact (zero fee adjustment)
log("")
log("=" * 104)
log("Check 1. (EM ex China minus EM) calendar-year spread, points. Proxy uses no fee adjustment here,")
log("so 'proxy minus EEM' is directly comparable to the index spread. Miss = proxy minus official.")
log("=" * 104)
rows = []
for y in list(MSCI_REPORTED) + list(range(2012, 2017)):
    xo, eo = MSCI_REPORTED.get(y) or MSCI_OFFICIAL[y]
    row = {"Year": y, "Source": "reported" if y in MSCI_REPORTED else "factsheet", "Official": xo - eo}
    for k in S0:
        row[f"miss {k}"] = cal_year(S0[k]).get(y, np.nan) - eem_y.get(y, np.nan) - (xo - eo)
    rows.append(row)
c1 = pd.DataFrame(rows).set_index("Year")
log(c1.to_string(float_format=lambda v: f"{v:+.2f}"))
fs = c1[c1["Source"] == "factsheet"]
log("2012-2016 mean absolute miss:  " + "  ".join(f"{k} {fs[f'miss {k}'].abs().mean():.2f}" for k in S0))
log("2010-2016 mean absolute miss:  " + "  ".join(f"{k} {c1[f'miss {k}'].abs().mean():.2f}" for k in S0))

# ---- Check 2: absolute tracking
log("")
log("=" * 104)
log("Check 2. Absolute tracking, points a year. Proxy (with fee adjustment) minus the published EM ex")
log("China return; a perfect EMXC-like series would sit about -0.25 below (EMXC's fee). For scale:")
log("EEM minus MSCI EM, and real EMXC minus its index.")
log("=" * 104)
rows = []
for y in list(MSCI_REPORTED) + list(range(2012, 2026)):
    xo, eo = MSCI_REPORTED.get(y) or MSCI_OFFICIAL[y]
    rows.append({
        "Year": y,
        "Proxy - index": cal_year(S["base"]).get(y, np.nan) - xo if y <= 2016 else np.nan,
        "EEM - MSCI EM": eem_y.get(y, np.nan) - eo,
        "EMXC - index": emxc_y.get(y, np.nan) - xo if y >= 2018 else np.nan,
    })
c2 = pd.DataFrame(rows).set_index("Year")
log(c2.to_string(float_format=lambda v: f"{v:+.2f}"))
p = c2["Proxy - index"].dropna()
e = c2["EMXC - index"].dropna()
log(f"Proxy 2010-2016: mean {p.mean():+.2f}, mean absolute {p.abs().mean():.2f}, sd {p.std():.2f}")
log(f"Real EMXC 2018-2025: mean {e.mean():+.2f}, mean absolute {e.abs().mean():.2f}, sd {e.std():.2f}")

# ---- Check 3: leave-one-out on anchors
log("")
log("=" * 104)
log("Check 3. Leave-one-out: drop each interior anchor, rebuild, and compare the model's weight there")
log("=" * 104)
for i in range(1, len(ANCHORS) - 1):
    a, wv, _ = ANCHORS[i]
    wp = weight_path(china["MCHI"], anchors=ANCHORS[:i] + ANCHORS[i + 1:])
    log(f"  {a}  published {wv:6.1%}   modeled without it {wp.asof(pd.Timestamp(a)):6.1%}")

# ---- Check 4: post-launch overlap
log("")
log("=" * 104)
log("Check 4. Post-launch overlap against real EMXC. Anchored through 2020-09-30; drift only after.")
log("=" * 104)


def overlap_stats(a, start, end):
    win = idx[(idx >= start) & (idx <= end)]
    real = px["EMXC"].reindex(win)
    s = a.reindex(win)
    dr = pd.concat({"p": s.pct_change(), "r": real.pct_change()}, axis=1).dropna()
    mr = pd.concat({"p": s, "r": real}, axis=1).resample("ME").last().pct_change().dropna()
    yrs = (win[-1] - win[0]).days / 365.25
    return {
        "Monthly corr": mr.corr().iloc[0, 1],
        "Monthly TE": (mr["p"] - mr["r"]).std() * np.sqrt(12),
        "Ann diff": (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - (real.iloc[-1] / real.iloc[0]) ** (1 / yrs),
        "Daily corr": dr.corr().iloc[0, 1],
        "Proxy dvol": dr["p"].std() * np.sqrt(252),
        "EMXC dvol": dr["r"].std() * np.sqrt(252),
        "Proxy mvol": mr["p"].std() * np.sqrt(12),
        "EMXC mvol": mr["r"].std() * np.sqrt(12),
    }


for label, start, end in [("Anchored", EMXC_START, ANCHORED_END), ("Drift only", ANCHORED_END, COMP_END)]:
    t = pd.DataFrame({k: overlap_stats(S[k], start, end) for k in ("base", "gxc", "fxi")}).T
    log(f"\n{label}  {start} -> {end}")
    log(t.to_string(float_format=lambda v: f"{v:+.4f}"))
log("\nCalendar years, proxy minus EMXC, points:")
yrows = []
for y in range(2018, 2026):
    yrows.append({"Year": y, **{k: cal_year(S[k]).get(y) - emxc_y.get(y) for k in ("base", "gxc", "fxi")}})
log(pd.DataFrame(yrows).set_index("Year").to_string(float_format=lambda v: f"{v:+.2f}"))

# ---- Check 5: trailing daily vol ratio at month-ends
log("")
log("=" * 104)
log("Check 5. Trailing 12-month daily vol, EMXC / proxy, at each month-end (what the engine's")
log("inverse-volatility weights see). Below 1 means the proxy looks riskier than EMXC did.")
log("=" * 104)
dr_all = pd.concat({"p": S["base"].pct_change(), "r": px["EMXC"].reindex(idx).pct_change()}, axis=1)
mends = dr_all.resample("ME").last().index
ratios = {}
for me in mends:
    if me < pd.Timestamp("2018-07-31") or me > pd.Timestamp(COMP_END):
        continue
    sub = dr_all.loc[me - pd.DateOffset(months=12):me].dropna()
    ratios[me] = sub["r"].std() / sub["p"].std()
ratios = pd.Series(ratios)
for label, sub in [("Anchored 2018-07 -> 2020-09", ratios[:ANCHORED_END]),
                   ("Drift only 2020-10 -> 2026-08", ratios[ANCHORED_END:].iloc[1:]),
                   ("All", ratios)]:
    log(f"  {label:30s} median {sub.median():.3f}  10th pct {sub.quantile(.1):.3f}  "
        f"90th pct {sub.quantile(.9):.3f}  n={len(sub)}")
VOL_S = float(ratios[:ANCHORED_END].median())
log(f"Sensitivity series SYN_DENOISED scales each day's deviation from its month's average log")
log(f"return by {VOL_S:.3f} (the anchored-window median), so month-end levels are unchanged and only")
log("daily volatility falls. It is a test case, not the default.")

lr = np.log(S["base"]).diff().fillna(0.0)
mkey = lr.index.to_period("M")
cm = lr.groupby(mkey).transform("mean")
den = 100.0 * np.exp((cm + VOL_S * (lr - cm)).cumsum())

# ---- Check 6: what the backfill changes
log("")
log("=" * 104)
log("Check 6. Annualized return and daily vol, 2004-10-08 to 2017-07-25")
log("=" * 104)
pre = idx[(idx >= SYNTH_START) & (idx <= "2017-07-25")]
yrs = (pre[-1] - pre[0]).days / 365.25
for name, s in [("EEM (old backfill)", px["EEM"])] + [(f"proxy {k}", v) for k, v in S.items()] \
        + [("proxy denoised", den), ("China sleeve (base)", china["MCHI"])]:
    a = s.reindex(pre).dropna()
    log(f"  {name:22s} {((a.iloc[-1] / a.iloc[0]) ** (1 / yrs) - 1):+.2%} a year, "
        f"daily vol {a.pct_change().std() * np.sqrt(252):.1%}")

# ---------------------------------------------------------------- write
out = S["base"]
out = out[(out.index >= SYNTH_START) & (out.index <= SYNTH_END)]
pd.DataFrame({"EMXC_SYNTH": out}).rename_axis("Date").to_csv(OUT_CSV, float_format="%.6f")
comp = pd.DataFrame({
    "EEM": eem, "CHINA_MCHI": china["MCHI"].reindex(idx),
    "W_BASE": W["base"], "W_LO": W["lo"], "W_HI": W["hi"], "W_GXC": W["gxc"], "W_FXI": W["fxi"],
    "SYN_BASE": S["base"], "SYN_LO": S["lo"], "SYN_HI": S["hi"],
    "SYN_GXC": S["gxc"], "SYN_FXI": S["fxi"], "SYN_DENOISED": den,
}).rename_axis("Date")
comp.to_csv(OUT_COMP, float_format="%.6f")
with open(OUT_STATS, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
print(f"\nWrote {os.path.abspath(OUT_CSV)}  ({out.index.min().date()} -> {out.index.max().date()}, "
      f"{len(out)} rows)")
print(f"Wrote {os.path.abspath(OUT_COMP)}")
print(f"Wrote {os.path.abspath(OUT_STATS)}")
