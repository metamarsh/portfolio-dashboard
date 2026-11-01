"""
Build Synthetic MNK.py

Builds a daily total-return proxy for the CME Micro Dollar-Denominated Nikkei
225 future (MNK), held fully collateralized and rolled quarterly, so the
backtest engine can test it in DXJ's equity slot. MNK itself only has about a
year of history at IBKR, so the long history is built from its ingredients
and then checked against real traded futures.

WHAT MNK ACTUALLY EARNS
    A dollar-denominated ("quanto") Nikkei future has no yen exposure. Held
    and rolled, with the margin cash earning interest, it returns roughly:

        Nikkei 225 price return (in index points)
      + carry = dividend yield - yen short rate + quanto term
      + interest on the dollar collateral

    The quanto term is corr(Nikkei, USD per JPY) x vol(Nikkei) x vol(FX). It
    is negative when the Nikkei rises as the yen weakens, which has been the
    usual pattern, and it is why MNK trades at a premium to a yen-priced
    Nikkei future.

SOURCES
    NKD=F      Yahoo Finance, full-size CME dollar Nikkei future, front month,
               Chicago close -> daily price moves from Feb 2004 on. Same
               contract terms as MNK, same exchange, same close.
    ^N225      Yahoo Finance, Nikkei 225 index close -> daily price moves
               before Feb 2004, and cross-checks
    1321.T     Yahoo Finance, NEXT FUNDS Nikkei 225 ETF, trailing dividends
               -> Nikkei dividend yield, 2010 on. Yahoo split-adjusts its
               closes but left the dividends before the 100-for-1 split of
               October 5, 2026 in pre-split yen; split_adjusted_dividends()
               rescales them, and the build stops if the yields still look
               wrong (see "Dividend yield" below).
    EWJ        Yahoo Finance, iShares MSCI Japan, trailing distributions
               -> Japan dividend yield before 2010
    JPY=X      Yahoo Finance, USD/JPY -> FX vol and correlation
    SPY        Yahoo Finance, used only for the US trading calendar
    IRSTCI01JPM156N  FRED (OECD), Japan call money rate, monthly
    EFFR       Margin Rate Daily.csv (FRED DFF), dollar collateral rate

CHECKS (written to "Synthetic MNK Validation Stats.txt")
    1. Yahoo ^N225 month-end closes vs Interactive Brokers N225 closes
    2. Modeled carry vs carry measured from real MNK calendar spreads at the
       four rolls Dec 2025 - Sep 2026 (IBKR contract closes)
    3. Proxy vs real MNK, held and rolled, Sep 2025 - Sep 2026
    4. Yahoo NKD=F vs Yahoo ^N225 monthly, 2004 - 2026 (the splice point)
    5. Context: proxy vs DXJ vs EWJ annual returns

OUTPUT
    MNK SYNTH Daily Prices.csv     column MNK_SYNTH, US trading days
    Synthetic MNK Components.csv   every ingredient, for audit (deliberately
                                   not named "<TICKER> Daily Prices.csv")
    Synthetic MNK Validation Stats.txt
    Japan Call Rate Monthly.csv    FRED cache

Run from the Backtest Engine folder:
    python -u "Build Synthetic MNK.py"
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

OUT_CSV = "MNK SYNTH Daily Prices.csv"
COMPONENTS_CSV = "Synthetic MNK Components.csv"
STATS_TXT = "Synthetic MNK Validation Stats.txt"
JP_RATE_CACHE = "Japan Call Rate Monthly.csv"
RATE_CSV = "Margin Rate Daily.csv"
IBKR_MNK = "MNK Contract Closes IBKR.json"
IBKR_N225 = "N225 Monthly Closes IBKR.json"

START = "1990-01-01"
FRED_JP = "IRSTCI01JPM156N"
LAST_FULL_MONTH = "2026-08"

# The modeled carry ran 1.0%/yr richer than the carry priced into real MNK
# calendar spreads over the four rolls Dec 2025 - Sep 2026 (CHECK 2). The
# proxy is shifted by that amount so it matches what the contract actually
# charged. The basket test brackets this with +/-1%/yr.
CARRY_CALIBRATION = -0.010

# Used ONLY if FRED cannot be reached and there is no cache. Approximate Bank
# of Japan overnight call rate, percent, effective from the month shown.
BOJ_FALLBACK = [
    ("1990-01", 7.0), ("1991-07", 6.5), ("1992-01", 5.0), ("1993-01", 3.5),
    ("1994-01", 2.2), ("1995-04", 1.2), ("1995-09", 0.45), ("1998-09", 0.25),
    ("1999-02", 0.03), ("2000-08", 0.25), ("2001-03", 0.01), ("2006-07", 0.25),
    ("2007-02", 0.50), ("2008-10", 0.30), ("2008-12", 0.10), ("2016-02", -0.05),
    ("2024-03", 0.05), ("2024-08", 0.23), ("2025-01", 0.48), ("2025-12", 0.73),
]

_lines = []


def log(s=""):
    print(s)
    _lines.append(str(s))


def yf_hist(t):
    h = yf.Ticker(t).history(start=START, auto_adjust=False, actions=True)
    if h is None or h.empty:
        sys.exit(f"ERROR: Yahoo returned no data for {t}")
    idx = pd.to_datetime(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h.index = idx.normalize()
    return h[~h.index.duplicated(keep="last")].sort_index()


def fetch_fred(sid, timeout=150, tries=2):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (backtest research)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
            dcol = df.columns[0]
            vcol = df.columns[1]
            s = pd.to_numeric(df[vcol], errors="coerce")
            s.index = pd.to_datetime(df[dcol])
            return s.dropna() / 100.0
        except Exception as e:  # noqa: BLE001
            print(f"  FRED {sid} attempt {k + 1} failed: {e}")
    return None


OECD_JP_NAME = "Japan immediate call money rate, monthly (IRSTCI)"
OECD_JP_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
               "JPN.M.IRSTCI.PA.....?startPeriod=1990-01&format=csvfilewithlabels")


def fetch_oecd_jp(timeout=90):
    try:
        req = urllib.request.Request(
            OECD_JP_URL, headers={"User-Agent": "Mozilla/5.0 (backtest research)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        s = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        s.index = pd.to_datetime(df["TIME_PERIOD"] + "-01")
        return s.dropna().sort_index() / 100.0
    except Exception as e:  # noqa: BLE001
        print(f"  OECD fetch failed: {e}")
        return None


def split_adjusted_dividends(h, ticker):
    """Yahoo split-adjusts Close but not always Dividends. After 1321.T's
    100-for-1 split on October 5, 2026 it left every earlier dividend in
    pre-split yen, so 1321.T looked as if it yielded 115% to 185% a year
    (the October 30, 2026 build used that and was discarded), while the
    July 2026 dividend was already adjusted. A dividend paid before a split
    is divided by the later splits' combined ratio only when its yield on
    that day's close is impossible as reported (over 20%) and plausible once
    divided (under 10%), so dividends Yahoo did adjust are left alone."""
    div = h["Dividends"].copy()
    splits = h["Stock Splits"] if "Stock Splits" in h.columns else pd.Series(dtype=float)
    splits = splits[splits.fillna(0.0) > 0]
    if splits.empty:
        return div
    close = h["Close"].dropna()
    fixed = []
    for d, v in div[div > 0].items():
        later = splits[splits.index > d]
        c = close.loc[:d]
        if later.empty or c.empty or c.iloc[-1] <= 0:
            continue
        f = float(later.prod())
        if v / c.iloc[-1] > 0.20 and v / f / c.iloc[-1] < 0.10:
            div.loc[d] = v / f
            fixed.append(d)
    if fixed:
        log(f"{ticker}: {len(fixed)} dividends from before a stock split rescaled to the"
            f" split-adjusted price ({fixed[0].date()} to {fixed[-1].date()})")
    return div


def trailing_yield(ticker, expense_ratio):
    h = yf_hist(ticker)
    close = h["Close"].dropna()
    close = close[close > 0]
    div = split_adjusted_dividends(h, ticker).reindex(close.index).fillna(0.0)
    ttm = div.rolling("365D").sum()
    y = ttm / close + expense_ratio
    first_ok = close.index[0] + pd.Timedelta(days=366)
    return y.loc[first_ok:]


def ann_stats(r_a, r_b, periods):
    both = pd.concat([r_a, r_b], axis=1).dropna()
    d = both.iloc[:, 0] - both.iloc[:, 1]
    return {
        "n": len(both),
        "corr": both.iloc[:, 0].corr(both.iloc[:, 1]),
        "te": d.std() * np.sqrt(periods),
        "mean_diff": d.mean() * periods,
    }


# ---------------------------------------------------------------------------
# 1. Raw data
# ---------------------------------------------------------------------------
log("Synthetic MNK build   " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
log("=" * 78)
print("Downloading from Yahoo ...")
spy = yf_hist("SPY")["Close"].dropna()
cal = spy.index                                  # US trading days
n225 = yf_hist("^N225")["Close"].dropna()
n225 = n225[n225 > 0]
fx = yf_hist("JPY=X")["Close"].dropna()          # JPY per USD
fx = fx[fx > 0]
nkd = yf_hist("NKD=F")["Close"].dropna()
dxj = yf_hist("DXJ")["Adj Close"].dropna()
ewj = yf_hist("EWJ")["Adj Close"].dropna()

log(f"^N225   {n225.index.min().date()} -> {n225.index.max().date()}  {len(n225):,} closes")
log(f"JPY=X   {fx.index.min().date()} -> {fx.index.max().date()}  {len(fx):,} closes")
log(f"NKD=F   {nkd.index.min().date()} -> {nkd.index.max().date()}  {len(nkd):,} closes")

# Large one-day Nikkei moves, printed for inspection (not removed)
big = n225.pct_change()
big = big[big.abs() > 0.09]
log(f"\n^N225 one-day moves beyond 9% (inspect for bad prints): {len(big)}")
for d, v in big.items():
    log(f"    {d.date()}  {v:+.2%}")

# FX cleanup: drop prints more than 10% away from the 5-day rolling median
med = fx.rolling(5, center=True, min_periods=3).median()
bad_fx = (fx / med - 1).abs() > 0.10
log(f"JPY=X prints removed as bad (>10% from 5-day median): {int(bad_fx.sum())}")
fx = fx[~bad_fx]

# ---------------------------------------------------------------------------
# 2. Components
# ---------------------------------------------------------------------------
def to_cal(s):
    return s.reindex(s.index.union(cal)).ffill().reindex(cal)


n225_c = to_cal(n225)
n225_c = n225_c.loc[n225_c.index >= n225.index.min()]
nkd_c = to_cal(nkd).reindex(n225_c.index)
PRICE_SWITCH = nkd.index.min()
r_px = n225_c.pct_change().where(n225_c.index <= PRICE_SWITCH,
                                 nkd_c.pct_change())
price = n225_c.iloc[0] * (1.0 + r_px.fillna(0.0)).cumprod()
_miss = nkd.index.union(cal)
_us_after = cal[cal > PRICE_SWITCH]
log(f"\nPrice moves: ^N225 through {PRICE_SWITCH.date()}, NKD=F after."
    f"  US trading days with no NKD=F print (carried flat): "
    f"{int((~_us_after.isin(nkd.index)).sum())} of {len(_us_after):,}")
bigf = nkd.pct_change()
bigf = bigf[bigf.abs() > 0.09]
log(f"NKD=F one-day moves beyond 9% (inspect for bad prints): {len(bigf)}")
for _d, _v in bigf.items():
    log(f"    {_d.date()}  {_v:+.2%}")
cal_p = price.index
dt = cal_p.to_series().diff().dt.days.fillna(0)

# Dividend yield
print("Dividend yields ...")
# Yahoo carries 1321.T distributions only from July 2013, so its trailing
# yield is usable from July 2014. Before that, EWJ's trailing yield is scaled
# to the Nikkei level using the 1321.T / EWJ ratio over their overlap. The
# overlap stops before December 2025, when EWJ paid a one-off capital gain.
y1321 = trailing_yield("1321.T", 0.0011).loc["2014-07-08":]
yewj = trailing_yield("EWJ", 0.0050)
ov = pd.concat([y1321, yewj], axis=1, keys=["1321", "EWJ"]).dropna().loc[:"2025-11-30"]
ratio = ov["1321"].mean() / ov["EWJ"].mean()
log(f"\nDividend yield overlap {ov.index.min().date()} -> {ov.index.max().date()}:"
    f"  1321.T mean {ov['1321'].mean():.2%}, EWJ-based mean {ov['EWJ'].mean():.2%}"
    f"  -> EWJ scaled by {ratio:.3f} before July 2014")
# Guard (November 1, 2026): the clip below would otherwise hide bad dividend
# data. A Nikkei yield averaging outside 0.3% to 5%, or an EWJ scale outside
# 0.3 to 3, means Yahoo's dividends are wrong, so stop before writing anything.
if not (0.003 <= ov["1321"].mean() <= 0.05 and 0.3 <= ratio <= 3.0):
    sys.exit(f"ERROR: implausible dividend yields (1321.T mean {ov['1321'].mean():.2%},"
             f" EWJ scale {ratio:.3f}). Nothing was written; check Yahoo's 1321.T and EWJ"
             f" dividends and stock splits.")
q = y1321.combine_first(yewj * ratio)
q = to_cal(q).reindex(cal_p).bfill().clip(0.0, 0.05)

# Yen short rate
print("Japan call rate ...")
jp = fetch_oecd_jp()
if jp is not None and len(jp) > 100:
    jp.rename("JP_CALL").to_csv(JP_RATE_CACHE, header=True)
    jp_src = (f"OECD {OECD_JP_NAME} (the source behind FRED {FRED_JP}), fetched now,"
              f" {jp.index.min().strftime('%Y-%m')} -> {jp.index.max().strftime('%Y-%m')}")
elif (jp := fetch_fred(FRED_JP, timeout=60, tries=1)) is not None and len(jp) > 100:
    jp.rename("JP_CALL").to_csv(JP_RATE_CACHE, header=True)
    jp_src = f"FRED {FRED_JP} (fetched now, {jp.index.min().date()} -> {jp.index.max().date()})"
elif os.path.isfile(JP_RATE_CACHE):
    jp = pd.read_csv(JP_RATE_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    jp_src = f"Japan call rate from cached file {JP_RATE_CACHE} (OECD/FRED unreachable now)"
else:
    jp = pd.Series({pd.Timestamp(m + "-01"): v / 100 for m, v in BOJ_FALLBACK})
    jp_src = "FALLBACK Bank of Japan table in this script (OECD and FRED unreachable)"
log(f"Yen short rate source: {jp_src}")
r_jpy = to_cal(jp).reindex(cal_p).bfill()

# Dollar collateral rate
rates = pd.read_csv(RATE_CSV, index_col=0, parse_dates=True)
r_usd = to_cal(rates["EFFR"]).reindex(cal_p).bfill()
log(f"Dollar collateral rate: EFFR from {RATE_CSV}")

# Quanto term from weekly returns (weekly limits the Tokyo/NY close mismatch)
usd_per_jpy = 1.0 / fx
wk = pd.concat([np.log(n225.resample("W-FRI").last()).diff(),
                np.log(usd_per_jpy.resample("W-FRI").last()).diff()],
               axis=1, keys=["n", "x"]).dropna()
rho = wk["n"].rolling(52).corr(wk["x"])
sn = wk["n"].rolling(52).std() * np.sqrt(52)
sx = wk["x"].rolling(52).std() * np.sqrt(52)
qa_w = (rho * sn * sx).dropna()
qa = to_cal(qa_w).reindex(cal_p).bfill()
rho_d = to_cal(rho.dropna()).reindex(cal_p).bfill()

carry_model = q - r_jpy + qa
carry = carry_model + CARRY_CALIBRATION

# ---------------------------------------------------------------------------
# 3. Proxy
# ---------------------------------------------------------------------------
price_ret = price.pct_change().fillna(0.0)
accr = dt / 365.0
excess_ret = price_ret + carry.shift(1).fillna(carry.iloc[0]) * accr
tr_ret = excess_ret + r_usd.shift(1).fillna(r_usd.iloc[0]) * accr
proxy = 100.0 * (1.0 + tr_ret).cumprod()
excess_idx = 100.0 * (1.0 + excess_ret).cumprod()

out = pd.DataFrame({"MNK_SYNTH": proxy})
out.index.name = "Date"
out.to_csv(OUT_CSV, float_format="%.6f")

comp = pd.DataFrame({
    "N225_PRICE": price, "DIV_YIELD": q, "JPY_RATE": r_jpy, "USD_RATE": r_usd,
    "CORR_N225_USDPERJPY": rho_d, "QUANTO_TERM": qa, "CARRY_MODEL": carry_model,
    "CARRY": carry,
    "EXCESS_INDEX": excess_idx, "MNK_SYNTH": proxy})
comp.index.name = "Date"
comp.to_csv(COMPONENTS_CSV, float_format="%.6f")

log(f"\nWrote {OUT_CSV}: {proxy.index.min().date()} -> {proxy.index.max().date()},"
    f" {len(proxy):,} US trading days")

log(f"\nAverage annual carry by era (dividend yield - yen rate + quanto term;"
    f" the proxy then adds {CARRY_CALIBRATION:+.1%} calibration):")
for a, b in [("1993", "2005"), ("2006", "2015"), ("2016", "2026")]:
    sl = slice(a, b)
    log(f"    {a}-{b}:  carry {carry_model.loc[sl].mean():+.2%}   = div {q.loc[sl].mean():.2%}"
        f"  - yen {r_jpy.loc[sl].mean():.2%}  + quanto {qa.loc[sl].mean():+.2%}"
        f"   (corr {rho_d.loc[sl].mean():+.2f});  collateral {r_usd.loc[sl].mean():.2%}")

# ---------------------------------------------------------------------------
# CHECK 1: Yahoo ^N225 vs Interactive Brokers N225, month-end closes
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("CHECK 1  Yahoo ^N225 vs Interactive Brokers N225 month-end closes")
ib = json.load(open(IBKR_N225, encoding="utf-8"))
ibs = pd.Series(ib["close"], index=pd.PeriodIndex(ib["month"], freq="M"))
ibs = ibs.loc[:LAST_FULL_MONTH]
ym = n225.groupby(n225.index.to_period("M")).last()
cmp1 = pd.concat([ym, ibs], axis=1, keys=["yahoo", "ibkr"]).dropna()
diff1 = cmp1["yahoo"] / cmp1["ibkr"] - 1
log(f"    months compared {len(cmp1)}   max |diff| {diff1.abs().max():.4%}"
    f"   months off by >0.05%: {int((diff1.abs() > 0.0005).sum())}")
for p, v in diff1[diff1.abs() > 0.0005].items():
    log(f"        {p}  {v:+.3%}")

# ---------------------------------------------------------------------------
# CHECK 2: modeled carry vs carry measured from real MNK calendar spreads
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("CHECK 2  Carry: model vs real MNK calendar spreads (IBKR closes)")
mk = json.load(open(IBKR_MNK, encoding="utf-8"))
C = {}
for k, v in mk.items():
    if k.startswith("_"):
        continue
    assert len(v["time"]) == len(v["close"]), f"{k}: time/close length mismatch"
    C[k] = pd.Series(v["close"], index=pd.to_datetime(v["time"]), dtype=float)

ROLLS = [
    ("MNKZ5", "MNKH6", ["2025-12-08", "2025-12-09", "2025-12-10"], "2025-12-09"),
    ("MNKH6", "MNKM6", ["2026-03-09", "2026-03-10", "2026-03-11"], "2026-03-10"),
    ("MNKM6", "MNKU6", ["2026-06-08", "2026-06-09", "2026-06-10"], "2026-06-09"),
    ("MNKU6", "MNKZ6", ["2026-09-08", "2026-09-09", "2026-09-10"], "2026-09-08"),
]
gaps = []
for front, nxt, days, _ in ROLLS:
    g = np.mean([np.log(C[nxt].loc[d] / C[front].loc[d]) for d in days])
    gaps.append(g)
    log(f"    {front} -> {nxt}:  next contract {'premium' if g > 0 else 'discount'}"
        f" {g:+.3%}  (mean of {', '.join(days)})")
meas = -sum(gaps)
model = carry_model.loc["2025-12-09":"2026-09-18"].mean()
log(f"    Measured carry, 4 rolls (about one year): {meas:+.2%} per year")
log(f"    Model carry, Dec 2025 - Sep 2026 average: {model:+.2%} per year")
log(f"    Gap (model - measured): {model - meas:+.2%} per year"
    f"  -> proxy uses model {CARRY_CALIBRATION:+.1%}, i.e. {model + CARRY_CALIBRATION:+.2%}")

# ---------------------------------------------------------------------------
# CHECK 3: proxy vs real MNK held and rolled, Sep 2025 - Sep 2026
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("CHECK 3  Proxy vs real MNK, held and rolled two days before expiry")
segs = [("MNKZ5", None, "2025-12-09"), ("MNKH6", "2025-12-09", "2026-03-10"),
        ("MNKM6", "2026-03-10", "2026-06-09"), ("MNKU6", "2026-06-09", "2026-09-08"),
        ("MNKZ6", "2026-09-08", None)]
rets = {}
for k, a, b in segs:
    s = C[k]
    lo = s.index.min() if a is None else pd.Timestamp(a)
    hi = s.index.max() if b is None else pd.Timestamp(b)
    s = s.loc[lo:hi]
    r = s.pct_change().dropna()
    for d, v in r.items():
        rets[d] = v
mnk_ex = pd.Series(rets).sort_index()
mnk_dt = mnk_ex.index.to_series().diff().dt.days.fillna(1) / 365.0
mnk_tr = mnk_ex + to_cal(rates["EFFR"]).reindex(mnk_ex.index).ffill() * mnk_dt
mnk_idx = (1 + mnk_tr).cumprod()
end3 = min(mnk_idx.index.max(), proxy.index.max())
st3 = C["MNKZ5"].index.min()
p3 = proxy.loc[st3:end3]
m3 = pd.concat([pd.Series([1.0], index=[st3]), mnk_idx.loc[:end3]])
tot_mnk = m3.iloc[-1] / 1.0 - 1
tot_px = p3.iloc[-1] / p3.iloc[0] - 1
tot_px_only = price.loc[st3:end3].iloc[-1] / price.loc[st3:end3].iloc[0] - 1
wk3 = ann_stats(m3.resample("W-FRI").last().pct_change(),
                p3.resample("W-FRI").last().pct_change(), 52)
dy3 = ann_stats(m3.pct_change(), p3.reindex(m3.index).ffill().pct_change(), 252)
log(f"    Window {st3.date()} -> {end3.date()}")
log(f"    Total return  real MNK {tot_mnk:+.2%}   proxy {tot_px:+.2%}"
    f"   (Nikkei price only {tot_px_only:+.2%})")
log(f"    Daily returns:  correlation {dy3['corr']:.3f}, tracking error "
    f"{dy3['te']:.2%}/yr, n={dy3['n']}")
log(f"    Weekly returns: correlation {wk3['corr']:.3f}, tracking error "
    f"{wk3['te']:.2%}/yr, n={wk3['n']}")
mm = pd.concat([m3.resample("ME").last(), p3.resample("ME").last()], axis=1,
               keys=["MNK", "PROXY"])
mm.iloc[0] = [1.0, p3.iloc[0]] if False else mm.iloc[0]
mr = mm.pct_change().dropna()
log("    Month     real MNK    proxy     diff")
for d, row in mr.iterrows():
    log(f"    {d.strftime('%Y-%m')}   {row['MNK']:+7.2%}   {row['PROXY']:+7.2%}"
        f"   {row['MNK'] - row['PROXY']:+6.2%}")

# ---------------------------------------------------------------------------
# CHECK 4: price component vs Yahoo NKD=F, monthly, 2004 - 2026
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("CHECK 4  The two price sources agree: NKD=F vs ^N225, monthly")
nkd_m = to_cal(nkd).loc[nkd.index.min():].resample("ME").last().pct_change()
px_m = to_cal(n225).resample("ME").last().pct_change()
st4 = ann_stats(nkd_m.loc[:LAST_FULL_MONTH], px_m.loc[:LAST_FULL_MONTH], 12)
log(f"    Monthly returns {nkd_m.dropna().index.min().strftime('%Y-%m')} -> {LAST_FULL_MONTH}:"
    f" correlation {st4['corr']:.4f}, tracking error {st4['te']:.2%}/yr,"
    f" mean gap {st4['mean_diff']:+.2%}/yr, n={st4['n']}")
log("    (NKD=F is Yahoo's unadjusted front month: roll jumps and the Tokyo vs"
    " Chicago close explain the residual.)")

# ---------------------------------------------------------------------------
# CHECK 5: context, why MNK and DXJ diverge
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("CHECK 5  Annual total returns: MNK proxy vs DXJ vs EWJ (through "
    f"{LAST_FULL_MONTH})")
ann = pd.concat([proxy, to_cal(dxj), to_cal(ewj)], axis=1,
                keys=["MNK proxy", "DXJ", "EWJ"]).loc["2006-06-30":LAST_FULL_MONTH + "-31"]
ye = ann.resample("YE").last()
ye = pd.concat([ann.iloc[[0]], ye])
ar = ye.pct_change().dropna()
log("    Year     MNK proxy      DXJ      EWJ    MNK-DXJ")
for d, row in ar.iterrows():
    log(f"    {d.year}   {row['MNK proxy']:+9.1%}  {row['DXJ']:+7.1%}  {row['EWJ']:+7.1%}"
        f"   {row['MNK proxy'] - row['DXJ']:+7.1%}")
mret = ann.resample("ME").last().pct_change().dropna()
log(f"    Monthly correlation 2006-07 -> {LAST_FULL_MONTH}:  proxy~DXJ"
    f" {mret['MNK proxy'].corr(mret['DXJ']):.3f}   proxy~EWJ"
    f" {mret['MNK proxy'].corr(mret['EWJ']):.3f}   DXJ~EWJ"
    f" {mret['DXJ'].corr(mret['EWJ']):.3f}")
cg = ann.iloc[-1] / ann.iloc[0]
yrs = (ann.index[-1] - ann.index[0]).days / 365.25
log("    CAGR 2006-07 -> " + LAST_FULL_MONTH + ":  " + "   ".join(
    f"{c} {(cg[c] ** (1 / yrs) - 1):+.2%}" for c in ann.columns))

with open(STATS_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
print(f"\nWrote {os.path.abspath(STATS_TXT)}")
