"""
Adaptive Portfolio Strategy - Backtest Dashboard
==================================================
A Streamlit app for rapidly testing parameter variations.

Setup (one time):
  pip install streamlit yfinance pandas numpy matplotlib openpyxl

Run:
  streamlit run "Adaptive Portfolio Dashboard.py"

A browser tab will open with the dashboard. Adjust parameters in the
sidebar and click "Run Backtest" to see results. Every run automatically
appends to "Backtest Results Log.xlsx" in the same folder.

Price data is cached locally in a "price_cache" folder. After the first
run, only new dates and missing tickers are downloaded. Use the "Force
full refresh" checkbox in the sidebar to re-download everything, or
"Clear Cache" to start fresh.

Local index files (e.g., "QCI Daily Prices.csv") placed in the same
folder as the dashboard are automatically detected and available as
backfill sources. These are never sent to Yahoo Finance.
"""

import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Adaptive Portfolio Backtest",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# GLOBAL STYLING (Asymmetric Edge design language, ported from app.py)
# =============================================================================
# Page chrome, typography, dividers, and tables are styled to match the public
# Asymmetric Edge dashboard: DM Sans throughout, a white canvas, charcoal text,
# light gridlines, and a brand-purple accent. The matplotlib rcParams below
# carry the same look into every chart (white background, light grid, muted
# tick labels, no top/right spines) so the backtest charts read like the
# newsletter charts. Adjust the tokens here to restyle the whole app at once.

# ---- Shared design tokens ----
AE_TEXT_PRIMARY = "#484848"     # Charcoal, headings and emphasis
AE_TEXT_SECONDARY = "#666677"   # Axis labels, body
AE_TEXT_MUTED = "#777788"       # Tick labels, captions
AE_GRID = "#eeeeee"             # Gridlines
AE_ZERO_LINE = "#dddddd"        # Zero / reference lines
AE_SPINE = "#d9d9d9"            # Axis spines

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    /* Trim the busiest chrome but keep the run/stop toolbar and menu usable */
    footer {visibility: hidden;}
    [data-testid="stDeployButton"] {display: none;}

    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', sans-serif;
        color: #484848;
    }
    .stApp { background-color: #ffffff; }

    /* Section headings (st.title -> h1, st.subheader -> h3) */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #484848 !important;
    }
    h2, h3 {
        font-weight: 700 !important;
        color: #484848 !important;
        letter-spacing: -0.01em;
        margin-top: 0.25rem !important;
        margin-bottom: 0.35rem !important;
    }

    /* Branded header */
    .ae-brand-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.15rem;
    }
    .ae-brand-header .ae-brand-name {
        font-size: 1.7rem;
        font-weight: 700;
        color: #484848;
        letter-spacing: -0.02em;
    }
    .ae-brand-header .ae-brand-tag {
        font-size: 0.9rem;
        color: #777788;
        border-left: 2px solid #3A0CA3;
        padding-left: 0.55rem;
    }
    .ae-brand-sub {
        font-size: 0.82rem;
        color: #666677;
        margin-bottom: 1.25rem;
    }

    /* Dividers */
    hr { border-color: #e8e8ee !important; }

    /* Pills (if used): brand-purple active state */
    button[data-testid="stBaseButton-pills"][aria-checked="true"] {
        background-color: #f0edf8 !important;
        border-color: #3A0CA3 !important;
        color: #3A0CA3 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- matplotlib defaults carry the same look into every chart ----
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["DM Sans", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
    "text.color": AE_TEXT_PRIMARY,
    "axes.labelcolor": AE_TEXT_SECONDARY,
    "axes.titlecolor": AE_TEXT_PRIMARY,
    "axes.edgecolor": AE_SPINE,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": AE_GRID,
    "grid.linewidth": 0.9,
    "grid.alpha": 1.0,
    "xtick.color": AE_TEXT_MUTED,
    "ytick.color": AE_TEXT_MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#E5E7EB",
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FIXED_ASSETS = {
    "TLT":     "Bonds (20+ Year Treasury)",
    "GLD":     "Gold",
    "BIL":     "Cash (1-3 Month T-Bill)",
    "BTAL":    "Defensive Equity Factor",
    "HGER":    "Commodity (All-Weather)",
    "IBIT":    "Bitcoin (Spot ETF)",
    "BTC-USD": "Bitcoin (Spot Price)",
}

# Expense-ratio drag applied to a backfilled (synthetic) segment, by primary
# ticker. Only needed where the backfill source is the underlying asset
# itself, so the synthetic era would otherwise carry no fund fees.
BACKFILL_FEE_ANNUAL = {
    "IBIT": 0.0025,   # iShares Bitcoin Trust, 0.25%/yr
}

DEFAULT_EQUITY_ETFS = {
    "QQQ":  "US Large-Cap Growth",
    "IWM":  "US Small-Cap Blend",
    "DXJ":  "Japan Hedged Equity",
    "EMXC": "EM ex-China",
}

EQUITY_LOOKBACK_OPTIONS = {
    "1-month only":                 ([1],       [1.0]),
    "2-month only":                 ([2],       [1.0]),
    "3-month only":                 ([3],       [1.0]),
    "1/2-month blend (50/50)":      ([1, 2],    [0.50, 0.50]),
    "1/3-month blend (50/50)":      ([1, 3],    [0.50, 0.50]),
    "2/3-month blend (50/50)":      ([2, 3],    [0.50, 0.50]),
    "1/2/3-month blend (34/33/33)": ([1, 2, 3], [0.34, 0.33, 0.33]),
}

MAIN_LOOKBACK_OPTIONS = {
    "6-month only":               ([6],       [1.0]),
    "6/8-month blend (50/50)":    ([6, 8],    [0.50, 0.50]),
    "1/8-month blend (50/50)":    ([1, 8],    [0.50, 0.50]),
    "2/8-month blend (50/50)":    ([2, 8],    [0.50, 0.50]),
    "2/11-month blend (50/50)":   ([2, 11],   [0.50, 0.50]),
}

BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 10_000

# ---- Brand colors (mirrors the Asymmetric Edge newsletter palette) ----
# Series colors are consistent across every chart in the dashboard:
#   Strategy = brand purple, SPY = muted indigo, 60/40 = brand teal.
# Categorical colors reuse teal (positive / risk-on) and purple (negative /
# risk-off), matching the asset class chart convention in app.py.
BRAND_PURPLE = "#3A0CA3"        # Strategy / primary (hero, matches app.py)
BRAND_PURPLE_FADE = "#8B7BC9"   # Lightened purple for secondary lines (e.g. 24mo Sortino)
BRAND_INDIGO = "#9CA3AF"        # SPY / benchmark (neutral gray, matches app.py S&P 500)
BRAND_BLUE = "#4361EE"          # 80/20 benchmark (matches app.py)
BRAND_CYAN = "#4CC9F0"          # 60/40 benchmark (matches app.py)
BRAND_TEAL = "#30C7B5"          # Positive regime / excess accent (brand teal)
BRAND_STEEL = "#6A9BAD"         # Legacy steel blue-green (retained for reference)
BRAND_RED = "#C0392B"           # Loss / drawdown / risk-off semantic (matches app.py)
BRAND_GRAY = "#A8A8A8"          # Neutral / reference lines
BRAND_TEXT = "#484848"          # Charcoal text
BRAND_HIGHLIGHT_BG = "#F0EDF8"  # Light purple tint for selected table rows

# ---- Balanced benchmarks ----
# Benchmarks shown alongside SPY in the equity, drawdown, and metrics
# tables. Edit this list to swap, add, or remove benchmarks. Each entry:
#   label      shown in legends and the metrics table column header
#   ticker     Yahoo Finance ticker (must be downloadable)
#   color      matplotlib color (use a BRAND_* constant or a hex string)
#   linestyle  matplotlib line style ("-", "--", "-.", ":")
#   backfill   (optional) list of older tickers used to extend the
#              benchmark's history backward via return-splicing. These are
#              applied automatically on every run and do not appear in the
#              sidebar backfill rows. AOR/AOA only go back to their 2008
#              ETF inception, so we splice the Vanguard LifeStrategy mutual
#              funds (global, same 60/40 and 80/20 structure, inception
#              Sept 1994) onto them to reach the mid-1990s.
# Set to [] to disable balanced benchmarks entirely.
BAL_BENCHMARKS = [
    {"label": "60/40", "ticker": "AOR", "color": BRAND_CYAN, "linestyle": "--", "backfill": ["VSMGX"]},
    {"label": "80/20", "ticker": "AOA", "color": BRAND_BLUE, "linestyle": "-.", "backfill": ["VASGX"]},
]

_base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
RESULTS_FILE = os.path.join(_base_dir, "Backtest Results Log.xlsx")
CACHE_DIR = os.path.join(_base_dir, "price_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "daily_prices.csv")
LOCAL_INDEX_DIR = _base_dir  # Local index CSVs live alongside the dashboard


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def load_price_cache():
    """Load cached daily prices from disk, if available."""
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return df
    except Exception:
        return pd.DataFrame()


def save_price_cache(df):
    """Save daily prices to the local disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(CACHE_FILE)


def get_cache_info():
    """Return a dict with cache metadata, or None if no cache exists."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        modified = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True, nrows=0)
        return {
            "last_updated": modified,
            "tickers": sorted(df.columns.tolist()),
            "n_tickers": len(df.columns),
            "size_mb": size_mb,
        }
    except Exception:
        return None


def load_local_index_data():
    """
    Load any local index CSV files that live alongside the dashboard.
    These provide synthetic price series for indices that aren't available
    through Yahoo Finance (e.g., QCI for HGER backfill).

    Each CSV should have a Date index column and one or more price columns.
    Files must follow the naming pattern: '<TICKER> Daily Prices.csv'
    """
    local_data = pd.DataFrame()
    if not os.path.isdir(LOCAL_INDEX_DIR):
        return local_data

    for fname in os.listdir(LOCAL_INDEX_DIR):
        if fname.endswith(" Daily Prices.csv"):
            fpath = os.path.join(LOCAL_INDEX_DIR, fname)
            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                if local_data.empty:
                    local_data = df
                else:
                    local_data = local_data.join(df, how="outer")
            except Exception:
                continue

    return local_data


def download_data(tickers_tuple, start_date_str, force_refresh=False):
    """
    Download adjusted close prices for a list of tickers, using a local
    disk cache to avoid redundant downloads.

    On the first run, all data is fetched from Yahoo Finance and saved
    to price_cache/daily_prices.csv. On later runs, only missing tickers
    and recent dates are downloaded. Set force_refresh=True to ignore
    the cache and re-download everything.

    Tickers that match a local index CSV file (e.g., QCI) are loaded
    from disk and never sent to Yahoo Finance.
    """
    unique_tickers = sorted(set(tickers_tuple))
    if not unique_tickers:
        return pd.DataFrame()

    # Separate local index tickers from Yahoo tickers
    local_index_data = load_local_index_data()
    local_tickers = [t for t in unique_tickers if t in local_index_data.columns]
    yahoo_tickers = [t for t in unique_tickers if t not in local_index_data.columns]

    # --- Yahoo Finance portion (only for non-local tickers) ---
    cached = pd.DataFrame() if force_refresh else load_price_cache()
    start_date = pd.Timestamp(start_date_str)

    needs_full_download = []
    needs_update = []

    for t in yahoo_tickers:
        if cached.empty or t not in cached.columns:
            needs_full_download.append(t)
        else:
            series = cached[t].dropna()
            if series.empty:
                needs_full_download.append(t)
            elif series.index.min() > start_date + pd.Timedelta(days=7):
                needs_full_download.append(t)
            else:
                needs_update.append(t)

    total_to_fetch = len(needs_full_download) + (1 if needs_update else 0)
    if total_to_fetch > 0:
        status_parts = []
        if needs_full_download:
            status_parts.append(f"downloading {len(needs_full_download)} new ticker(s)")
        if needs_update:
            status_parts.append(f"updating {len(needs_update)} cached ticker(s)")
        if local_tickers:
            status_parts.append(f"{len(local_tickers)} from local index files")
        status_msg = "Cache: " + ", ".join(status_parts) + "..."
    elif local_tickers:
        status_msg = f"Loading from cache + {len(local_tickers)} local index file(s)..."
    else:
        status_msg = "Loading from cache..."

    progress = st.sidebar.empty()
    progress.caption(status_msg)

    full_dl = pd.DataFrame()
    if needs_full_download:
        data = yf.download(
            needs_full_download, start=start_date_str,
            auto_adjust=True, progress=False
        )
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                full_dl = data["Close"]
            else:
                full_dl = data[["Close"]]
                if len(needs_full_download) == 1:
                    full_dl.columns = needs_full_download

    delta_dl = pd.DataFrame()
    if needs_update:
        latest_cached = cached[needs_update].dropna(how="all").index.max()
        update_from = (latest_cached - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        data = yf.download(
            needs_update, start=update_from,
            auto_adjust=True, progress=False
        )
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                delta_dl = data["Close"]
            else:
                delta_dl = data[["Close"]]
                if len(needs_update) == 1:
                    delta_dl.columns = needs_update

    result = pd.DataFrame()

    if needs_update and not cached.empty:
        result = cached[[t for t in needs_update if t in cached.columns]].copy()
        if not delta_dl.empty:
            new_dates = delta_dl.index.difference(result.index)
            if len(new_dates) > 0:
                result = pd.concat([result, delta_dl.loc[new_dates]])
            overlap = delta_dl.index.intersection(result.index)
            if len(overlap) > 0:
                result.loc[overlap, delta_dl.columns] = delta_dl.loc[overlap]

    if not full_dl.empty:
        if result.empty:
            result = full_dl
        else:
            result = result.join(full_dl, how="outer")

    # --- Merge local index data ---
    if local_tickers:
        local_subset = local_index_data[local_tickers]
        if result.empty:
            result = local_subset
        else:
            result = result.join(local_subset, how="outer")

    if result.empty:
        progress.empty()
        return pd.DataFrame()

    result = result.sort_index()

    # Persist Yahoo data to disk cache (exclude local index tickers)
    yahoo_cols = [c for c in result.columns if c not in local_tickers]
    if yahoo_cols:
        yahoo_result = result[yahoo_cols]
        if not cached.empty and not force_refresh:
            # Union the indexes first. A plain `cached[col] = ...` aligns on
            # the cached index only, so newly downloaded dates were silently
            # dropped and the cache never advanced past its original last
            # date -- forcing a full delta re-download on every launch.
            merged = cached.reindex(cached.index.union(yahoo_result.index))
            for col in yahoo_result.columns:
                incoming = yahoo_result[col].reindex(merged.index)
                if col in cached.columns:
                    merged[col] = incoming.combine_first(merged[col])
                else:
                    merged[col] = incoming
            save_price_cache(merged.sort_index())
        else:
            save_price_cache(yahoo_result)

    progress.caption(f"Cache: {len(unique_tickers)} tickers loaded ({len(local_tickers)} from local files).")

    available = [t for t in unique_tickers if t in result.columns]
    return result[available]


_LOCAL_INDEX_COLS = None


def _local_index_columns():
    """Column names sourced from local '<TICKER> Daily Prices.csv' files."""
    global _LOCAL_INDEX_COLS
    if _LOCAL_INDEX_COLS is None:
        try:
            _LOCAL_INDEX_COLS = set(load_local_index_data().columns)
        except Exception:
            _LOCAL_INDEX_COLS = set()
    return _LOCAL_INDEX_COLS


def market_trading_dates(daily_prices):
    """
    The exchange trading calendar implied by daily_prices.

    daily_prices may contain 24/7 assets (BTC-USD and other "-USD" pairs),
    which carry bars on weekends and market holidays. Treating the raw
    index as a trading calendar puts execution dates on days the ETFs did
    not trade -- roughly one month in four, since ~28% of calendar
    month-ends fall on a Saturday or Sunday.

    Local index CSVs (GOLD, BAB, WTI) are synthetic 7-day series and are
    excluded for the same reason -- they are not exchange-listed and would
    otherwise vote weekend dates into the calendar.

    Build the calendar from the remaining columns: every weekday on which
    at least one exchange-listed ticker printed a close. Falls back to the
    full index if nothing qualifies.
    """
    if daily_prices.empty:
        return daily_prices.index

    market_cols = [
        c for c in daily_prices.columns
        if not str(c).upper().endswith("-USD")
        and c not in _local_index_columns()
    ]
    if not market_cols:
        return daily_prices.index.sort_values()

    valid = daily_prices[market_cols].dropna(how="all")
    if valid.empty:
        return daily_prices.index.sort_values()

    # Hard guard: no US exchange trades on a Saturday or Sunday, whatever
    # a data file claims.
    idx = valid.index
    return idx[idx.dayofweek < 5].sort_values()


def build_backfilled_series(daily_prices, primary_ticker, backfill_tickers,
                            fee_annual=0.0):
    """
    Build a single price series for primary_ticker, extending it backward
    using backfill tickers (in priority order) where the primary has no data.

    The backfill uses return-splicing: we compute daily returns from the
    backfill ETF and chain them onto the primary's earliest known price,
    so the price levels connect smoothly.

    fee_annual applies an expense-ratio drag to the synthetic (backfilled)
    segment only. Use it when the primary is an ETF and the backfill source
    is the underlying asset itself -- e.g. IBIT backfilled with BTC-USD.
    The primary's own history already has its fees baked into price, so
    without this the synthetic era would look better than the ETF era.

    The synthetic segment is also masked to the exchange calendar whenever
    the primary is not itself a 24/7 asset. A 24/7 backfill source such as
    BTC-USD would otherwise hand the ETF weekend closes it could never have
    printed, and would put weekend dates back into the trading calendar.

    Returns the spliced series and a list of segment descriptions.
    """
    is_crypto_primary = str(primary_ticker).upper().endswith("-USD")
    calendar = None if is_crypto_primary else market_trading_dates(daily_prices)
    if primary_ticker in daily_prices.columns:
        primary = daily_prices[primary_ticker].copy()
    else:
        primary = pd.Series(dtype=float, index=daily_prices.index, name=primary_ticker)

    primary_valid = primary.dropna()
    segments = []

    if not primary_valid.empty:
        segments.append((primary_ticker, primary_valid.index.min(), primary_valid.index.max()))

    combined = primary.copy()

    for bf_ticker in backfill_tickers:
        if bf_ticker not in daily_prices.columns:
            continue

        bf_series = daily_prices[bf_ticker].dropna()
        if bf_series.empty:
            continue

        combined_valid = combined.dropna()
        if combined_valid.empty:
            # No primary data at all, just use this backfill directly
            combined = bf_series.reindex(daily_prices.index)
            segments.insert(0, (bf_ticker, bf_series.index.min(), bf_series.index.max()))
            continue

        earliest_combined = combined_valid.index.min()

        # Only backfill dates before what we already have
        bf_before = bf_series.loc[:earliest_combined]
        if len(bf_before) < 2:
            continue

        # Return-splice: scale backfill prices so the last backfill price
        # matches the first combined price
        anchor_price = combined_valid.iloc[0]
        bf_last_price = bf_before.iloc[-1]
        if bf_last_price == 0:
            continue

        scale_factor = anchor_price / bf_last_price
        bf_scaled = bf_before * scale_factor

        # Expense-ratio drag on the synthetic segment. Anchored at the
        # primary's first real price and worked backwards, so a fee makes
        # the synthetic series start HIGHER and grow more slowly.
        if fee_annual and fee_annual > 0:
            yrs = (earliest_combined - bf_scaled.index).days / 365.25
            bf_scaled = bf_scaled / ((1.0 - fee_annual) ** yrs)

        # Fill in the earlier dates (exclude the overlap date itself)
        fill_dates = bf_scaled.index[bf_scaled.index < earliest_combined]
        if calendar is not None:
            fill_dates = fill_dates.intersection(calendar)
        if len(fill_dates) == 0:
            continue
        combined.loc[fill_dates] = bf_scaled.loc[fill_dates]

        segments.insert(0, (bf_ticker, fill_dates.min(), fill_dates.max()))

    return combined, segments


def get_month_end_prices(prices):
    # Forward-fill daily prices so that weekends/holidays at month-end
    # carry the last valid trading day's close, preventing spurious NaNs
    return prices.ffill().resample("ME").last()


# =============================================================================
# STRATEGY FUNCTIONS
# =============================================================================

def compute_total_return(prices_monthly, lookback_months):
    return prices_monthly / prices_monthly.shift(lookback_months) - 1


def compute_blended_momentum(prices_monthly, lookbacks, weights):
    scores = pd.DataFrame(index=prices_monthly.index, columns=prices_monthly.columns, dtype=float)
    scores[:] = 0.0
    for lb, w in zip(lookbacks, weights):
        ret = compute_total_return(prices_monthly, lb)
        scores = scores + ret * w
    return scores


def compute_risk_parity_weights(daily_prices, selected_tickers, as_of_date, vol_months=12):
    lookback_start = as_of_date - pd.DateOffset(months=vol_months)
    subset = daily_prices[selected_tickers].loc[lookback_start:as_of_date]

    # Drop any row where ANY ticker is NaN so volatility is computed over
    # identical trading days for all assets.  This prevents spurious large
    # returns from calendar mismatches (BTC-USD weekends, foreign holidays).
    subset = subset.dropna()

    if len(subset) < 20:
        n = len(selected_tickers)
        return {t: 1.0 / n for t in selected_tickers}

    daily_returns = subset.pct_change().dropna()
    if len(daily_returns) < 20:
        n = len(selected_tickers)
        return {t: 1.0 / n for t in selected_tickers}

    vols = daily_returns.std() * np.sqrt(252)
    vols = vols.replace(0, np.nan)
    valid_tickers = vols.dropna().index.tolist()

    if len(valid_tickers) == 0:
        n = len(selected_tickers)
        return {t: 1.0 / n for t in selected_tickers}

    inv_vol = 1.0 / vols[valid_tickers]
    weights_raw = inv_vol / inv_vol.sum()

    weights = {}
    for t in selected_tickers:
        weights[t] = weights_raw.get(t, 0.0)

    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


def compute_leverage_signal(daily_prices, as_of_date, leverage_params,
                            final_holdings=None):
    """
    Compute a leverage multiplier using one of two methods.

    Method 'holdings_gate':
        Binary approach. If any defensive tickers (e.g., BIL, BTAL) are
        present in the final holdings, leverage = 1.0x (the strategy is
        already signaling caution). Otherwise, apply the configured
        multiplier. No external data needed.

    Method 'vt_trend':
        Uses a global equity ETF's moving averages to assess trend.
        Risk-on (price > long SMA and short SMA rising) -> leverage up.
        Neutral (price > long SMA but short SMA falling) -> 1.0x.
        Risk-off (price < long SMA) -> scale down.

    Returns:
        (multiplier, regime_label, detail_dict)
    """
    if not leverage_params.get("enabled", False):
        return 1.0, "off", {}

    method = leverage_params.get("method", "holdings_gate")

    # ------------------------------------------------------------------
    # METHOD: Constant
    # ------------------------------------------------------------------
    if method == "constant":
        mult = leverage_params.get("mult_constant", 1.25)
        return mult, "constant", {"leveraged": True}

    # ------------------------------------------------------------------
    # METHOD: Holdings gate
    # ------------------------------------------------------------------
    elif method == "holdings_gate":
        defensive_tickers = leverage_params.get("defensive_tickers", ["BIL", "BTAL"])
        mult_leveraged = leverage_params.get("mult_leveraged", 1.25)

        if final_holdings is None:
            return 1.0, "no holdings", {}

        defensives_held = [t for t in defensive_tickers if t in final_holdings]

        if defensives_held:
            return 1.0, "defensive", {
                "defensives_held": defensives_held,
                "leveraged": False,
            }
        else:
            return mult_leveraged, "leveraged", {
                "defensives_held": [],
                "leveraged": True,
            }

    # ------------------------------------------------------------------
    # METHOD: VT trend
    # ------------------------------------------------------------------
    elif method == "vt_trend":
        ticker = leverage_params["ticker"]
        long_sma = leverage_params["long_sma"]
        short_sma = leverage_params["short_sma"]
        slope_lookback = leverage_params["slope_lookback"]

        if ticker not in daily_prices.columns:
            return 1.0, "no data", {}

        prices = daily_prices[ticker].loc[:as_of_date].dropna()

        min_needed = max(long_sma, short_sma + slope_lookback) + 10
        if len(prices) < min_needed:
            return 1.0, "insufficient history", {}

        current_price = prices.iloc[-1]
        long_sma_val = prices.iloc[-long_sma:].mean()
        short_sma_now = prices.iloc[-short_sma:].mean()

        prices_earlier = prices.iloc[:-(slope_lookback)]
        if len(prices_earlier) < short_sma:
            return 1.0, "insufficient history", {}
        short_sma_prev = prices_earlier.iloc[-short_sma:].mean()

        price_above_long = current_price > long_sma_val
        short_sma_rising = short_sma_now > short_sma_prev

        detail = {
            "price": current_price,
            "long_sma": long_sma_val,
            "short_sma_now": short_sma_now,
            "short_sma_prev": short_sma_prev,
            "price_above_long": price_above_long,
            "short_sma_rising": short_sma_rising,
        }

        if price_above_long and short_sma_rising:
            return leverage_params["mult_risk_on"], "risk-on", detail
        elif not price_above_long:
            return leverage_params["mult_risk_off"], "risk-off", detail
        else:
            return leverage_params["mult_neutral"], "neutral", detail

    # Unknown method
    return 1.0, "unknown", {}


# =============================================================================
# BITCOIN TREND GATE (Radius Trend filter)
# =============================================================================

def download_gate_ohlc(ticker, start_date_str, force_refresh=False):
    """
    Download daily OHLC for a single trend-gate ticker (e.g. BTC-USD),
    used only to compute the Bitcoin trend gate. The main price cache keeps
    adjusted close only, so the gate needs high/low fetched separately.
    Cached on its own in price_cache/gate_ohlc_<ticker>.csv. Returns a
    DataFrame with High, Low, Close columns (daily), or an empty DataFrame
    on failure. BTC-USD has no dividends or splits, so adjusted and raw
    prices match.
    """
    safe = "".join(c if c.isalnum() else "_" for c in ticker)
    cache_path = os.path.join(CACHE_DIR, f"gate_ohlc_{safe}.csv")

    cached = pd.DataFrame()
    if not force_refresh and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except Exception:
            cached = pd.DataFrame()

    start_from = start_date_str
    if not cached.empty:
        # Re-download fully if the cache starts later than we now need
        if cached.index.min() > pd.Timestamp(start_date_str) + pd.Timedelta(days=7):
            cached = pd.DataFrame()
        else:
            last = cached.dropna(how="all").index.max()
            start_from = (last - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    fresh = pd.DataFrame()
    try:
        data = yf.download(ticker, start=start_from, auto_adjust=True, progress=False)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            keep = [c for c in ["High", "Low", "Close"] if c in data.columns]
            if len(keep) == 3:
                fresh = data[keep].copy()
    except Exception:
        fresh = pd.DataFrame()

    if fresh.empty and cached.empty:
        return pd.DataFrame()

    if fresh.empty:
        combined = cached
    elif cached.empty:
        combined = fresh
    else:
        combined = cached.copy()
        new_dates = fresh.index.difference(combined.index)
        if len(new_dates) > 0:
            combined = pd.concat([combined, fresh.loc[new_dates]])
        overlap = fresh.index.intersection(combined.index)
        if len(overlap) > 0:
            combined.loc[overlap, fresh.columns] = fresh.loc[overlap]

    combined = combined.sort_index()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        combined.to_csv(cache_path)
    except Exception:
        pass

    keep = [c for c in ["High", "Low", "Close"] if c in combined.columns]
    return combined[keep] if len(keep) == 3 else pd.DataFrame()


def compute_radius_trend_state(ohlc, step=0.15, multi=0.5, n=3):
    """
    Python port of the 'Radius Trend Strategy [ChartPrime]' Pine Script v5,
    a long-only trailing-band trend filter, run on one asset's daily OHLC.

    The band resets just below the recent low when trend flips up and just
    above the recent high when it flips down, then ratchets toward price at
    an accelerating rate every n-th bar (the multi1/multi2 step accumulation).

    Returns a DataFrame indexed like `ohlc` with columns:
        trend      1.0 when close is above the band (uptrend), 0.0 below,
                   NaN before the indicator is seeded
        band       the trailing band level
        band1      the trade trigger line (upper band in an uptrend, lower
                   band in a downtrend); an n-bar SMA, per the script
        position   1.0 when the long-only strategy would be in the market,
                   0.0 when flat. Enters on trend up with close > band1,
                   exits on trend down with close < band1, holds in between.
        close      the bar's close (carried for display)

    The indicator is causal (each bar depends only on prior and current
    bars), so computing the full series once and reading it as of a past
    date introduces no look-ahead. This is a close approximation of the
    TradingView indicator; small differences can arise from data-feed
    calendar and start-date differences, since the n-bar ratchet cadence
    is bar-count dependent.
    """
    cols = ["trend", "band", "band1", "position", "close"]
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.DataFrame(columns=cols)

    df = ohlc[["High", "Low", "Close"]].dropna()
    nbars = len(df)
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    if nbars < 103:
        return out  # not enough bars to seed the 100-bar range average

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    rng = np.abs(high - low)
    range_sma = pd.Series(rng).rolling(100).mean().to_numpy()
    distance = range_sma * multi      # band reset offset
    distance1 = range_sma * 0.2       # per-step ratchet size

    trend_arr = np.full(nbars, np.nan)
    band_arr = np.full(nbars, np.nan)

    band = np.nan
    multi1 = 0.0
    multi2 = 0.0
    prev_trend = np.nan

    for i in range(nbars):
        trend = np.nan

        # Seed at bar index 101 (matches Pine's `if bar_index == 101`)
        if i == 101:
            band = low[i] * 0.8

        if not np.isnan(band):
            if close[i] < band:
                trend = 0.0
            elif close[i] > band:
                trend = 1.0

        if np.isnan(trend):
            band_arr[i] = band
            prev_trend = trend
            continue

        # Band reset on a trend flip (uses this bar's distance)
        if (not np.isnan(prev_trend)) and (trend != prev_trend):
            if prev_trend == 0.0 and trend == 1.0:
                band = low[i] - distance[i]
            elif prev_trend == 1.0 and trend == 0.0:
                band = high[i] + distance[i]

        # Accelerating ratchet every n-th bar
        if (i % n == 0) and trend == 1.0:
            multi1 = 0.0
            multi2 += step
            band += distance1[i] * multi2
        elif (i % n == 0) and trend == 0.0:
            multi1 += step
            multi2 = 0.0
            band -= distance1[i] * multi1

        trend_arr[i] = trend
        band_arr[i] = band
        prev_trend = trend

    # Trigger line band1 = n-bar SMA of (band +/- distance*0.5)
    band_series = pd.Series(band_arr, index=df.index)
    dist_series = pd.Series(distance, index=df.index)
    trend_series = pd.Series(trend_arr, index=df.index)
    band_upper = (band_series + dist_series * 0.5).rolling(n).mean()
    band_lower = (band_series - dist_series * 0.5).rolling(n).mean()
    band1 = band_upper.where(trend_series == 1.0, band_lower)
    b1 = band1.to_numpy()

    # Long-only position state (stateful sweep matching strategy.entry/close)
    position = np.zeros(nbars)
    in_pos = False
    for i in range(nbars):
        if np.isnan(trend_arr[i]) or np.isnan(b1[i]):
            position[i] = 1.0 if in_pos else 0.0
            continue
        if (trend_arr[i] == 1.0) and (close[i] > b1[i]):
            in_pos = True
        elif (trend_arr[i] == 0.0) and (close[i] < b1[i]):
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0

    out["trend"] = trend_series
    out["band"] = band_series
    out["band1"] = band1
    out["position"] = position
    out["close"] = pd.Series(close, index=df.index)
    return out


def compute_sma_cross_state(ohlc, sma_len=150, rf_len=20, rf_period=20):
    """
    Daily port of the "Simple SMA Cross Strategy: 8H BTC" Pine Script v5.

    The original runs on 8-hour bars with a 450-bar long SMA and a 60-bar
    rising/falling SMA. Bitcoin trades around the clock, so a day holds three
    8-hour bars. Dividing the bar counts by three converts the rules to daily
    bars (450 -> 150, 60 -> 20) while preserving the same calendar windows.

    Long-only state machine, matching the script's entries and exits:
        ENTER  when close > long SMA, the prior close was at or above the long
               SMA, and the fast SMA has risen on each of the last rf_period
               bars (Pine's ta.rising).
        EXIT   when close is below the long SMA for three straight bars, OR the
               fast SMA has fallen on each of the last rf_period bars (Pine's
               ta.falling).
        HOLD   in between (pyramiding is 1 in the script, so a single
               long-or-flat state).

    Returns a DataFrame indexed like ohlc with columns:
        position   1.0 while the long-only strategy is in the market, 0.0 flat.
                   This is the faithful entry/exit state machine.
        trend      1.0 whenever close is above the long SMA and the fast SMA is
                   rising, 0.0 otherwise. A looser, stateless reading used by
                   the "trend up only" gate mode.
        sma_long   the long trend SMA (default 150-day)
        sma_fast   the fast rising/falling SMA (default 20-day)
        close      the bar's close (carried for display)

    The state machine is causal (each bar depends only on prior and current
    bars), so reading the position as of a past date introduces no look-ahead.
    Because the daily bar count differs from the 8-hour chart, this is a close
    approximation of the TradingView result rather than an exact reproduction.
    """
    cols = ["position", "trend", "sma_long", "sma_fast", "close"]
    if ohlc is None or ohlc.empty or "Close" not in ohlc.columns:
        return pd.DataFrame(columns=cols)

    close = ohlc["Close"].dropna().astype(float)
    n = len(close)
    out = pd.DataFrame(index=close.index, columns=cols, dtype=float)
    if n < max(sma_len, rf_len + rf_period) + 3:
        out["close"] = close
        return out  # not enough history to seed the SMAs

    sma_long = close.rolling(sma_len).mean()
    sma_fast = close.rolling(rf_len).mean()

    # ta.rising(src, len) / ta.falling(src, len): every one of the last `len`
    # bar-over-bar steps of the fast SMA was up (rising) or down (falling).
    step = sma_fast.diff()
    up_steps = (step > 0).astype(float)
    dn_steps = (step < 0).astype(float)
    rising = (up_steps.rolling(rf_period).sum() == rf_period)
    falling = (dn_steps.rolling(rf_period).sum() == rf_period)

    above = close > sma_long
    prev_at_or_above = close.shift(1) >= sma_long.shift(1)
    long_cond = (above & prev_at_or_above & rising).to_numpy()

    below = close < sma_long
    below1 = below.shift(1, fill_value=False)
    below2 = below.shift(2, fill_value=False)
    exit_three_below = below & below1 & below2
    exit_cond = (exit_three_below | falling).to_numpy()

    # Stateful sweep matching strategy.entry("Long") / strategy.close_all()
    position = np.zeros(n)
    in_pos = False
    for i in range(n):
        if not in_pos:
            if long_cond[i]:
                in_pos = True
        else:
            if exit_cond[i]:
                in_pos = False
        position[i] = 1.0 if in_pos else 0.0

    out["position"] = position
    out["trend"] = (above & rising).astype(float)
    out["sma_long"] = sma_long
    out["sma_fast"] = sma_fast
    out["close"] = close
    return out


def gate_signal_ticker(gate):
    """
    The ticker whose price the gate's trend signal is computed FROM.

    Distinct from gate["ticker"], which is the basket asset the gate
    admits or excludes. They differ when the tradable instrument is a
    wrapper around the real asset: hold IBIT, but read the trend from
    BTC-USD, which has a decade more history and trades every day of the
    week. Falls back to the gated ticker so older configs still work.
    """
    return gate.get("signal_ticker") or gate.get("ticker")


def gate_label(gate):
    """'IBIT (signal: BTC-USD)', or just 'IBIT' when they are the same."""
    held = gate.get("ticker")
    src = gate_signal_ticker(gate)
    return held if src == held else f"{held} (signal: {src})"


def _gate_open_as_of(btc_gate, as_of_date):
    """
    Return True if the Bitcoin trend gate is open as of `as_of_date`.
    When the gate is disabled or unconfigured, returns True so the gated
    ticker behaves like any other fixed-basket asset (default behavior is
    unchanged unless the gate is explicitly enabled).
    """
    if not btc_gate or not btc_gate.get("enabled"):
        return True
    series = btc_gate.get("gate_series")
    if series is None or len(series) == 0:
        return False
    s = series.loc[:as_of_date]
    if len(s) == 0:
        return False
    return bool(s.iloc[-1])


def run_backtest(daily_prices, monthly_prices, params):
    """
    Run the adaptive portfolio backtest with the given parameters.

    Execution timing (params["exec_timing"]):
      "last_day"  (default) - allocations (momentum, gates, risk parity,
                  leverage) are computed from data through the close of the
                  trading day BEFORE the last trading day of the month, and
                  trades execute at the last trading day's close. This
                  mirrors real-world execution: take the signal from the
                  penultimate close, enter positions near the close on the
                  final trading day. No look-ahead.
      "first_day" - signals are computed from month-end closing prices and
                  trades execute at the NEXT trading day's close (1-day
                  lag). This is the legacy behavior.
    """
    fixed_tickers = params["fixed_tickers"]
    equity_tickers = params["equity_tickers"]
    equity_top_n = params["equity_top_n"]
    equity_lookbacks = params["equity_lookbacks"]
    equity_lookback_weights = params["equity_lookback_weights"]
    main_lookbacks = params["main_lookbacks"]
    main_lookback_weights = params["main_lookback_weights"]
    final_top_n = params["final_top_n"]
    vol_months = params["vol_months"]
    backtest_start = params["backtest_start"]
    backtest_end = params.get("backtest_end", "")
    leverage_params = params.get("leverage_params", {"enabled": False})
    exec_timing = params.get("exec_timing", "last_day")

    # Bitcoin trend gate (Radius Trend). When enabled, the gated ticker is
    # only admitted to the basket on months when its trend signal is "in".
    btc_gate = params.get("btc_gate", {})
    btc_gate_ticker = btc_gate.get("ticker") if btc_gate.get("enabled") else None

    # Bitcoin SMA cross gate (separate, parallel filter ported from the 8H
    # Pine strategy). Independent of the Radius Trend gate. When both gates are
    # enabled on the same ticker, the ticker must pass BOTH to be eligible
    # (the stricter, drawdown-friendly combination).
    btc_sma_gate = params.get("btc_sma_gate", {})
    btc_sma_gate_ticker = btc_sma_gate.get("ticker") if btc_sma_gate.get("enabled") else None

    min_lookback = max(main_lookbacks)
    valid_months = monthly_prices.index[min_lookback:]

    # Build sorted array of all trading dates for execution-day lookups.
    # Anchored to the exchange calendar, NOT daily_prices.index -- BTC-USD
    # weekend bars would otherwise place month-end executions on Saturdays
    # and Sundays.
    all_trading_dates = market_trading_dates(daily_prices)

    def next_trading_day(date):
        """Return the first trading day strictly AFTER the given date."""
        future = all_trading_dates[all_trading_dates > date]
        if len(future) == 0:
            return None
        return future[0]

    def last_trading_day(date):
        """Return the last trading day AT OR BEFORE the given date."""
        past = all_trading_dates[all_trading_dates <= date]
        if len(past) == 0:
            return None
        return past[-1]

    def resolve_exec_date(month_end_date):
        """Map a calendar month-end label to its execution trading day."""
        if exec_timing == "first_day":
            return next_trading_day(month_end_date)
        # "last_day": execute at the close of the final trading day of the
        # month. Skip incomplete months (label beyond the latest data).
        if month_end_date > all_trading_dates[-1] and month_end_date.month == all_trading_dates[-1].month and month_end_date.year == all_trading_dates[-1].year:
            return None
        return last_trading_day(month_end_date)

    # Forward-filled daily prices for signal-date lookups (computed once)
    daily_ffill = daily_prices.ffill()

    results = []
    trade_log = []
    daily_portfolio_values = {}  # {date: portfolio_value} for daily drawdown
    cumulative_multiplier = 1.0  # tracks growth across holding periods

    for i, rebal_date in enumerate(valid_months):
        if i >= len(valid_months) - 1:
            break

        next_month_end = valid_months[i + 1]

        # Execution day for this rebalance (needed up front so the signal
        # cutoff can be derived from it in "last_day" mode).
        exec_date_entry = resolve_exec_date(rebal_date)
        if exec_date_entry is None:
            continue

        # Signal cutoff: the last date whose data may inform this
        # rebalance's allocations.
        #   "last_day"  -> the day before the execution day (penultimate
        #                  trading day's close). No look-ahead.
        #   "first_day" -> the month-end itself (execution is the next
        #                  trading day, so there is already a 1-day lag).
        if exec_timing == "last_day":
            signal_date = exec_date_entry - pd.Timedelta(days=1)
        else:
            signal_date = rebal_date

        def signal_monthly_frame(cols):
            """Monthly closes through rebal_date, with the current month's
            value replaced by the close as of signal_date (last_day mode)."""
            frame = monthly_prices[cols].loc[:rebal_date].copy()
            if exec_timing == "last_day" and len(frame) > 0:
                asof = daily_ffill[cols].loc[:signal_date]
                if len(asof) > 0:
                    frame.iloc[-1] = asof.iloc[-1]
            return frame

        # Build basket
        basket = []

        for t in fixed_tickers:
            if t in monthly_prices.columns:
                ps = monthly_prices[t].loc[:rebal_date].dropna()
                if len(ps) > min_lookback:
                    # Bitcoin trend gate: skip the gated ticker on months
                    # when its Radius Trend signal is "out".
                    if t == btc_gate_ticker and not _gate_open_as_of(btc_gate, signal_date):
                        continue
                    # Bitcoin SMA cross gate: skip the gated ticker on months
                    # when its SMA cross signal is "out".
                    if t == btc_sma_gate_ticker and not _gate_open_as_of(btc_sma_gate, signal_date):
                        continue
                    basket.append(t)

        selected_equities = []
        if equity_top_n > 0 and len(equity_tickers) > 0:
            eq_available = []
            for t in equity_tickers:
                if t in monthly_prices.columns:
                    ps = monthly_prices[t].loc[:rebal_date].dropna()
                    # Fix 7: require enough history for BOTH equity and main lookbacks
                    min_eq_history = max(max(equity_lookbacks), min_lookback)
                    if len(ps) > min_eq_history:
                        eq_available.append(t)

            if len(eq_available) >= equity_top_n:
                eq_subset = signal_monthly_frame(eq_available)
                eq_mom = compute_blended_momentum(eq_subset, equity_lookbacks, equity_lookback_weights)
                latest_eq = eq_mom.iloc[-1].dropna().sort_values(ascending=False)
                selected_equities = latest_eq.head(equity_top_n).index.tolist()

            basket.extend(selected_equities)

        if len(basket) == 0:
            continue

        basket_monthly = signal_monthly_frame(basket)
        basket_daily = daily_prices[basket].copy()

        basket_mom = compute_blended_momentum(basket_monthly, main_lookbacks, main_lookback_weights)
        latest_basket = basket_mom.iloc[-1].dropna().sort_values(ascending=False)

        n_select = min(final_top_n, len(latest_basket))
        final_holdings = latest_basket.head(n_select).index.tolist()

        if len(final_holdings) == 0:
            continue

        weights = compute_risk_parity_weights(
            basket_daily, final_holdings, signal_date, vol_months
        )

        # Exit execution day (entry was resolved at the top of the loop):
        #   "last_day"  -> close of the final trading day of the month
        #   "first_day" -> close of the next trading day after month-end
        exec_date_exit = resolve_exec_date(next_month_end)

        if exec_date_exit is None:
            continue

        port_return = 0.0
        holding_details = {}

        for t in final_holdings:
            # Get execution-day closing prices from daily data
            entry_prices = daily_prices[t].loc[:exec_date_entry].dropna()
            exit_prices = daily_prices[t].loc[:exec_date_exit].dropna()

            if len(entry_prices) == 0 or len(exit_prices) == 0:
                continue

            p0 = entry_prices.iloc[-1]
            p1 = exit_prices.iloc[-1]

            if p0 == 0:
                continue

            asset_ret = (p1 / p0) - 1.0
            w = weights.get(t, 0.0)
            port_return += w * asset_ret
            holding_details[t] = {"weight": w, "return": asset_ret}

        # --- Dynamic leverage ---
        lev_mult, lev_regime, lev_detail = compute_leverage_signal(
            daily_prices, signal_date, leverage_params,
            final_holdings=final_holdings
        )
        port_return_raw = port_return
        port_return = port_return * lev_mult

        # Borrowing cost on leveraged portion (only when actually leveraged)
        lev_cost_annual = leverage_params.get("leverage_cost_annual", 0.0)
        monthly_borrow_cost = 0.0
        if lev_mult > 1.0 and lev_cost_annual > 0:
            monthly_borrow_cost = (lev_mult - 1.0) * (lev_cost_annual / 12.0)
            port_return -= monthly_borrow_cost

        # --- Daily equity tracking for this holding period ---
        holding_mask = (
            (daily_prices.index >= exec_date_entry)
            & (daily_prices.index <= exec_date_exit)
        )
        holding_days = daily_prices.index[holding_mask]
        n_holding_days = len(holding_days)

        # Entry prices for each held asset (computed once per period)
        entry_px = {}
        for t in final_holdings:
            ep = daily_prices[t].loc[:exec_date_entry].dropna()
            if len(ep) > 0 and ep.iloc[-1] != 0:
                entry_px[t] = ep.iloc[-1]

        for day_idx, day in enumerate(holding_days):
            day_weighted_return = 0.0
            for t in final_holdings:
                if t not in entry_px:
                    continue
                dp = daily_prices[t].loc[:day].dropna()
                if len(dp) == 0:
                    continue
                day_weighted_return += weights.get(t, 0.0) * (
                    (dp.iloc[-1] / entry_px[t]) - 1.0
                )
            # Accrue borrowing cost pro-rata through the holding period
            accrued_cost = 0.0
            if n_holding_days > 0 and monthly_borrow_cost > 0:
                accrued_cost = monthly_borrow_cost * (day_idx + 1) / n_holding_days
            daily_portfolio_values[day] = (
                cumulative_multiplier * (1 + day_weighted_return * lev_mult - accrued_cost) * INITIAL_CAPITAL
            )

        # Update cumulative multiplier at period end
        cumulative_multiplier *= (1 + port_return)

        results.append({
            "date": next_month_end,
            "portfolio_return": port_return,
            "leverage_mult": lev_mult,
            "borrow_cost": monthly_borrow_cost,
        })

        holdings_str_parts = []
        for t in final_holdings:
            if t in holding_details:
                w = weights.get(t, 0.0)
                r = holding_details[t]["return"]
                holdings_str_parts.append(f"{t}: {w:.1%} (ret: {r:+.2%})")

        # Bitcoin gate status for this month (for the trade log)
        if btc_gate_ticker is None:
            btc_gate_status = "off"
        elif btc_gate_ticker not in monthly_prices.columns or len(
            monthly_prices[btc_gate_ticker].loc[:rebal_date].dropna()
        ) <= min_lookback:
            btc_gate_status = "n/a"
        else:
            btc_gate_status = "in" if _gate_open_as_of(btc_gate, signal_date) else "out"

        # Bitcoin SMA cross gate status for this month (for the trade log)
        if btc_sma_gate_ticker is None:
            btc_sma_gate_status = "off"
        elif btc_sma_gate_ticker not in monthly_prices.columns or len(
            monthly_prices[btc_sma_gate_ticker].loc[:rebal_date].dropna()
        ) <= min_lookback:
            btc_sma_gate_status = "n/a"
        else:
            btc_sma_gate_status = "in" if _gate_open_as_of(btc_sma_gate, signal_date) else "out"

        trade_log.append({
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "hold_month": next_month_end.strftime("%Y-%m"),
            "exec_entry": exec_date_entry.strftime("%Y-%m-%d"),
            "exec_exit": exec_date_exit.strftime("%Y-%m-%d") if i < len(valid_months) - 2 else "",
            "basket": ", ".join(basket),
            "equities_selected": ", ".join(selected_equities) if selected_equities else "N/A",
            "btc_gate": btc_gate_status,
            "btc_sma_gate": btc_sma_gate_status,
            "final_holdings": " | ".join(holdings_str_parts),
            "leverage": f"{lev_mult:.2f}x",
            "lev_regime": lev_regime,
            "portfolio_return": f"{port_return:+.2%}",
        })

    results_df = pd.DataFrame(results).set_index("date")
    trade_df = pd.DataFrame(trade_log)

    if len(results_df) > 0:
        results_df = results_df[results_df.index >= backtest_start]
        trade_df = trade_df[trade_df["hold_month"] >= backtest_start].reset_index(drop=True)

        if backtest_end:
            results_df = results_df[results_df.index <= backtest_end]
            trade_df = trade_df[trade_df["hold_month"] <= backtest_end].reset_index(drop=True)

    # Build daily equity curve for daily drawdown calculation
    if daily_portfolio_values:
        daily_equity = pd.Series(daily_portfolio_values).sort_index()
        if backtest_start:
            daily_equity = daily_equity[daily_equity.index >= backtest_start]
        if backtest_end:
            daily_equity = daily_equity[daily_equity.index <= backtest_end]
    else:
        daily_equity = pd.Series(dtype=float)

    return results_df, trade_df, daily_equity


def _compute_signal_at(daily_prices, monthly_prices_local, params, signal_date,
                       full_details=True):
    """
    Compute the model signal as of a specific date. Internal helper used by
    compute_current_signals to compute both the live signal (using the most
    recent daily bar) and reference signals (e.g., the previous month-end).

    When full_details=False, the asset_details table is skipped and only the
    holdings/weights/equity picks are returned. Used for lightweight reference
    signals where the full ranking table isn't needed.
    """
    fixed_tickers = params["fixed_tickers"]
    equity_tickers = params["equity_tickers"]
    equity_top_n = params["equity_top_n"]
    equity_lookbacks = params["equity_lookbacks"]
    equity_lookback_weights = params["equity_lookback_weights"]
    main_lookbacks = params["main_lookbacks"]
    main_lookback_weights = params["main_lookback_weights"]
    final_top_n = params["final_top_n"]
    vol_months = params["vol_months"]

    btc_gate = params.get("btc_gate", {})
    btc_gate_ticker = btc_gate.get("ticker") if btc_gate.get("enabled") else None

    btc_sma_gate = params.get("btc_sma_gate", {})
    btc_sma_gate_ticker = btc_sma_gate.get("ticker") if btc_sma_gate.get("enabled") else None

    min_lookback = max(main_lookbacks)

    # Build basket
    basket = []
    for t in fixed_tickers:
        if t in monthly_prices_local.columns:
            ps = monthly_prices_local[t].loc[:signal_date].dropna()
            if len(ps) > min_lookback:
                if t == btc_gate_ticker and not _gate_open_as_of(btc_gate, signal_date):
                    continue
                if t == btc_sma_gate_ticker and not _gate_open_as_of(btc_sma_gate, signal_date):
                    continue
                basket.append(t)

    selected_equities = []
    if equity_top_n > 0 and len(equity_tickers) > 0:
        eq_available = []
        for t in equity_tickers:
            if t in monthly_prices_local.columns:
                ps = monthly_prices_local[t].loc[:signal_date].dropna()
                min_eq_history = max(max(equity_lookbacks), min_lookback)
                if len(ps) > min_eq_history:
                    eq_available.append(t)

        if len(eq_available) >= equity_top_n:
            eq_subset = monthly_prices_local[eq_available].loc[:signal_date]
            eq_mom = compute_blended_momentum(
                eq_subset, equity_lookbacks, equity_lookback_weights
            )
            latest_eq = eq_mom.iloc[-1].dropna().sort_values(ascending=False)
            selected_equities = latest_eq.head(equity_top_n).index.tolist()

        basket.extend(selected_equities)

    if len(basket) == 0:
        return None

    asset_details = []
    if full_details:
        all_assets = list(set(fixed_tickers + equity_tickers))
        all_assets = [t for t in all_assets if t in monthly_prices_local.columns]

        for t in all_assets:
            ps = monthly_prices_local[t].loc[:signal_date].dropna()
            if len(ps) < min_lookback + 1:
                continue

            row = {"Asset": t, "In Basket": t in basket}

            for lb in sorted(set(main_lookbacks)):
                ret = compute_total_return(
                    monthly_prices_local[[t]].loc[:signal_date], lb
                )
                val = ret.iloc[-1].values[0] if len(ret) > 0 else np.nan
                row[f"{lb}mo Return"] = val

            basket_sub = monthly_prices_local[[t]].loc[:signal_date]
            blended = compute_blended_momentum(
                basket_sub, main_lookbacks, main_lookback_weights
            )
            blended_val = blended.iloc[-1].values[0] if len(blended) > 0 else np.nan
            row["Blended Score"] = blended_val

            lookback_start = signal_date - pd.DateOffset(months=vol_months)
            vol_subset = daily_prices[t].loc[lookback_start:signal_date].dropna()
            if len(vol_subset) > 20:
                daily_ret = vol_subset.pct_change().dropna()
                ann_vol = daily_ret.std() * np.sqrt(252)
                row[f"{vol_months}mo Vol (ann)"] = ann_vol
            else:
                row[f"{vol_months}mo Vol (ann)"] = np.nan

            asset_details.append(row)

    # Rank basket by blended momentum
    basket_monthly = monthly_prices_local[basket].loc[:signal_date].copy()
    basket_mom = compute_blended_momentum(
        basket_monthly, main_lookbacks, main_lookback_weights
    )
    latest_basket = basket_mom.iloc[-1].dropna().sort_values(ascending=False)

    n_select = min(final_top_n, len(latest_basket))
    final_holdings = latest_basket.head(n_select).index.tolist()

    weights = compute_risk_parity_weights(
        daily_prices, final_holdings, signal_date, vol_months
    )

    if full_details:
        for row in asset_details:
            t = row["Asset"]
            row["Selected"] = t in final_holdings
            row["Weight"] = weights.get(t, 0.0) if t in final_holdings else 0.0
            row["Equity Pick"] = t in selected_equities

        asset_details.sort(key=lambda x: x.get("Blended Score", -999), reverse=True)

    return {
        "signal_date": signal_date,
        "asset_details": asset_details,
        "selected_equities": selected_equities,
        "final_holdings": final_holdings,
        "weights": weights,
        "basket": basket,
    }


def compute_equity_subselection_details(daily_prices, monthly_prices_local,
                                        params, signal_date):
    """
    Build the equity sub-selection ranking table as of signal_date.

    The main "Current Model Signals" table ranks every basket asset by the
    MAIN momentum lookback. Equities, however, are pre-filtered by a SEPARATE
    equity momentum lookback before they ever reach the main basket: only the
    top equity_top_n equity ETFs (ranked by the equity blend) are admitted.

    This table exposes that pre-filter, so it is clear why an equity ETF that
    looks strong on the main lookback (and would therefore rank high in the
    signals table) can still be left out: it lost the equity-lookback contest.

    Columns mirror the signals table minus Weight, but the per-period return
    columns and the Blended score use the EQUITY lookbacks, not the main ones.

    Returns a list of row dicts sorted by equity blended momentum (descending),
    or [] when equity selection is disabled or there is not enough history. The
    selection logic here intentionally mirrors _compute_signal_at so the picks
    match the basket exactly.
    """
    equity_tickers = params["equity_tickers"]
    equity_top_n = params["equity_top_n"]
    equity_lookbacks = params["equity_lookbacks"]
    equity_lookback_weights = params["equity_lookback_weights"]
    main_lookbacks = params["main_lookbacks"]
    vol_months = params["vol_months"]

    if equity_top_n <= 0 or len(equity_tickers) == 0:
        return []

    min_lookback = max(main_lookbacks)
    min_eq_history = max(max(equity_lookbacks), min_lookback)

    # Equity ETFs with enough history to be eligible this period (same gate
    # as _compute_signal_at).
    eq_available = []
    for t in equity_tickers:
        if t in monthly_prices_local.columns:
            ps = monthly_prices_local[t].loc[:signal_date].dropna()
            if len(ps) > min_eq_history:
                eq_available.append(t)

    if len(eq_available) == 0:
        return []

    # Rank by the equity-lookback blend (the actual selection rule).
    eq_subset = monthly_prices_local[eq_available].loc[:signal_date]
    eq_mom = compute_blended_momentum(
        eq_subset, equity_lookbacks, equity_lookback_weights
    )
    latest_eq = eq_mom.iloc[-1].dropna().sort_values(ascending=False)
    selected_equities = latest_eq.head(equity_top_n).index.tolist()

    eq_lb_cols = sorted(set(equity_lookbacks))

    rows = []
    for t in latest_eq.index:
        row = {"Asset": t, "Equity Pick": t in selected_equities}

        for lb in eq_lb_cols:
            ret = compute_total_return(
                monthly_prices_local[[t]].loc[:signal_date], lb
            )
            val = ret.iloc[-1].values[0] if len(ret) > 0 else np.nan
            row[f"{lb}mo Return"] = val

        row["Blended Score"] = latest_eq.get(t, np.nan)

        lookback_start = signal_date - pd.DateOffset(months=vol_months)
        vol_subset = daily_prices[t].loc[lookback_start:signal_date].dropna()
        if len(vol_subset) > 20:
            daily_ret = vol_subset.pct_change().dropna()
            row[f"{vol_months}mo Vol (ann)"] = daily_ret.std() * np.sqrt(252)
        else:
            row[f"{vol_months}mo Vol (ann)"] = np.nan

        rows.append(row)

    return rows


def last_market_date(daily_prices):
    """
    Most recent date on which the traditional-market assets actually traded.

    daily_prices may contain 24/7 assets (BTC-USD and other "-USD" pairs).
    Those carry bars on weekends and market holidays, and Yahoo publishes
    the current UTC day's bar as soon as the UTC date rolls over, which is
    6:00 p.m. the previous day in Mountain time. Anchoring the live signal
    to daily_prices.index.max() therefore stamps it with a date that has
    not happened yet locally and on which no ETF has traded.

    Anchor to the exchange calendar instead: the last date carrying data
    for at least one non-crypto ticker.
    """
    if daily_prices.empty:
        return None
    dates = market_trading_dates(daily_prices)
    if len(dates) == 0:
        return None
    return dates.max()


def compute_current_signals(daily_prices, monthly_prices, params):
    """
    Compute model signals for the most recent available trading date.
    The table answers: "If I rebalanced today, what should I hold?"

    Returns a dict containing:
        signal_date           the most recent trading date (live rebalance moment)
        asset_details         full per-asset ranking table for the live signal
        selected_equities     equity ETFs selected this period
        final_holdings        list of tickers held in the live signal
        weights               dict of {ticker: weight} for the live signal
        basket                full basket of eligible tickers
        prev_month_end_date   the most recent COMPLETED calendar month-end
                              (None if no prior month-end is available)
        prev_holdings         list of tickers from the prev month-end signal
        prev_weights          dict of {ticker: weight} from the prev month-end signal
    """
    # ---- LIVE SIGNAL ----
    # Use the most recent daily bar as the signal date. If that date falls
    # inside an incomplete calendar month, the resampled monthly bin for
    # that month carries a future-dated label (e.g., "2026-04-30" while we
    # only have data through 2026-04-28). The bin already contains the
    # most recent close (forward-filled), so we just relabel its index to
    # last_daily so all date-based slicing matches the actual data window.
    last_daily = last_market_date(daily_prices)
    if last_daily is None:
        return None
    candidate = monthly_prices.index[-1]
    if candidate > last_daily:
        new_index = list(monthly_prices.index[:-1]) + [last_daily]
        effective_monthly = monthly_prices.copy()
        effective_monthly.index = pd.DatetimeIndex(new_index)
        signal_date = last_daily
    else:
        effective_monthly = monthly_prices
        signal_date = candidate

    live = _compute_signal_at(
        daily_prices, effective_monthly, params, signal_date, full_details=True
    )
    if live is None:
        return None

    # Equity sub-selection ranking (ranked by the equity lookback, not the
    # main lookback). Explains which equity ETFs win the pre-filter.
    equity_details = compute_equity_subselection_details(
        daily_prices, effective_monthly, params, signal_date
    )

    # ---- PREVIOUS MONTH-END SIGNAL (reference) ----
    # Find the most recent completed calendar month-end (a true month-end
    # in the original monthly_prices index that is at or before last_daily,
    # but not equal to the live signal date). This documents what the model
    # was holding heading into the current month for continuity context.
    prev_month_end_date = None
    prev_holdings = []
    prev_weights = {}
    completed = monthly_prices.index[monthly_prices.index <= last_daily]
    prev_candidates = [d for d in completed if d != signal_date]
    if len(prev_candidates) > 0:
        prev_month_end_date = prev_candidates[-1]
        prev = _compute_signal_at(
            daily_prices, monthly_prices, params, prev_month_end_date,
            full_details=False
        )
        if prev is not None:
            prev_holdings = prev["final_holdings"]
            prev_weights = prev["weights"]

    return {
        **live,
        "equity_details": equity_details,
        "prev_month_end_date": prev_month_end_date,
        "prev_holdings": prev_holdings,
        "prev_weights": prev_weights,
    }


def compute_metrics(monthly_returns, rf_monthly=None, daily_equity=None):
    """
    Compute performance metrics.

    Args:
        monthly_returns: Series of monthly portfolio returns
        rf_monthly: Optional Series of monthly risk-free returns (e.g. BIL)
                    aligned to the same index. Used for Sharpe and Sortino.
        daily_equity: Optional Series of daily portfolio values for daily max DD.
    """
    r = monthly_returns.dropna()
    if len(r) == 0:
        return {}

    cumulative = (1 + r).cumprod()
    total_return = cumulative.iloc[-1] - 1.0

    n_years = len(r) / 12.0
    cagr = (1 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    ann_vol = r.std() * np.sqrt(12)

    # Fix 5: Sharpe ratio with risk-free rate
    # Use BIL returns if available, otherwise assume 0
    if rf_monthly is not None:
        rf_aligned = rf_monthly.reindex(r.index).fillna(0)
        excess = r - rf_aligned
        rf_annual = (1 + rf_aligned).prod() ** (12.0 / len(rf_aligned)) - 1.0
    else:
        excess = r
        rf_annual = 0.0

    sharpe = (cagr - rf_annual) / ann_vol if ann_vol > 0 else 0.0

    # Fix 6: Sortino with proper downside deviation
    # Standard formula: sqrt(mean(min(excess, 0)^2)) * sqrt(12)
    excess_downside = np.minimum(excess.values, 0.0)
    downside_var = np.mean(excess_downside ** 2)
    downside_dev = np.sqrt(downside_var) * np.sqrt(12)
    sortino = (cagr - rf_annual) / downside_dev if downside_dev > 0 else 0.0

    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Daily max drawdown (if daily equity curve provided)
    if daily_equity is not None and len(daily_equity) > 0:
        daily_peak = daily_equity.cummax()
        daily_drawdown = (daily_equity - daily_peak) / daily_peak
        max_dd_daily = daily_drawdown.min()
        trough_date = daily_drawdown.idxmin()
        peak_date = daily_equity.loc[:trough_date].idxmax()
    else:
        max_dd_daily = np.nan
        trough_date = None
        peak_date = None

    # Worst single-day decline (close-to-close)
    if daily_equity is not None and len(daily_equity) > 1:
        daily_pct = daily_equity.pct_change().dropna()
        worst_day = daily_pct.min()
        worst_day_date = daily_pct.idxmin()
    else:
        worst_day = np.nan
        worst_day_date = None

    # Fix 8: Filter partial years for best/worst year
    r_copy = r.copy()
    r_copy.index = pd.to_datetime(r_copy.index)
    yearly = (1 + r_copy).resample("YE").prod() - 1
    months_per_year = r_copy.resample("YE").count()
    # Only include years with at least 6 months of data
    full_years = yearly[months_per_year >= 6]
    best_year = full_years.max() if len(full_years) > 0 else np.nan
    worst_year = full_years.min() if len(full_years) > 0 else np.nan

    win_rate = (r > 0).sum() / len(r)

    return {
        "Period": f"{r.index[0].strftime('%Y-%m')} to {r.index[-1].strftime('%Y-%m')}",
        "Total Months": len(r),
        "CAGR": cagr,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Max Drawdown (Daily)": max_dd_daily,
        "DD Peak Date": peak_date,
        "DD Trough Date": trough_date,
        "Worst Day": worst_day,
        "Worst Day Date": worst_day_date,
        "Calmar Ratio": calmar,
        "Best Year": best_year,
        "Worst Year": worst_year,
        "Win Rate (monthly)": win_rate,
        "Total Return": total_return,
        f"Growth of ${INITIAL_CAPITAL:,}": INITIAL_CAPITAL * (1 + total_return),
    }


# =============================================================================
# EXCEL LOGGING
# =============================================================================

def append_to_excel(params, strat_metrics, bench_metrics, filepath):
    lev = params.get("leverage_params", {})
    lev_summary = "Off"
    if lev.get("enabled"):
        if lev.get("method") == "constant":
            cost_pct = lev.get("leverage_cost_annual", 0.0) * 100
            lev_summary = f"Constant: {lev.get('mult_constant', 1.25):.2f}x, cost={cost_pct:.1f}%"
        elif lev.get("method") == "holdings_gate":
            defensives = ", ".join(lev.get("defensive_tickers", []))
            cost_pct = lev.get("leverage_cost_annual", 0.0) * 100
            lev_summary = (
                f"Holdings gate: defensives={defensives}, "
                f"mult={lev.get('mult_leveraged', 1.25):.2f}x, "
                f"cost={cost_pct:.1f}%"
            )
        elif lev.get("method") == "vt_trend":
            lev_summary = (
                f"VT trend: {lev['ticker']} {lev['long_sma']}/{lev['short_sma']}d SMA, "
                f"slope {lev['slope_lookback']}d, "
                f"on={lev['mult_risk_on']:.2f}/off={lev['mult_risk_off']:.2f}"
            )

    bg = params.get("btc_gate", {})
    if bg.get("enabled"):
        bg_mode = "long-position" if bg.get("mode") == "position" else "trend-up"
        bg_summary = (
            f"{gate_label(bg)} Radius Trend ({bg_mode}), "
            f"step={bg.get('step', 0.15)}, dist={bg.get('multi', 0.5)}"
        )
    else:
        bg_summary = "Off"

    sg = params.get("btc_sma_gate", {})
    if sg.get("enabled"):
        sg_mode = "long-position" if sg.get("mode") == "position" else "trend-up"
        sg_summary = (
            f"{gate_label(sg)} SMA cross ({sg_mode}), "
            f"SMA={sg.get('sma_len', 150)}, rf_len={sg.get('rf_len', 20)}, "
            f"rf_period={sg.get('rf_period', 20)}"
        )
    else:
        sg_summary = "Off"

    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fixed Assets": ", ".join(params["fixed_tickers"]),
        "Backfills": params.get("backfill_summary", "None"),
        "Equity Universe": ", ".join(params["equity_tickers"]),
        "Equity Top N": params["equity_top_n"],
        "Equity Lookback": params["equity_lookback_label"],
        "Main Lookback": params["main_lookback_label"],
        "Final Top N": params["final_top_n"],
        "Vol Window (months)": params["vol_months"],
        "Execution": params.get("exec_timing_label", params.get("exec_timing", "last_day")),
        "Leverage": lev_summary,
        "BTC Gate": bg_summary,
        "BTC SMA Gate": sg_summary,
        "Backtest Start": params["backtest_start"],
        "Backtest End": params.get("backtest_end", ""),
        "Period": strat_metrics.get("Period", ""),
        "CAGR": strat_metrics.get("CAGR", 0),
        "Ann. Vol": strat_metrics.get("Annualized Vol", 0),
        "Sharpe": strat_metrics.get("Sharpe Ratio", 0),
        "Sortino": strat_metrics.get("Sortino Ratio", 0),
        "Max Drawdown": strat_metrics.get("Max Drawdown", 0),
        "Max DD (Daily)": strat_metrics.get("Max Drawdown (Daily)", 0),
        "Calmar": strat_metrics.get("Calmar Ratio", 0),
        "Best Year": strat_metrics.get("Best Year", 0),
        "Worst Year": strat_metrics.get("Worst Year", 0),
        "Win Rate": strat_metrics.get("Win Rate (monthly)", 0),
        "Total Return": strat_metrics.get("Total Return", 0),
        "SPY CAGR": bench_metrics.get("CAGR", 0),
        "SPY Max DD": bench_metrics.get("Max Drawdown", 0),
        "SPY Max DD (Daily)": bench_metrics.get("Max Drawdown (Daily)", 0),
        "SPY Sharpe": bench_metrics.get("Sharpe Ratio", 0),
    }

    new_row_df = pd.DataFrame([row])

    if os.path.exists(filepath):
        existing = pd.read_excel(filepath, engine="openpyxl")
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_excel(filepath, index=False, engine="openpyxl")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_equity_and_drawdown(strat_returns, bench_returns, balanced_data=None):
    common_idx = strat_returns.index.intersection(bench_returns.index)
    strat = strat_returns.loc[common_idx]
    bench = bench_returns.loc[common_idx]

    strat_cum = INITIAL_CAPITAL * (1 + strat).cumprod()
    bench_cum = INITIAL_CAPITAL * (1 + bench).cumprod()

    # Build cumulative series for each balanced benchmark that has data
    bal_plots = []
    if balanced_data:
        for entry in balanced_data:
            bal = entry["returns"].reindex(common_idx).dropna()
            if len(bal) > 0:
                bal_plots.append({**entry, "cum": INITIAL_CAPITAL * (1 + bal).cumprod()})

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    title_parts = ["Adaptive Strategy", "SPY"] + [e["label"] for e in bal_plots]
    fig.suptitle("Adaptive Strategy vs. " + " vs. ".join(title_parts[1:]),
                 fontsize=14, fontweight="bold", y=0.97)

    ax1 = axes[0]
    ax1.plot(strat_cum.index, strat_cum.values, label="Strategy", linewidth=2.8, color=BRAND_PURPLE)
    ax1.plot(bench_cum.index, bench_cum.values, label="SPY", linewidth=2.0, color=BRAND_INDIGO, alpha=0.85)
    for entry in bal_plots:
        ax1.plot(
            entry["cum"].index, entry["cum"].values,
            label=entry["label"], linewidth=1.6,
            color=entry["color"], linestyle=entry["linestyle"], alpha=0.75,
        )
    ax1.set_ylabel(f"Growth of ${INITIAL_CAPITAL:,}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    dollar_fmt = mtick.FuncFormatter(lambda x, _: f"${x:,.0f}")
    ax1.yaxis.set_major_formatter(dollar_fmt)
    ax1.yaxis.set_minor_formatter(dollar_fmt)

    ax2 = axes[1]
    wealth = (1 + strat).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    ax2.plot(dd.index, dd.values, color=BRAND_PURPLE, linewidth=1.8, label="Strategy")

    wealth_b = (1 + bench).cumprod()
    peak_b = wealth_b.cummax()
    dd_b = (wealth_b - peak_b) / peak_b
    ax2.plot(dd_b.index, dd_b.values, color=BRAND_INDIGO, linewidth=1.4, label="SPY")

    for entry in bal_plots:
        bal = entry["returns"].reindex(common_idx).dropna()
        if len(bal) == 0:
            continue
        wealth_bal = (1 + bal).cumprod()
        peak_bal = wealth_bal.cummax()
        dd_bal = (wealth_bal - peak_bal) / peak_bal
        ax2.plot(
            dd_bal.index, dd_bal.values,
            color=entry["color"], linewidth=1.4,
            linestyle=entry["linestyle"], label=entry["label"],
        )

    ax2.set_ylabel("Drawdown")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    return fig


def plot_daily_drawdown(daily_equity, daily_equity_bench=None, balanced_data=None):
    """Plot daily drawdown chart from daily equity curves."""
    if daily_equity is None or len(daily_equity) == 0:
        return None

    peak = daily_equity.cummax()
    dd = (daily_equity - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Daily Drawdown (close-to-close)", fontsize=14, fontweight="bold")
    ax.fill_between(dd.index, dd.values, 0, color=BRAND_PURPLE, alpha=0.18)
    ax.plot(dd.index, dd.values, color=BRAND_PURPLE, linewidth=1.8, label="Strategy")

    if daily_equity_bench is not None and len(daily_equity_bench) > 0:
        peak_b = daily_equity_bench.cummax()
        dd_b = (daily_equity_bench - peak_b) / peak_b
        ax.plot(dd_b.index, dd_b.values, color=BRAND_INDIGO, linewidth=1.4, label="SPY")

    if balanced_data:
        for entry in balanced_data:
            de = entry.get("daily_equity")
            if de is None or len(de) == 0:
                continue
            peak_bal = de.cummax()
            dd_bal = (de - peak_bal) / peak_bal
            ax.plot(
                dd_bal.index, dd_bal.values,
                color=entry["color"], linewidth=1.4,
                linestyle=entry["linestyle"], label=entry["label"],
            )

    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    return fig


def plot_annual_returns(strat_returns, bench_returns):
    """Return a matplotlib figure with side-by-side annual return bars."""
    common_idx = strat_returns.index.intersection(bench_returns.index)
    strat = strat_returns.loc[common_idx].copy()
    bench = bench_returns.loc[common_idx].copy()

    strat.index = pd.to_datetime(strat.index)
    bench.index = pd.to_datetime(bench.index)

    strat_yearly = (1 + strat).resample("YE").prod() - 1
    bench_yearly = (1 + bench).resample("YE").prod() - 1

    years = [d.year for d in strat_yearly.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Annual Returns", fontsize=14, fontweight="bold")

    bar_width = 0.35
    x = np.arange(len(years))

    bars_strat = ax.bar(
        x - bar_width / 2, strat_yearly.values, bar_width,
        label="Strategy", color=BRAND_PURPLE, edgecolor="white", linewidth=0.7
    )
    bars_bench = ax.bar(
        x + bar_width / 2, bench_yearly.values, bar_width,
        label="SPY", color=BRAND_INDIGO, edgecolor="white", linewidth=0.7,
        hatch="///"
    )

    # Add value labels on each bar
    for bar in bars_strat:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{h:.1%}", ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=10, fontweight="bold", color=BRAND_PURPLE
        )
    for bar in bars_bench:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{h:.1%}", ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=10, fontweight="bold", color=BRAND_INDIGO
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    return fig


# =============================================================================
# STRATEGY HEALTH DIAGNOSTICS
# =============================================================================

def compute_rolling_sortino(returns, window, rf_returns=None):
    """Compute rolling annualized Sortino ratio over a trailing window."""
    rolling_sortino = pd.Series(dtype=float, index=returns.index)
    for i in range(window, len(returns) + 1):
        chunk = returns.iloc[i - window:i]
        if rf_returns is not None:
            rf_chunk = rf_returns.reindex(chunk.index).fillna(0)
            excess = chunk - rf_chunk
            rf_ann = (1 + rf_chunk).prod() ** (12.0 / len(rf_chunk)) - 1.0
        else:
            excess = chunk
            rf_ann = 0.0

        total_ret = (1 + chunk).prod() - 1.0
        n_years = len(chunk) / 12.0
        cagr = (1 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        downside = np.minimum(excess.values, 0.0)
        dd_var = np.mean(downside ** 2)
        dd_dev = np.sqrt(dd_var) * np.sqrt(12)

        if dd_dev > 0:
            rolling_sortino.iloc[i - 1] = (cagr - rf_ann) / dd_dev
        else:
            rolling_sortino.iloc[i - 1] = np.nan

    return rolling_sortino


def compute_rolling_calmar(monthly_returns, daily_equity, window_months):
    """
    Compute rolling Calmar ratio (CAGR / |max drawdown|) over a trailing window.

    Uses the daily equity curve for max drawdown when available, which matches
    the dashboard's headline 'Max DD (Daily)' metric. Falls back to monthly
    drawdown if no daily equity is provided. Returns NaN before the window
    is filled and during periods with no observed drawdown.
    """
    rolling = pd.Series(dtype=float, index=monthly_returns.index)
    use_daily = daily_equity is not None and len(daily_equity) > 0

    for i in range(window_months, len(monthly_returns) + 1):
        chunk = monthly_returns.iloc[i - window_months:i]

        # CAGR over the window
        total_ret = (1 + chunk).prod() - 1.0
        n_years = len(chunk) / 12.0
        cagr = (1 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        # Max drawdown over the same window (daily preferred, monthly fallback)
        max_dd = 0.0
        if use_daily:
            window_daily = daily_equity.loc[chunk.index[0]:chunk.index[-1]]
            if len(window_daily) > 1:
                peak = window_daily.cummax()
                dd_series = (window_daily - peak) / peak
                max_dd = abs(dd_series.min())

        if max_dd == 0.0:
            wealth = (1 + chunk).cumprod()
            peak = wealth.cummax()
            dd_series = (wealth - peak) / peak
            max_dd = abs(dd_series.min())

        if max_dd > 0:
            rolling.iloc[i - 1] = cagr / max_dd
        else:
            rolling.iloc[i - 1] = np.nan

    return rolling


def compute_rolling_cagr(returns, window):
    """Compute rolling annualized return over a trailing window."""
    rolling = pd.Series(dtype=float, index=returns.index)
    for i in range(window, len(returns) + 1):
        chunk = returns.iloc[i - window:i]
        total_ret = (1 + chunk).prod() - 1.0
        n_years = len(chunk) / 12.0
        cagr = (1 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
        rolling.iloc[i - 1] = cagr
    return rolling


def compute_rolling_hit_rate(returns, window):
    """Compute rolling percentage of positive months."""
    return returns.rolling(window).apply(lambda x: (x > 0).sum() / len(x), raw=True)


def compute_underwater(returns):
    """Compute time underwater (months since last equity high) and drawdown."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak

    underwater = pd.Series(0, index=returns.index)
    count = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            count += 1
        else:
            count = 0
        underwater.iloc[i] = count

    return drawdown, underwater


def plot_rolling_sortino(strat_returns, bench_returns, rf_monthly=None):
    """Plot rolling Sortino at 36mo (solid) and 24mo (faded)."""
    strat_36 = compute_rolling_sortino(strat_returns, 36, rf_monthly)
    bench_36 = compute_rolling_sortino(bench_returns, 36, rf_monthly)
    strat_24 = compute_rolling_sortino(strat_returns, 24, rf_monthly)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Rolling Sortino Ratio", fontsize=14, fontweight="bold")

    ax.plot(strat_36.index, strat_36.values, label="Strategy (36mo)", linewidth=2.0, color=BRAND_PURPLE)
    ax.plot(strat_24.index, strat_24.values, label="Strategy (24mo)", linewidth=1.0, color=BRAND_PURPLE_FADE, alpha=0.5)
    ax.plot(bench_36.index, bench_36.values, label="SPY (36mo)", linewidth=1.8, color=BRAND_INDIGO, linestyle="--")

    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.axhline(1.0, color=BRAND_GRAY, linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylabel("Sortino Ratio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rolling_calmar(strat_returns, bench_returns,
                        daily_equity_strat=None, daily_equity_bench=None):
    """
    Plot rolling Calmar ratio (CAGR / |max drawdown|).

    Primary line: 60-month window (Calmar is fundamentally about tail events,
    and a 36-month window often hasn't seen a real drawdown yet).
    Faded line: 36-month window for shorter-term trend.
    SPY shown at 60-month for comparison.

    Y-axis is capped at 0 to 5 for visual clarity. Values exceeding the cap
    are annotated above the chart edge so calm periods don't blow out the scale.
    """
    strat_60 = compute_rolling_calmar(strat_returns, daily_equity_strat, 60)
    strat_36 = compute_rolling_calmar(strat_returns, daily_equity_strat, 36)
    bench_60 = compute_rolling_calmar(bench_returns, daily_equity_bench, 60)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Rolling Calmar Ratio (CAGR / Max Drawdown)",
                 fontsize=14, fontweight="bold")

    ax.plot(strat_60.index, strat_60.values, label="Strategy (60mo)",
            linewidth=2.0, color=BRAND_PURPLE)
    ax.plot(strat_36.index, strat_36.values, label="Strategy (36mo)",
            linewidth=1.0, color=BRAND_PURPLE_FADE, alpha=0.5)
    ax.plot(bench_60.index, bench_60.values, label="SPY (60mo)",
            linewidth=1.8, color=BRAND_INDIGO, linestyle="--")

    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.axhline(1.0, color=BRAND_GRAY, linewidth=0.5, linestyle=":", alpha=0.5)

    # Cap y-axis. Annotate any windows that exceed the cap.
    y_min, y_max = 0, 5
    ax.set_ylim(y_min, y_max)

    for series, color in [
        (strat_60, BRAND_PURPLE),
        (strat_36, BRAND_PURPLE_FADE),
        (bench_60, BRAND_INDIGO),
    ]:
        spikes = series[series > y_max].dropna()
        for date, val in spikes.items():
            ax.annotate(
                f"{val:.1f}", xy=(date, y_max), xytext=(0, 3),
                textcoords="offset points", ha="center",
                fontsize=7, color=color, alpha=0.85,
            )

    ax.set_ylabel("Calmar Ratio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rolling_cagr(strat_returns, bench_returns):
    """Plot 36-month rolling annualized return."""
    strat_36 = compute_rolling_cagr(strat_returns, 36)
    bench_36 = compute_rolling_cagr(bench_returns, 36)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Rolling 36-Month CAGR", fontsize=14, fontweight="bold")

    ax.plot(strat_36.index, strat_36.values, label="Strategy", linewidth=2.0, color=BRAND_PURPLE)
    ax.plot(bench_36.index, bench_36.values, label="SPY", linewidth=1.8, color=BRAND_INDIGO, linestyle="--")

    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.set_ylabel("Annualized Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rolling_hit_rate(strat_returns, bench_returns):
    """Plot 36-month rolling win rate (% of positive months)."""
    strat_hr = compute_rolling_hit_rate(strat_returns, 36)
    bench_hr = compute_rolling_hit_rate(bench_returns, 36)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Rolling 36-Month Hit Rate (% Positive Months)", fontsize=14, fontweight="bold")

    ax.plot(strat_hr.index, strat_hr.values, label="Strategy", linewidth=2.0, color=BRAND_PURPLE)
    ax.plot(bench_hr.index, bench_hr.values, label="SPY", linewidth=1.8, color=BRAND_INDIGO, linestyle="--")

    ax.axhline(0.5, color=BRAND_GRAY, linewidth=0.8, linestyle="--", alpha=0.5, label="50% line")
    ax.set_ylabel("Win Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_underwater(strat_returns, bench_returns):
    """Plot drawdown duration (months underwater) for strategy and benchmark."""
    dd_strat, uw_strat = compute_underwater(strat_returns)
    dd_bench, uw_bench = compute_underwater(bench_returns)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("Time Underwater", fontsize=14, fontweight="bold", y=0.97)

    ax1 = axes[0]
    ax1.bar(uw_strat.index, uw_strat.values, width=25, color=BRAND_PURPLE, alpha=0.7, label="Strategy")
    ax1.bar(uw_bench.index, uw_bench.values, width=25, color=BRAND_INDIGO, alpha=0.35, label="SPY")
    ax1.set_ylabel("Months Since\nEquity High")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(dd_strat.index, dd_strat.values, color=BRAND_PURPLE, linewidth=1.8, label="Strategy")
    ax2.plot(dd_bench.index, dd_bench.values, color=BRAND_INDIGO, linewidth=1.4, label="SPY")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rolling_excess(strat_returns, bench_returns):
    """Plot 36-month rolling excess annualized return vs benchmark."""
    strat_36 = compute_rolling_cagr(strat_returns, 36)
    bench_36 = compute_rolling_cagr(bench_returns, 36)
    excess = strat_36 - bench_36

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Rolling 36-Month Excess Return vs. SPY", fontsize=14, fontweight="bold")

    ax.fill_between(excess.index, excess.values, 0,
                    where=(excess.values >= 0), color=BRAND_TEAL, alpha=0.4, interpolate=True)
    ax.fill_between(excess.index, excess.values, 0,
                    where=(excess.values < 0), color=BRAND_RED, alpha=0.4, interpolate=True)
    ax.plot(excess.index, excess.values, color=BRAND_TEXT, linewidth=1.2)

    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.set_ylabel("Excess Annualized Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_leverage_timeline(trade_df, leverage_params):
    """Plot the leverage multiplier over time from the trade log."""
    if trade_df.empty or "leverage" not in trade_df.columns:
        return None
    if not leverage_params.get("enabled", False):
        return None
    # No chart needed for constant leverage (it's a flat line)
    if leverage_params.get("method") == "constant":
        return None

    df = trade_df.copy()
    df["date"] = pd.to_datetime(df["hold_month"])
    df["lev_val"] = df["leverage"].str.replace("x", "").astype(float)

    method = leverage_params.get("method", "holdings_gate")

    fig, ax = plt.subplots(figsize=(12, 3.5))

    if method == "holdings_gate":
        defensives = ", ".join(leverage_params.get("defensive_tickers", ["BIL", "BTAL"]))
        fig.suptitle(
            f"Dynamic Leverage (Holdings Gate: {defensives})",
            fontsize=14, fontweight="bold"
        )
        colors = []
        for regime in df["lev_regime"]:
            if regime == "leveraged":
                colors.append(BRAND_TEAL)
            else:
                colors.append(BRAND_GRAY)
    else:
        ticker = leverage_params.get("ticker", "VT")
        long_sma = leverage_params.get("long_sma", 200)
        short_sma = leverage_params.get("short_sma", 50)
        fig.suptitle(
            f"Dynamic Leverage ({ticker} {long_sma}/{short_sma}d SMA Trend)",
            fontsize=14, fontweight="bold"
        )
        colors = []
        for regime in df["lev_regime"]:
            if regime == "risk-on":
                colors.append(BRAND_TEAL)
            elif regime == "risk-off":
                colors.append(BRAND_RED)
            else:
                colors.append(BRAND_GRAY)

    ax.bar(df["date"], df["lev_val"], width=25, color=colors, alpha=0.7, edgecolor="none")
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Leverage Multiplier")
    ax.set_ylim(0, max(df["lev_val"].max() * 1.15, 1.5))
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    if method == "holdings_gate":
        legend_elements = [
            Patch(facecolor=BRAND_TEAL, alpha=0.7, label="Leveraged (no defensives)"),
            Patch(facecolor=BRAND_GRAY, alpha=0.7, label="Neutral (defensives held)"),
        ]
    else:
        legend_elements = [
            Patch(facecolor=BRAND_TEAL, alpha=0.7, label="Risk-on"),
            Patch(facecolor=BRAND_GRAY, alpha=0.7, label="Neutral"),
            Patch(facecolor=BRAND_RED, alpha=0.7, label="Risk-off"),
        ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    return fig


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

st.sidebar.title("Backtest Parameters")

# ---- MAIN STRATEGY (first position) ----
st.sidebar.header("Main Strategy")

main_lookback_label = st.sidebar.selectbox(
    "Main momentum lookback (for final ranking)",
    options=list(MAIN_LOOKBACK_OPTIONS.keys()),
    index=3,
)
main_lookbacks, main_lookback_weights = MAIN_LOOKBACK_OPTIONS[main_lookback_label]

final_top_n = st.sidebar.radio(
    "How many assets to hold in the final portfolio?",
    options=[3, 4, 5, 6],
    index=1,
    horizontal=True,
)

EXEC_TIMING_OPTIONS = {
    "Last trading day of month (at close)": "last_day",
    "First trading day of next month (at close)": "first_day",
}
exec_timing_label = st.sidebar.selectbox(
    "Trade execution timing",
    options=list(EXEC_TIMING_OPTIONS.keys()),
    index=0,
    help=(
        "Last trading day: allocations are computed from the close of the "
        "trading day BEFORE the last trading day of the month, and trades "
        "execute at the last trading day's close. Matches real-world "
        "execution (signal from the penultimate close, enter near the "
        "final close) with no look-ahead. "
        "First trading day: signals use the month-end close and trades "
        "execute at the next trading day's close (1-day lag)."
    ),
)
exec_timing = EXEC_TIMING_OPTIONS[exec_timing_label]

vol_months = st.sidebar.slider(
    "Risk parity volatility window (months)",
    min_value=3,
    max_value=24,
    value=12,
    step=1,
)

# ---- DYNAMIC LEVERAGE ----
leverage_enabled = False
leverage_params_ui = {"enabled": False}
with st.sidebar.expander("Leverage", expanded=False):
    st.caption(
        "Scale portfolio exposure using a leverage multiplier. "
        "Choose a method below or leave off. When enabled, benchmarks "
        "receive the same leverage for fair comparison."
    )

    leverage_method = st.selectbox(
        "Leverage method",
        options=["Off", "Constant", "Holdings gate", "VT trend"],
        index=0,
        key="leverage_method",
        help=(
            "Constant: fixed multiplier applied every month. "
            "Holdings gate: leverage when no defensive assets (BIL/BTAL) are held. "
            "VT trend: leverage based on global equity moving averages."
        ),
    )

    leverage_enabled = leverage_method != "Off"

    if leverage_method == "Constant":
        st.caption(
            "A fixed leverage multiplier applied to every month of the backtest. "
            "The same multiplier is also applied to SPY and the balanced benchmarks for fair comparison."
        )
        mult_constant = st.number_input(
            "Leverage multiplier",
            min_value=0.5,
            max_value=3.0,
            value=1.25,
            step=0.05,
            key="mult_constant",
        )

        leverage_cost_annual_const = st.number_input(
            "Annual borrowing cost (%)",
            min_value=0.0,
            max_value=15.0,
            value=3.0,
            step=0.25,
            key="leverage_cost_const",
            help=(
                "Annual interest rate charged on the borrowed portion. "
                "For example, at 1.25x leverage with a 3% cost, you pay "
                "3% per year on the 0.25x borrowed amount (0.75% annually). "
                "Set to 0 to ignore borrowing costs."
            ),
        )

        leverage_params_ui = {
            "enabled": True,
            "method": "constant",
            "mult_constant": mult_constant,
            "leverage_cost_annual": leverage_cost_annual_const / 100.0,
        }

    elif leverage_method == "Holdings gate":
        st.caption(
            "When the strategy selects defensive assets into the final portfolio, "
            "it's already signaling caution. This method applies leverage only when "
            "no defensive assets are held."
        )
        defensive_tickers_input = st.text_input(
            "Defensive tickers (comma-separated)",
            value="BIL, BTAL",
            key="defensive_tickers",
            help="If ANY of these appear in the final holdings, leverage = 1.0x.",
        )
        defensive_tickers_list = [
            t.strip().upper() for t in defensive_tickers_input.split(",") if t.strip()
        ]

        mult_leveraged = st.number_input(
            "Leverage multiplier (when no defensives held)",
            min_value=1.0,
            max_value=4.0,
            value=1.25,
            step=0.05,
            key="mult_leveraged",
        )

        leverage_cost_annual = st.number_input(
            "Annual borrowing cost (%)",
            min_value=0.0,
            max_value=15.0,
            value=3.0,
            step=0.25,
            key="leverage_cost",
            help=(
                "Annual interest rate charged on the borrowed portion. "
                "For example, at 1.25x leverage with a 3% cost, you pay "
                "3% per year on the 0.25x borrowed amount (0.75% annually). "
                "Set to 0 to ignore borrowing costs."
            ),
        )

        leverage_params_ui = {
            "enabled": True,
            "method": "holdings_gate",
            "defensive_tickers": defensive_tickers_list,
            "mult_leveraged": mult_leveraged,
            "leverage_cost_annual": leverage_cost_annual / 100.0,
        }

    elif leverage_method == "VT trend":
        st.caption(
            "Uses a global equity ETF's moving averages to gauge trend. "
            "Risk-on when price is above the long SMA and the short SMA is rising."
        )
        leverage_ticker = st.text_input(
            "Trend ticker",
            value="VT",
            key="leverage_ticker",
            help="Global equity ETF to measure trend. VT (Vanguard Total World Stock) is default.",
        ).strip().upper()

        lev_col1, lev_col2 = st.columns(2)
        with lev_col1:
            leverage_long_sma = st.number_input(
                "Long SMA (days)",
                min_value=50,
                max_value=400,
                value=200,
                step=10,
                key="leverage_long_sma",
                help="Price must be above this SMA for a bullish trend reading."
            )
        with lev_col2:
            leverage_short_sma = st.number_input(
                "Short SMA (days)",
                min_value=10,
                max_value=200,
                value=50,
                step=5,
                key="leverage_short_sma",
                help="This SMA must be rising for a confirmed uptrend."
            )

        leverage_slope_lookback = st.number_input(
            "Slope lookback (days)",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            key="leverage_slope_lookback",
            help="Compare the short SMA now vs. this many trading days ago to determine if it's rising."
        )

        st.markdown("**Multipliers by regime:**")
        lev_m1, lev_m2 = st.columns(2)
        with lev_m1:
            mult_risk_on = st.number_input(
                "Risk-on",
                min_value=1.0,
                max_value=2.0,
                value=1.25,
                step=0.05,
                key="mult_risk_on",
                help="Price > long SMA AND short SMA rising."
            )
        with lev_m2:
            mult_risk_off = st.number_input(
                "Risk-off",
                min_value=0.1,
                max_value=1.0,
                value=0.75,
                step=0.05,
                key="mult_risk_off",
                help="Price < long SMA."
            )

        leverage_params_ui = {
            "enabled": True,
            "method": "vt_trend",
            "ticker": leverage_ticker,
            "long_sma": int(leverage_long_sma),
            "short_sma": int(leverage_short_sma),
            "slope_lookback": int(leverage_slope_lookback),
            "mult_risk_on": mult_risk_on,
            "mult_neutral": 1.0,
            "mult_risk_off": mult_risk_off,
        }

# ---- BITCOIN TREND GATE ----
btc_gate_ui = {"enabled": False}
with st.sidebar.expander("Bitcoin Trend Gate", expanded=False):
    st.caption(
        "Gate Bitcoin into the basket using the Radius Trend filter ported "
        "from the TradingView strategy. When on, BTC-USD is only eligible for "
        "the monthly basket while its trend signal is in the market, and it "
        "still competes for a slot on momentum like every other asset. When "
        "off, BTC behaves like any other fixed-basket asset."
    )

    btc_gate_on = st.checkbox(
        "Enable Bitcoin trend gate",
        value=False,
        key="btc_gate_on",
    )

    if btc_gate_on:
        btc_gate_ticker_ui = st.text_input(
            "Gated asset (held in the basket)",
            value="IBIT",
            key="btc_gate_ticker",
            help="The fixed-basket asset this gate admits or excludes. This "
                 "must match a ticker you actually hold, e.g. IBIT.",
        ).strip().upper()

        btc_gate_signal_ui = st.text_input(
            "Signal source",
            value="BTC-USD",
            key="btc_gate_signal_ticker",
            help="The ticker the trend is computed FROM. Leave as BTC-USD "
                 "when holding a Bitcoin ETF: spot BTC has a decade more "
                 "history and trades 7 days a week, so the trend signal is "
                 "far better conditioned than the ETF's short weekday-only "
                 "series. Set it equal to the gated asset to read the trend "
                 "off the ETF instead.",
        ).strip().upper() or btc_gate_ticker_ui

        btc_gate_mode_label = st.selectbox(
            "Gate rule",
            options=[
                "Long position (matches the strategy)",
                "Trend up only (looser)",
            ],
            index=0,
            key="btc_gate_mode",
            help=(
                "Long position mirrors the script's actual entries and exits: "
                "enter on trend up with price above the upper band, hold until "
                "trend down with price below the lower band. Trend up only "
                "admits BTC whenever the band reads as an uptrend."
            ),
        )
        btc_gate_mode = "position" if btc_gate_mode_label.startswith("Long") else "trend"

        bg_c1, bg_c2 = st.columns(2)
        with bg_c1:
            btc_gate_step = st.number_input(
                "Radius Step",
                min_value=0.001,
                max_value=2.0,
                value=0.15,
                step=0.001,
                format="%.3f",
                key="btc_gate_step",
                help="Matches the 'Radius Step' input in the TradingView strategy.",
            )
        with bg_c2:
            btc_gate_multi = st.number_input(
                "Start Points Distance",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                key="btc_gate_multi",
                help="Matches the 'Start Points Distance' input in the TradingView strategy.",
            )

        st.caption(
            "Computed on daily BTC-USD bars. This is a close approximation of "
            "the TradingView indicator; small differences can arise from data "
            "feed and bar-count alignment."
        )

        btc_gate_ui = {
            "enabled": True,
            "ticker": btc_gate_ticker_ui,
            "signal_ticker": btc_gate_signal_ui,
            "mode": btc_gate_mode,
            "step": float(btc_gate_step),
            "multi": float(btc_gate_multi),
            "n": 3,
            "timeframe": "daily",
        }

# ---- BITCOIN SMA CROSS GATE ----
btc_sma_gate_ui = {"enabled": False}
with st.sidebar.expander("Bitcoin SMA Cross Gate", expanded=False):
    st.caption(
        "A second, independent gate ported from the 8-hour 'Simple SMA Cross' "
        "TradingView strategy, converted to daily bars. When on, BTC-USD is "
        "only eligible for the monthly basket while the SMA cross signal is in "
        "the market, and it still competes for a slot on momentum like every "
        "other asset. This is separate from the Radius Trend gate above. If "
        "both gates are enabled on the same ticker, BTC must pass both."
    )

    btc_sma_gate_on = st.checkbox(
        "Enable Bitcoin SMA cross gate",
        value=False,
        key="btc_sma_gate_on",
    )

    if btc_sma_gate_on:
        btc_sma_gate_ticker_ui = st.text_input(
            "Gated asset (held in the basket)",
            value="IBIT",
            key="btc_sma_gate_ticker",
            help="The fixed-basket asset this gate admits or excludes. This "
                 "must match a ticker you actually hold, e.g. IBIT.",
        ).strip().upper()

        btc_sma_gate_signal_ui = st.text_input(
            "Signal source",
            value="BTC-USD",
            key="btc_sma_gate_signal_ticker",
            help="The ticker the SMA cross is computed FROM. Leave as BTC-USD "
                 "when holding a Bitcoin ETF: the 150-day SMA needs long "
                 "history, and IBIT only starts in January 2024.",
        ).strip().upper() or btc_sma_gate_ticker_ui

        btc_sma_gate_mode_label = st.selectbox(
            "Gate rule",
            options=[
                "Long position (matches the strategy)",
                "Trend up only (looser)",
            ],
            index=0,
            key="btc_sma_gate_mode",
            help=(
                "Long position mirrors the script's entries and exits: enter "
                "when price is above the long SMA, the prior bar was above it, "
                "and the fast SMA has been rising; exit when price closes below "
                "the long SMA for three straight bars or the fast SMA turns "
                "down. Trend up only admits BTC whenever price is above the "
                "long SMA and the fast SMA is rising, ignoring the exit logic."
            ),
        )
        btc_sma_gate_mode = (
            "position" if btc_sma_gate_mode_label.startswith("Long") else "trend"
        )

        st.caption(
            "Defaults convert the 8-hour script to daily bars (three 8-hour "
            "bars per day): SMA 450 -> 150, rising/falling SMA 60 -> 20, "
            "rising/falling period 60 -> 20."
        )

        sg_c1, sg_c2 = st.columns(2)
        with sg_c1:
            btc_sma_long_len = st.number_input(
                "Long SMA length (days)",
                min_value=2,
                max_value=600,
                value=150,
                step=1,
                key="btc_sma_long_len",
                help="Daily equivalent of the script's 450-bar 8-hour SMA.",
            )
        with sg_c2:
            btc_sma_fast_len = st.number_input(
                "Rising/falling SMA length (days)",
                min_value=2,
                max_value=200,
                value=20,
                step=1,
                key="btc_sma_fast_len",
                help="Daily equivalent of the script's 60-bar 8-hour fast SMA.",
            )

        btc_sma_rf_period = st.number_input(
            "Rising/falling period (days)",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
            key="btc_sma_rf_period",
            help=(
                "How many consecutive daily bars the fast SMA must rise (to "
                "enter) or fall (to exit), matching Pine's ta.rising/ta.falling. "
                "Daily equivalent of the script's 60-bar period."
            ),
        )

        st.caption(
            "Computed on daily BTC-USD bars. This is a close approximation of "
            "the 8-hour TradingView strategy; signals differ somewhat because "
            "the bar count and bar timing are not the same on daily data."
        )

        btc_sma_gate_ui = {
            "enabled": True,
            "ticker": btc_sma_gate_ticker_ui,
            "signal_ticker": btc_sma_gate_signal_ui,
            "mode": btc_sma_gate_mode,
            "sma_len": int(btc_sma_long_len),
            "rf_len": int(btc_sma_fast_len),
            "rf_period": int(btc_sma_rf_period),
            "timeframe": "daily",
        }

# ---- RUN BACKTEST (second position) ----
st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# ---- PERIOD (third position) ----
st.sidebar.markdown("---")
st.sidebar.header("Backtest Period")

backtest_start = st.sidebar.text_input(
    "Backtest start (YYYY-MM)",
    value="1995-06",
    help="Results will be filtered to start from this month."
)

backtest_end = st.sidebar.text_input(
    "Backtest end (YYYY-MM)",
    value="",
    help="Leave blank to test through the most recent data. Enter YYYY-MM to stop at a specific month.",
    placeholder="e.g. 2024-12"
)

try:
    start_year = int(backtest_start.split("-")[0])
    download_year = start_year - 2
    download_start = f"{download_year}-01-01"
except (ValueError, IndexError):
    download_start = "2011-01-01"

# ---- FIXED BASKET (fourth, collapsible) ----
st.sidebar.markdown("---")

default_fixed = ["TLT", "GLD", "BIL", "BTAL", "HGER"]
selected_fixed = []
with st.sidebar.expander("Fixed Basket Assets", expanded=False):
    st.caption("Toggle built-in assets and/or add custom tickers below.")
    for ticker, label in DEFAULT_FIXED_ASSETS.items():
        checked = st.checkbox(
            f"{ticker} - {label}",
            value=(ticker in default_fixed),
            key=f"fixed_{ticker}"
        )
        if checked:
            selected_fixed.append(ticker)

    custom_fixed_input = st.text_input(
        "Add custom fixed tickers (comma-separated)",
        value="",
        placeholder="e.g. SHY, UUP, SGOL",
        key="custom_fixed",
    )
    custom_fixed_tickers = [t.strip().upper() for t in custom_fixed_input.split(",") if t.strip()]
    selected_fixed.extend(custom_fixed_tickers)

# ---- EQUITY (fifth, collapsible) ----
default_equity = ["QQQ", "IWM", "DXJ", "EMXC"]
selected_equity = []
with st.sidebar.expander("Equity Sub-Selection", expanded=False):
    st.caption("Choose which equity ETFs are eligible for monthly ranking.")
    for ticker, label in DEFAULT_EQUITY_ETFS.items():
        checked = st.checkbox(
            f"{ticker} - {label}",
            value=(ticker in default_equity),
            key=f"eq_{ticker}"
        )
        if checked:
            selected_equity.append(ticker)

    custom_equity_input = st.text_input(
        "Add custom equity tickers (comma-separated)",
        value="",
        placeholder="e.g. VGK, EWJ, XBI",
        key="custom_equity",
    )
    custom_equity_tickers = [t.strip().upper() for t in custom_equity_input.split(",") if t.strip()]
    selected_equity.extend(custom_equity_tickers)

    equity_top_n = st.radio(
        "How many equity ETFs to select each month?",
        options=[0, 1, 2, 3, 4],
        index=3,
        horizontal=True,
        help="0 = no equity sub-selection (only fixed assets in basket)"
    )

    equity_lookback_label = st.selectbox(
        "Equity momentum lookback",
        options=list(EQUITY_LOOKBACK_OPTIONS.keys()),
        index=4,
    )
    equity_lookbacks, equity_lookback_weights = EQUITY_LOOKBACK_OPTIONS[equity_lookback_label]

# ---- BACKFILL ----
st.sidebar.markdown("---")

# Pre-filled defaults based on Portfolio Visualizer backfill mappings
BACKFILL_DEFAULTS = [
    ("HGER",    "QCI, QCI_SYNTH"),
    ("BTAL",    "BAB"),
    ("BIL",     "VFISX"),
    ("TLT",     "VUSTX"),
    ("GLD",     "GOLD"),
    ("QQQ",     "FSPTX"),
    ("IWM",     "OTCFX"),
    ("EMXC",    "EEM, VEIEX"),
    ("DXJ",     "FJPNX"),
    ("IBIT",    "BTC-USD"),
    ("",        ""),
    ("",        ""),
]

backfill_rules = {}
with st.sidebar.expander("Backfill Rules", expanded=False):
    st.caption(
        "Extend an asset's history using older ETFs or mutual funds. "
        "Enter the primary ticker on the left and its backfill chain on the right "
        "(tried in order, oldest last). Leave blank rows if not needed."
    )
    for slot in range(1, len(BACKFILL_DEFAULTS) + 1):
        default_primary, default_chain = BACKFILL_DEFAULTS[slot - 1]
        cols = st.columns([1, 2])
        with cols[0]:
            bf_primary = st.text_input(
                f"Asset {slot}",
                value=default_primary,
                key=f"bf_primary_{slot}",
                placeholder="Ticker",
            ).strip().upper()
        with cols[1]:
            bf_chain = st.text_input(
                f"Backfill chain {slot}",
                value=default_chain,
                key=f"bf_chain_{slot}",
                placeholder="e.g. USCI, DBC",
            )

        if bf_primary:
            chain_list = [t.strip().upper() for t in bf_chain.split(",") if t.strip()]
            if chain_list:
                backfill_rules[bf_primary] = chain_list

# Always back-fill the balanced benchmarks from the proxies declared in
# BAL_BENCHMARKS, regardless of the sidebar backfill rows above. This keeps
# AOR/AOA history extended back to the mid-1990s (via VSMGX/VASGX) on every
# run so the benchmark lines match the strategy and SPY across the full window.
for _bcfg in BAL_BENCHMARKS:
    _bf_chain = _bcfg.get("backfill")
    if _bf_chain:
        backfill_rules[_bcfg["ticker"]] = [t.strip().upper() for t in _bf_chain]

# ---- DATA CACHE ----
st.sidebar.markdown("---")
st.sidebar.header("Data Cache")
st.sidebar.caption(
    "Price data is cached locally so the dashboard only downloads "
    "new dates and missing tickers on subsequent runs."
)

force_refresh = st.sidebar.checkbox(
    "Force full refresh (ignore cache)",
    value=False,
    help="Re-download all price data from Yahoo Finance, replacing the local cache."
)

cache_info = get_cache_info()
if cache_info:
    st.sidebar.caption(
        f"Last updated: {cache_info['last_updated'].strftime('%Y-%m-%d %H:%M')} · "
        f"{cache_info['n_tickers']} tickers · "
        f"{cache_info['size_mb']:.1f} MB"
    )
    if st.sidebar.button("Clear Cache", type="secondary"):
        try:
            os.remove(CACHE_FILE)
            st.sidebar.success("Cache cleared.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Could not clear cache: {e}")
else:
    st.sidebar.caption("No cached data yet. Data will be cached on the first run.")


# =============================================================================
# MAIN AREA
# =============================================================================

st.markdown(
    """
    <div class="ae-brand-header">
        <span class="ae-brand-name">Adaptive Portfolio Strategy</span>
        <span class="ae-brand-tag">Backtest Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="ae-brand-sub">Momentum + risk-parity allocation across an '
    f'uncorrelated asset basket &nbsp;&middot;&nbsp; '
    f'Loaded {datetime.now().strftime("%b %d, %Y, %I:%M %p")}</p>',
    unsafe_allow_html=True,
)

st.markdown("---")


if run_button:
    if len(selected_fixed) == 0 and equity_top_n == 0:
        st.error("You need at least some assets in the basket. Select fixed assets or enable equity sub-selection.")
    else:
        # Gather ALL tickers needed (including backfills)
        all_tickers_needed = (
            list(selected_fixed) + list(selected_equity)
            + [BENCHMARK_TICKER, "BIL"]
            + [c["ticker"] for c in BAL_BENCHMARKS]
        )
        if leverage_enabled and leverage_params_ui.get("method") == "vt_trend" and leverage_params_ui.get("ticker"):
            all_tickers_needed.append(leverage_params_ui["ticker"])
        for bf_chain in backfill_rules.values():
            all_tickers_needed.extend(bf_chain)
        # Pass as tuple for cache hashability
        tickers_tuple = tuple(sorted(set(all_tickers_needed)))

        daily_prices = download_data(tickers_tuple, download_start, force_refresh=force_refresh)

        if daily_prices is None or daily_prices.empty:
            st.error("Failed to download price data. Check your internet connection.")
        else:
            # Make a mutable copy (cached data is read-only)
            daily_prices = daily_prices.copy()

            # Apply backfill rules
            backfill_info = []
            for primary, chain in backfill_rules.items():
                if primary in daily_prices.columns or any(bf in daily_prices.columns for bf in chain):
                    spliced, segments = build_backfilled_series(
                        daily_prices, primary, chain,
                        fee_annual=BACKFILL_FEE_ANNUAL.get(primary, 0.0),
                    )
                    daily_prices[primary] = spliced
                    for seg_ticker, seg_start, seg_end in segments:
                        backfill_info.append({
                            "Asset": primary,
                            "Source": seg_ticker,
                            "From": seg_start.strftime("%Y-%m-%d"),
                            "To": seg_end.strftime("%Y-%m-%d"),
                        })

            monthly_prices = get_month_end_prices(daily_prices)

            # ---- DATA VALIDATION ----
            all_selected = list(selected_fixed)
            if equity_top_n > 0:
                all_selected += list(selected_equity)

            max_lb = max(main_lookbacks)
            try:
                bs_year = int(backtest_start.split("-")[0])
                bs_month = int(backtest_start.split("-")[1])
                required_by = pd.Timestamp(year=bs_year, month=bs_month, day=1) - pd.DateOffset(months=max_lb + 1)
            except (ValueError, IndexError):
                required_by = pd.Timestamp("2012-01-01")

            data_issues = []
            for ticker in all_selected:
                if ticker not in daily_prices.columns:
                    data_issues.append({
                        "Ticker": ticker,
                        "Issue": "Not found in downloaded data (check spelling)",
                        "Earliest Data": "N/A",
                        "Data Required By": required_by.strftime("%Y-%m"),
                    })
                else:
                    first_valid = daily_prices[ticker].dropna().index.min()
                    if first_valid is None:
                        data_issues.append({
                            "Ticker": ticker,
                            "Issue": "No price data available",
                            "Earliest Data": "N/A",
                            "Data Required By": required_by.strftime("%Y-%m"),
                        })
                    elif first_valid > required_by:
                        data_issues.append({
                            "Ticker": ticker,
                            "Issue": f"Data starts too late (need {max_lb}mo lookback before {backtest_start})",
                            "Earliest Data": first_valid.strftime("%Y-%m-%d"),
                            "Data Required By": required_by.strftime("%Y-%m"),
                        })

            if data_issues:
                st.error("**Backtest blocked.** The following assets do not have enough data for your selected start date:")
                st.dataframe(pd.DataFrame(data_issues), use_container_width=True, hide_index=True)

                latest_inception = pd.Timestamp("1900-01-01")
                for ticker in all_selected:
                    if ticker in daily_prices.columns:
                        fv = daily_prices[ticker].dropna().index.min()
                        if fv is not None and fv > latest_inception:
                            latest_inception = fv

                if latest_inception > pd.Timestamp("1900-01-01"):
                    earliest_viable = latest_inception + pd.DateOffset(months=max_lb + 2)
                    st.info(
                        f"**Suggestion:** The earliest backtest start date that would include "
                        f"all selected assets is approximately **{earliest_viable.strftime('%Y-%m')}**."
                    )
            else:
                # ---- RUN BACKTEST ----
                backfill_summary = "; ".join(
                    [f"{k} <- {', '.join(v)}" for k, v in backfill_rules.items()]
                ) if backfill_rules else "None"

                # ---- BITCOIN TREND GATE (compute the daily signal once) ----
                btc_gate_config = {"enabled": False}
                btc_gate_state_for_display = None
                if btc_gate_ui.get("enabled"):
                    g_ticker = btc_gate_ui["ticker"]
                    g_signal = gate_signal_ticker(btc_gate_ui)
                    if g_ticker not in selected_fixed:
                        st.warning(
                            f"Bitcoin trend gate is on, but {g_ticker} is not in the "
                            f"fixed basket, so the gate has no effect. Set the gated "
                            f"asset to a ticker you hold (e.g. IBIT), or add "
                            f"{g_ticker} under Fixed Basket Assets."
                        )
                    with st.spinner(f"Computing {g_signal} trend gate..."):
                        gate_ohlc = download_gate_ohlc(
                            g_signal, download_start, force_refresh=force_refresh
                        )
                    if gate_ohlc is None or gate_ohlc.empty:
                        st.warning(
                            f"Could not load OHLC data for {g_signal}; the Bitcoin "
                            f"trend gate is disabled for this run."
                        )
                    else:
                        gate_state = compute_radius_trend_state(
                            gate_ohlc,
                            step=btc_gate_ui["step"],
                            multi=btc_gate_ui["multi"],
                            n=btc_gate_ui.get("n", 3),
                        )
                        open_col = "position" if btc_gate_ui["mode"] == "position" else "trend"
                        if gate_state is None or gate_state.empty or open_col not in gate_state.columns:
                            st.warning(
                                f"Trend gate signal could not be computed for "
                                f"{g_signal}; gate disabled for this run."
                            )
                        else:
                            btc_gate_config = {
                                "enabled": True,
                                "ticker": g_ticker,
                                "signal_ticker": g_signal,
                                "mode": btc_gate_ui["mode"],
                                "step": btc_gate_ui["step"],
                                "multi": btc_gate_ui["multi"],
                                "n": btc_gate_ui.get("n", 3),
                                "gate_series": (gate_state[open_col] == 1.0),
                            }
                            btc_gate_state_for_display = gate_state

                # ---- BITCOIN SMA CROSS GATE (compute the daily signal once) ----
                btc_sma_gate_config = {"enabled": False}
                btc_sma_gate_state_for_display = None
                if btc_sma_gate_ui.get("enabled"):
                    sg_ticker = btc_sma_gate_ui["ticker"]
                    sg_signal = gate_signal_ticker(btc_sma_gate_ui)
                    if sg_ticker not in selected_fixed:
                        st.warning(
                            f"Bitcoin SMA cross gate is on, but {sg_ticker} is not in "
                            f"the fixed basket, so the gate has no effect. Set the "
                            f"gated asset to a ticker you hold (e.g. IBIT), or add "
                            f"{sg_ticker} under Fixed Basket Assets."
                        )
                    with st.spinner(f"Computing {sg_signal} SMA cross gate..."):
                        sma_gate_ohlc = download_gate_ohlc(
                            sg_signal, download_start, force_refresh=force_refresh
                        )
                    if sma_gate_ohlc is None or sma_gate_ohlc.empty:
                        st.warning(
                            f"Could not load price data for {sg_signal}; the Bitcoin "
                            f"SMA cross gate is disabled for this run."
                        )
                    else:
                        sma_gate_state = compute_sma_cross_state(
                            sma_gate_ohlc,
                            sma_len=btc_sma_gate_ui["sma_len"],
                            rf_len=btc_sma_gate_ui["rf_len"],
                            rf_period=btc_sma_gate_ui["rf_period"],
                        )
                        sma_open_col = (
                            "position" if btc_sma_gate_ui["mode"] == "position" else "trend"
                        )
                        if (
                            sma_gate_state is None
                            or sma_gate_state.empty
                            or sma_open_col not in sma_gate_state.columns
                            or sma_gate_state[sma_open_col].notna().sum() == 0
                        ):
                            st.warning(
                                f"SMA cross gate signal could not be computed for "
                                f"{sg_signal} (not enough daily history for the "
                                f"{btc_sma_gate_ui['sma_len']}-day SMA); gate disabled "
                                f"for this run."
                            )
                        else:
                            btc_sma_gate_config = {
                                "enabled": True,
                                "ticker": sg_ticker,
                                "signal_ticker": sg_signal,
                                "mode": btc_sma_gate_ui["mode"],
                                "sma_len": btc_sma_gate_ui["sma_len"],
                                "rf_len": btc_sma_gate_ui["rf_len"],
                                "rf_period": btc_sma_gate_ui["rf_period"],
                                "gate_series": (sma_gate_state[sma_open_col] == 1.0),
                            }
                            btc_sma_gate_state_for_display = sma_gate_state

                params = {
                    "fixed_tickers": selected_fixed,
                    "equity_tickers": selected_equity,
                    "equity_top_n": equity_top_n,
                    "equity_lookbacks": equity_lookbacks,
                    "equity_lookback_weights": equity_lookback_weights,
                    "equity_lookback_label": equity_lookback_label,
                    "main_lookbacks": main_lookbacks,
                    "main_lookback_weights": main_lookback_weights,
                    "main_lookback_label": main_lookback_label,
                    "final_top_n": final_top_n,
                    "vol_months": vol_months,
                    "exec_timing": exec_timing,
                    "exec_timing_label": exec_timing_label,
                    "backtest_start": backtest_start,
                    "backtest_end": backtest_end,
                    "backfill_summary": backfill_summary,
                    "leverage_params": leverage_params_ui,
                    "btc_gate": btc_gate_config,
                    "btc_sma_gate": btc_sma_gate_config,
                }

                with st.spinner("Running backtest..."):
                    results_df, trade_df, daily_equity = run_backtest(daily_prices, monthly_prices, params)

                if results_df.empty:
                    st.error("No results produced. The backtest start date may be too early for the selected assets.")
                else:
                    bench_monthly = monthly_prices[BENCHMARK_TICKER].pct_change()
                    common_dates = results_df.index.intersection(bench_monthly.index)
                    bench_returns = bench_monthly.loc[common_dates]
                    strat_returns = results_df["portfolio_return"].loc[common_dates]

                    # Extract leverage multipliers and borrowing costs
                    leverage_mults = results_df["leverage_mult"].loc[common_dates]
                    borrow_costs = results_df["borrow_cost"].loc[common_dates]

                    # Apply dynamic leverage + borrowing costs to benchmarks
                    if leverage_enabled:
                        bench_returns = bench_returns * leverage_mults - borrow_costs

                    # Balanced benchmark returns — one entry per BAL_BENCHMARKS config.
                    # Each entry carries label, ticker, color, linestyle, and the
                    # leverage-adjusted monthly return series. Metrics and daily
                    # equity get added below once those are computed.
                    balanced_data = []
                    for cfg in BAL_BENCHMARKS:
                        if cfg["ticker"] not in monthly_prices.columns:
                            continue
                        bal_monthly = monthly_prices[cfg["ticker"]].pct_change()
                        bal_returns = bal_monthly.reindex(common_dates)
                        if leverage_enabled and bal_returns is not None:
                            bal_returns = bal_returns * leverage_mults - borrow_costs
                        balanced_data.append({
                            **cfg,
                            "returns": bal_returns,
                            "daily_equity": pd.Series(dtype=float),
                            "metrics": {},
                        })

                    # Risk-free rate from BIL for Sharpe/Sortino calculation
                    rf_monthly = None
                    if "BIL" in monthly_prices.columns:
                        rf_monthly = monthly_prices["BIL"].pct_change().loc[common_dates]

                    # Build daily equity curves for benchmarks (for daily drawdown)
                    # When leverage is enabled, apply the same monthly multipliers
                    # and borrowing costs to daily benchmark returns.
                    spy_daily_equity = pd.Series(dtype=float)
                    if len(daily_equity) > 0:
                        de_start, de_end = daily_equity.index[0], daily_equity.index[-1]

                        # Build daily leverage and cost maps from the trade log
                        daily_lev_map = None
                        daily_cost_map = None
                        if leverage_enabled and not trade_df.empty:
                            daily_idx = daily_prices.loc[de_start:de_end].index
                            daily_lev_map = pd.Series(1.0, index=daily_idx)
                            daily_cost_map = pd.Series(0.0, index=daily_idx)
                            for _, trow in trade_df.iterrows():
                                entry_str = trow.get("exec_entry", "")
                                exit_str = trow.get("exec_exit", "")
                                if not entry_str or not exit_str:
                                    continue
                                t_entry = pd.Timestamp(entry_str)
                                t_exit = pd.Timestamp(exit_str)
                                t_lev = float(trow["leverage"].replace("x", ""))
                                mask = (
                                    (daily_lev_map.index >= t_entry)
                                    & (daily_lev_map.index <= t_exit)
                                )
                                daily_lev_map[mask] = t_lev
                                # Spread monthly borrowing cost across trading days
                                n_days = mask.sum()
                                if t_lev > 1.0 and n_days > 0:
                                    lev_cost_ann = leverage_params_ui.get("leverage_cost_annual", 0.0)
                                    monthly_cost = (t_lev - 1.0) * (lev_cost_ann / 12.0)
                                    daily_cost_map[mask] = monthly_cost / n_days

                        if BENCHMARK_TICKER in daily_prices.columns:
                            spy_daily = daily_prices[BENCHMARK_TICKER].loc[de_start:de_end].dropna()
                            if len(spy_daily) > 0:
                                if daily_lev_map is not None:
                                    spy_ret = spy_daily.pct_change().fillna(0)
                                    spy_lev = daily_lev_map.reindex(spy_daily.index).fillna(1.0)
                                    spy_cost = daily_cost_map.reindex(spy_daily.index).fillna(0.0)
                                    spy_ret_lev = spy_ret * spy_lev - spy_cost
                                    spy_daily_equity = INITIAL_CAPITAL * (1 + spy_ret_lev).cumprod()
                                else:
                                    spy_daily_equity = INITIAL_CAPITAL * (spy_daily / spy_daily.iloc[0])

                        for entry in balanced_data:
                            t = entry["ticker"]
                            if t not in daily_prices.columns:
                                continue
                            bal_daily = daily_prices[t].loc[de_start:de_end].dropna()
                            if len(bal_daily) == 0:
                                continue
                            if daily_lev_map is not None:
                                bal_ret = bal_daily.pct_change().fillna(0)
                                bal_lev = daily_lev_map.reindex(bal_daily.index).fillna(1.0)
                                bal_cost = daily_cost_map.reindex(bal_daily.index).fillna(0.0)
                                bal_ret_lev = bal_ret * bal_lev - bal_cost
                                entry["daily_equity"] = INITIAL_CAPITAL * (1 + bal_ret_lev).cumprod()
                            else:
                                entry["daily_equity"] = INITIAL_CAPITAL * (bal_daily / bal_daily.iloc[0])

                    strat_metrics = compute_metrics(strat_returns, rf_monthly=rf_monthly, daily_equity=daily_equity)
                    bench_metrics = compute_metrics(bench_returns, rf_monthly=rf_monthly, daily_equity=spy_daily_equity)
                    for entry in balanced_data:
                        entry["metrics"] = compute_metrics(
                            entry["returns"].dropna(),
                            rf_monthly=rf_monthly,
                            daily_equity=entry["daily_equity"],
                        )

                    try:
                        append_to_excel(params, strat_metrics, bench_metrics, RESULTS_FILE)
                        st.success(f"Results appended to **{os.path.basename(RESULTS_FILE)}**")
                    except Exception as e:
                        st.warning(f"Could not write to Excel (file may be open): {e}")

                    # ---- METRICS TABLE (moved to top of results) ----
                    st.subheader("Performance Summary")

                    def fmt_pct(v):
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            return f"{v:.2%}"
                        elif isinstance(v, float) and np.isnan(v):
                            return "N/A"
                        return v

                    def fmt_ratio(v):
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            return f"{v:.2f}"
                        elif isinstance(v, float) and np.isnan(v):
                            return "N/A"
                        return v

                    def fmt_dollar(v):
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            return f"${v:,.0f}"
                        elif isinstance(v, float) and np.isnan(v):
                            return "N/A"
                        return v

                    # Build columns: Strategy, SPY, then one per balanced benchmark
                    metric_cols = [
                        ("Strategy", strat_metrics),
                        ("SPY", bench_metrics),
                    ]
                    for entry in balanced_data:
                        metric_cols.append((entry["label"], entry["metrics"]))

                    def metric_row(name, key, fmt, bold):
                        vals = [fmt(m.get(key, 0)) for _, m in metric_cols]
                        return (name, vals, bold)

                    # Ordered with key metrics first; Sharpe and Total Months removed
                    growth_key = f"Growth of ${INITIAL_CAPITAL:,}"
                    metrics_rows = [
                        metric_row("Sortino Ratio",     "Sortino Ratio",       fmt_ratio,  True),
                        metric_row("CAGR",              "CAGR",                fmt_pct,    True),
                        metric_row("Max Drawdown",      "Max Drawdown",        fmt_pct,    True),
                        metric_row("Max DD (Daily)",    "Max Drawdown (Daily)",fmt_pct,    True),
                        metric_row("Worst Single Day",  "Worst Day",           fmt_pct,    False),
                        metric_row("Calmar Ratio",      "Calmar Ratio",        fmt_ratio,  False),
                        metric_row("Annualized Vol",    "Annualized Vol",      fmt_pct,    False),
                        metric_row("Best Year",         "Best Year",           fmt_pct,    False),
                        metric_row("Worst Year",        "Worst Year",          fmt_pct,    False),
                        metric_row("Win Rate",          "Win Rate (monthly)",  fmt_pct,    False),
                        metric_row("Total Return",      "Total Return",        fmt_pct,    False),
                        metric_row(growth_key,          growth_key,            fmt_dollar, False),
                        ("Period", [m.get("Period", "") for _, m in metric_cols], False),
                    ]

                    # Build HTML table with bold top 3 rows; columns dynamic
                    lev_tag = " (lev)" if leverage_enabled else ""
                    header_cells = "".join(
                        f"<th>{label}{lev_tag if label != 'Strategy' else ''}</th>"
                        for label, _ in metric_cols
                    )
                    html_parts = [
                        '<style>',
                        ".metrics-table { width: 100%; border-collapse: collapse; font-family: 'DM Sans', sans-serif; font-size: 14px; color: #484848; margin: 0.4rem 0 0.8rem 0; }",
                        '.metrics-table thead th { text-align: right; padding: 11px 14px; border-bottom: 2px solid #3A0CA3; background-color: #fafafa; font-weight: 600; font-size: 13px; color: #484848; letter-spacing: 0.01em; }',
                        '.metrics-table thead th:first-child { text-align: left; }',
                        '.metrics-table tbody td { padding: 9px 14px; border-bottom: 1px solid #eeeeee; text-align: right; font-variant-numeric: tabular-nums; }',
                        '.metrics-table tbody td:first-child { text-align: left; color: #666677; font-weight: 500; }',
                        '.metrics-table tbody tr:hover { background-color: #f7f5fb; }',
                        '.metrics-table .bold-row td { font-weight: 700; color: #484848; }',
                        '</style>',
                        '<table class="metrics-table">',
                        f'<thead><tr><th>Metric</th>{header_cells}</tr></thead>',
                        '<tbody>',
                    ]
                    for metric, vals, is_bold in metrics_rows:
                        row_class = ' class="bold-row"' if is_bold else ""
                        cells = "".join(f"<td>{v}</td>" for v in vals)
                        html_parts.append(f'<tr{row_class}><td>{metric}</td>{cells}</tr>')
                    html_parts.append('</tbody></table>')
                    html = "\n".join(html_parts)

                    st.markdown(html, unsafe_allow_html=True)

                    # Show worst daily drawdown date range
                    peak_dt = strat_metrics.get("DD Peak Date")
                    trough_dt = strat_metrics.get("DD Trough Date")
                    if peak_dt is not None and trough_dt is not None:
                        st.caption(
                            f"Worst daily drawdown: peak on {peak_dt.strftime('%Y-%m-%d')}, "
                            f"trough on {trough_dt.strftime('%Y-%m-%d')}"
                        )
                    worst_day_dt = strat_metrics.get("Worst Day Date")
                    if worst_day_dt is not None:
                        st.caption(
                            f"Worst single-day decline: {strat_metrics.get('Worst Day', 0):.2%} "
                            f"on {worst_day_dt.strftime('%Y-%m-%d')}"
                        )

                    # ---- CURRENT SIGNALS ----
                    signals = compute_current_signals(daily_prices, monthly_prices, params)
                    if signals is not None:
                        st.subheader("Current Model Signals")
                        sig_date = signals["signal_date"]
                        st.caption(
                            f"As of {sig_date.strftime('%m/%d/%Y')} close · "
                            "this is what the model would recommend if you "
                            "rebalanced today."
                        )

                        # Holdings summary (live signal)
                        holdings_parts = []
                        for t in signals["final_holdings"]:
                            w = signals["weights"].get(t, 0)
                            holdings_parts.append(f"{w:.2%} {t}")
                        st.markdown("**Recommended holdings:** " + " | ".join(holdings_parts))

                        if signals["selected_equities"]:
                            st.markdown(
                                "**Equity picks:** "
                                + ", ".join(signals["selected_equities"])
                            )

                        # Previous month-end reference line
                        prev_date = signals.get("prev_month_end_date")
                        prev_holdings = signals.get("prev_holdings", [])
                        prev_weights = signals.get("prev_weights", {})
                        if prev_date is not None and prev_holdings:
                            prev_parts = []
                            for t in prev_holdings:
                                w = prev_weights.get(t, 0)
                                prev_parts.append(f"{w:.2%} {t}")
                            st.caption(
                                f"Previous month-end signal "
                                f"({prev_date.strftime('%m/%d/%Y')}): "
                                + " | ".join(prev_parts)
                            )

                        # Current leverage signal
                        if leverage_params_ui.get("enabled", False):
                            method = leverage_params_ui.get("method", "holdings_gate")
                            if method == "constant":
                                mult_c = leverage_params_ui.get("mult_constant", 1.25)
                                st.markdown(f"**Leverage:** {mult_c:.2f}x (constant)")
                            elif method == "holdings_gate":
                                # For holdings gate, use the current signal's final holdings
                                lev_m, lev_r, lev_d = compute_leverage_signal(
                                    daily_prices, sig_date, leverage_params_ui,
                                    final_holdings=signals["final_holdings"]
                                )
                                regime_colors = {
                                    "leveraged": "🟢", "defensive": "🟡",
                                    "off": "⚪", "no holdings": "⚪",
                                }
                                emoji = regime_colors.get(lev_r, "⚪")
                                lev_line = f"**Leverage:** {emoji} {lev_r} ({lev_m:.2f}x)"
                                if lev_d.get("defensives_held"):
                                    lev_line += f"  (holding {', '.join(lev_d['defensives_held'])})"
                                st.markdown(lev_line)
                            else:
                                lev_m, lev_r, lev_d = compute_leverage_signal(
                                    daily_prices, sig_date, leverage_params_ui,
                                    final_holdings=signals["final_holdings"]
                                )
                                lev_ticker = leverage_params_ui["ticker"]
                                regime_colors = {
                                    "risk-on": "🟢", "neutral": "🟡", "risk-off": "🔴",
                                    "off": "⚪", "no data": "⚪", "insufficient history": "⚪",
                                }
                                emoji = regime_colors.get(lev_r, "⚪")
                                lev_parts = [f"**Leverage:** {emoji} {lev_r} ({lev_m:.2f}x)"]
                                if lev_d:
                                    lev_parts.append(
                                        f"  {lev_ticker}: ${lev_d['price']:.2f} | "
                                        f"{leverage_params_ui['long_sma']}d SMA: ${lev_d['long_sma']:.2f} "
                                        f"({'above' if lev_d['price_above_long'] else 'below'}) | "
                                        f"{leverage_params_ui['short_sma']}d SMA: "
                                        f"{'rising' if lev_d['short_sma_rising'] else 'falling'}"
                                    )
                                st.markdown("  \n".join(lev_parts))

                        # Current Bitcoin gate signal
                        if btc_gate_config.get("enabled"):
                            g_ticker = gate_label(btc_gate_config)
                            open_now = _gate_open_as_of(btc_gate_config, sig_date)
                            emoji = "🟢" if open_now else "🔴"
                            status_word = "in the market" if open_now else "out of the market"
                            mode_word = (
                                "long-position" if btc_gate_config["mode"] == "position"
                                else "trend-up"
                            )
                            gate_line = (
                                f"**Bitcoin gate ({mode_word}):** {emoji} {g_ticker} "
                                f"{status_word}"
                            )
                            if btc_gate_state_for_display is not None:
                                sub = btc_gate_state_for_display.loc[:sig_date]
                                if len(sub) > 0:
                                    last = sub.iloc[-1]
                                    cl = last.get("close", np.nan)
                                    bnd = last.get("band", np.nan)
                                    if not (np.isnan(cl) or np.isnan(bnd)):
                                        gate_line += (
                                            f"  ·  close ${cl:,.0f} vs band ${bnd:,.0f}"
                                        )
                            if not open_now and btc_gate_config["ticker"] in signals.get("basket", []):
                                pass  # gated-out ticker won't be in the basket
                            elif not open_now:
                                gate_line += "  (excluded from the basket)"
                            st.markdown(gate_line)

                        # Current Bitcoin SMA cross gate signal
                        if btc_sma_gate_config.get("enabled"):
                            sg_ticker = gate_label(btc_sma_gate_config)
                            sma_open_now = _gate_open_as_of(btc_sma_gate_config, sig_date)
                            sma_emoji = "🟢" if sma_open_now else "🔴"
                            sma_status_word = (
                                "in the market" if sma_open_now else "out of the market"
                            )
                            sma_mode_word = (
                                "long-position" if btc_sma_gate_config["mode"] == "position"
                                else "trend-up"
                            )
                            sma_gate_line = (
                                f"**Bitcoin SMA cross gate ({sma_mode_word}):** "
                                f"{sma_emoji} {sg_ticker} {sma_status_word}"
                            )
                            if btc_sma_gate_state_for_display is not None:
                                sub = btc_sma_gate_state_for_display.loc[:sig_date]
                                if len(sub) > 0:
                                    last = sub.iloc[-1]
                                    cl = last.get("close", np.nan)
                                    sl = last.get("sma_long", np.nan)
                                    if not (np.isnan(cl) or np.isnan(sl)):
                                        sma_gate_line += (
                                            f"  ·  close ${cl:,.0f} vs "
                                            f"{btc_sma_gate_config['sma_len']}d SMA ${sl:,.0f}"
                                        )
                            if not sma_open_now and sg_ticker not in signals.get("basket", []):
                                sma_gate_line += "  (excluded from the basket)"
                            st.markdown(sma_gate_line)

                        # Details table
                        details = signals["asset_details"]
                        vol_months_label = f"{params['vol_months']}mo Vol (ann)"

                        # Build lookback column headers
                        lb_cols = sorted(set(params["main_lookbacks"]))
                        lb_headers = [f"{lb}mo Return" for lb in lb_cols]

                        sig_html = [
                            '<style>',
                            ".sig-table { width: 100%; border-collapse: collapse; font-family: 'DM Sans', sans-serif; font-size: 13px; color: #484848; }",
                            '.sig-table th { text-align: right; padding: 8px 11px; border-bottom: 2px solid #3A0CA3; background-color: #fafafa; font-weight: 600; color: #484848; }',
                            '.sig-table th:first-child { text-align: left; }',
                            '.sig-table td { padding: 7px 11px; border-bottom: 1px solid #eeeeee; text-align: right; font-variant-numeric: tabular-nums; }',
                            '.sig-table td:first-child { text-align: left; color: #666677; }',
                            '.sig-table tr:hover { background-color: #f7f5fb; }',
                            '.sig-selected { background-color: #F0EDF8; }',
                            '.sig-equity { font-style: italic; }',
                            '</style>',
                            '<table class="sig-table">',
                            '<thead><tr>',
                            '<th>Asset</th>',
                        ]
                        for h in lb_headers:
                            sig_html.append(f'<th>{h}</th>')
                        sig_html.append('<th>Blended</th>')
                        sig_html.append(f'<th>{vol_months_label}</th>')
                        sig_html.append('<th>Weight</th>')
                        sig_html.append('<th>Status</th>')
                        sig_html.append('</tr></thead><tbody>')

                        for row in details:
                            classes = []
                            if row.get("Selected"):
                                classes.append("sig-selected")
                            class_str = f' class="{" ".join(classes)}"' if classes else ""

                            sig_html.append(f'<tr{class_str}>')

                            # Asset name
                            name = row["Asset"]
                            if row.get("Equity Pick"):
                                name = f'<span class="sig-equity">{name} (eq)</span>'
                            sig_html.append(f'<td><strong>{name}</strong></td>')

                            # Lookback returns
                            for h in lb_headers:
                                val = row.get(h, np.nan)
                                if isinstance(val, (int, float)) and not np.isnan(val):
                                    sig_html.append(f'<td>{val:.2%}</td>')
                                else:
                                    sig_html.append('<td>-</td>')

                            # Blended score
                            blended = row.get("Blended Score", np.nan)
                            if isinstance(blended, (int, float)) and not np.isnan(blended):
                                sig_html.append(f'<td><strong>{blended:.2%}</strong></td>')
                            else:
                                sig_html.append('<td>-</td>')

                            # Volatility
                            vol = row.get(vol_months_label, np.nan)
                            if isinstance(vol, (int, float)) and not np.isnan(vol):
                                sig_html.append(f'<td>{vol:.2%}</td>')
                            else:
                                sig_html.append('<td>-</td>')

                            # Weight
                            w = row.get("Weight", 0)
                            if w > 0:
                                sig_html.append(f'<td><strong>{w:.2%}</strong></td>')
                            else:
                                sig_html.append('<td>-</td>')

                            # Status
                            if row.get("Selected"):
                                sig_html.append('<td>HELD</td>')
                            elif row.get("In Basket"):
                                sig_html.append('<td>in basket</td>')
                            else:
                                sig_html.append('<td>-</td>')

                            sig_html.append('</tr>')

                        sig_html.append('</tbody></table>')
                        st.markdown("\n".join(sig_html), unsafe_allow_html=True)
                        st.markdown("")

                        # ---- EQUITY SUB-SELECTION RANKING ----
                        # Equities are pre-filtered by a separate equity
                        # momentum lookback before reaching the main basket.
                        # This table shows that ranking so it is clear why an
                        # equity ETF strong on the main lookback can still miss
                        # the cut (it lost the equity-lookback contest).
                        eq_details = signals.get("equity_details", [])
                        if eq_details:
                            eq_lbs = params["equity_lookbacks"]
                            if len(eq_lbs) == 1:
                                eq_desc = f"{eq_lbs[0]}-month"
                            else:
                                eq_desc = (
                                    "/".join(str(l) for l in eq_lbs) + "-month blend"
                                )
                            eq_top_n = params["equity_top_n"]

                            st.subheader("Equity Sub-Selection Ranking")
                            st.caption(
                                f"Equity ETFs ranked by the equity momentum "
                                f"lookback ({eq_desc}). The top {eq_top_n} "
                                f"advance into the main basket; the rest are "
                                f"dropped before the main-lookback ranking "
                                f"above is applied. This is why an ETF can rank "
                                f"high above yet still not be an equity pick."
                            )

                            eq_lb_cols = sorted(set(eq_lbs))
                            eq_lb_headers = [f"{lb}mo Return" for lb in eq_lb_cols]

                            eq_html = [
                                '<table class="sig-table">',
                                '<thead><tr>',
                                '<th>Asset</th>',
                            ]
                            for h in eq_lb_headers:
                                eq_html.append(f'<th>{h}</th>')
                            eq_html.append('<th>Blended</th>')
                            eq_html.append(f'<th>{vol_months_label}</th>')
                            eq_html.append('<th>Status</th>')
                            eq_html.append('</tr></thead><tbody>')

                            for row in eq_details:
                                is_pick = row.get("Equity Pick")
                                class_str = ' class="sig-selected"' if is_pick else ""
                                eq_html.append(f'<tr{class_str}>')

                                eq_html.append(
                                    f'<td><strong>{row["Asset"]}</strong></td>'
                                )

                                for h in eq_lb_headers:
                                    val = row.get(h, np.nan)
                                    if isinstance(val, (int, float)) and not np.isnan(val):
                                        eq_html.append(f'<td>{val:.2%}</td>')
                                    else:
                                        eq_html.append('<td>-</td>')

                                blended = row.get("Blended Score", np.nan)
                                if isinstance(blended, (int, float)) and not np.isnan(blended):
                                    eq_html.append(f'<td><strong>{blended:.2%}</strong></td>')
                                else:
                                    eq_html.append('<td>-</td>')

                                vol = row.get(vol_months_label, np.nan)
                                if isinstance(vol, (int, float)) and not np.isnan(vol):
                                    eq_html.append(f'<td>{vol:.2%}</td>')
                                else:
                                    eq_html.append('<td>-</td>')

                                if is_pick:
                                    eq_html.append('<td>PICK</td>')
                                else:
                                    eq_html.append('<td>dropped</td>')

                                eq_html.append('</tr>')

                            eq_html.append('</tbody></table>')
                            st.markdown("\n".join(eq_html), unsafe_allow_html=True)
                            st.markdown("")

                    # ---- CHARTS ----
                    st.subheader("Equity Curve and Drawdown")
                    fig = plot_equity_and_drawdown(strat_returns, bench_returns, balanced_data)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Daily drawdown chart
                    if len(daily_equity) > 0:
                        fig_dd = plot_daily_drawdown(
                            daily_equity,
                            daily_equity_bench=spy_daily_equity if len(spy_daily_equity) > 0 else None,
                            balanced_data=balanced_data,
                        )
                        if fig_dd is not None:
                            st.pyplot(fig_dd)
                            plt.close(fig_dd)

                    st.subheader("Annual Returns")
                    fig2 = plot_annual_returns(strat_returns, bench_returns)
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # ---- STRATEGY HEALTH ----
                    st.subheader("Strategy Health")
                    st.caption(
                        "Rolling diagnostics to assess whether strategy performance "
                        "is holding up or deteriorating over time. All rolling windows are 36 months "
                        "unless otherwise noted."
                    )

                    fig_rs = plot_rolling_sortino(strat_returns, bench_returns, rf_monthly)
                    st.pyplot(fig_rs)
                    plt.close(fig_rs)

                    fig_rcal = plot_rolling_calmar(
                        strat_returns,
                        bench_returns,
                        daily_equity_strat=daily_equity if len(daily_equity) > 0 else None,
                        daily_equity_bench=spy_daily_equity if len(spy_daily_equity) > 0 else None,
                    )
                    st.pyplot(fig_rcal)
                    plt.close(fig_rcal)

                    fig_rc = plot_rolling_cagr(strat_returns, bench_returns)
                    st.pyplot(fig_rc)
                    plt.close(fig_rc)

                    fig_rh = plot_rolling_hit_rate(strat_returns, bench_returns)
                    st.pyplot(fig_rh)
                    plt.close(fig_rh)

                    fig_uw = plot_underwater(strat_returns, bench_returns)
                    st.pyplot(fig_uw)
                    plt.close(fig_uw)

                    fig_re = plot_rolling_excess(strat_returns, bench_returns)
                    st.pyplot(fig_re)
                    plt.close(fig_re)

                    # Leverage timeline (only shown when leverage is enabled)
                    if leverage_params_ui.get("enabled", False):
                        fig_lev = plot_leverage_timeline(trade_df, leverage_params_ui)
                        if fig_lev is not None:
                            st.pyplot(fig_lev)
                            plt.close(fig_lev)

                    # ---- TRADE LOG ----
                    st.subheader("Trade Log")
                    st.caption("Most recent trades shown first. Expand to see all monthly rebalance decisions.")
                    trade_df_display = trade_df.iloc[::-1].reset_index(drop=True)
                    with st.expander("Show full trade log"):
                        st.dataframe(trade_df_display, use_container_width=True, hide_index=True)

                    # ---- DOWNLOADS ----
                    st.subheader("Downloads")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "Download Trade Log (CSV)",
                            data=trade_df.to_csv(index=False),
                            file_name="Trade Log.csv",
                            mime="text/csv",
                        )
                    with col_dl2:
                        export_df = pd.DataFrame({
                            "date": strat_returns.index,
                            "strategy_return": strat_returns.values,
                            "benchmark_return": bench_returns.values,
                        })
                        export_df["strategy_cumulative"] = INITIAL_CAPITAL * (1 + export_df["strategy_return"]).cumprod()
                        export_df["benchmark_cumulative"] = INITIAL_CAPITAL * (1 + export_df["benchmark_return"]).cumprod()
                        st.download_button(
                            "Download Monthly Returns (CSV)",
                            data=export_df.to_csv(index=False),
                            file_name="Monthly Returns.csv",
                            mime="text/csv",
                        )

                    # ---- CONFIG SUMMARY & DATA NOTES (at end of dashboard) ----
                    st.markdown("---")
                    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                    with col_cfg1:
                        st.markdown(f"**Fixed basket:** {', '.join(selected_fixed) if selected_fixed else 'None'}")
                        if backfill_rules:
                            bf_parts = [f"{k} <- {', '.join(v)}" for k, v in backfill_rules.items()]
                            st.markdown(f"**Backfills:** {'; '.join(bf_parts)}")
                    with col_cfg2:
                        eq_display = f"Top {equity_top_n} from {', '.join(selected_equity)}" if equity_top_n > 0 else "None"
                        st.markdown(f"**Equity selection:** {eq_display}")
                    with col_cfg3:
                        st.markdown(f"**Final:** Top {final_top_n} by {main_lookback_label}, risk parity ({vol_months}mo vol)")
                        if leverage_params_ui.get("enabled", False):
                            lp = leverage_params_ui
                            if lp.get("method") == "constant":
                                cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                                cost_str = f", cost {cost_pct:.1f}%" if cost_pct > 0 else ""
                                st.markdown(
                                    f"**Leverage:** Constant {lp.get('mult_constant', 1.25):.2f}x{cost_str}"
                                )
                            elif lp.get("method") == "holdings_gate":
                                defensives = ", ".join(lp.get("defensive_tickers", []))
                                cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                                cost_str = f", cost {cost_pct:.1f}%" if cost_pct > 0 else ""
                                st.markdown(
                                    f"**Leverage:** Holdings gate ({defensives}), "
                                    f"{lp.get('mult_leveraged', 1.25):.2f}x when clear{cost_str}"
                                )
                            elif lp.get("method") == "vt_trend":
                                st.markdown(
                                    f"**Leverage:** {lp['ticker']} {lp['long_sma']}/{lp['short_sma']}d SMA, "
                                    f"on={lp['mult_risk_on']:.2f}x / off={lp['mult_risk_off']:.2f}x"
                                )
                        if btc_gate_config.get("enabled"):
                            bg_mode = (
                                "long-position" if btc_gate_config["mode"] == "position"
                                else "trend-up"
                            )
                            st.markdown(
                                f"**BTC gate:** {gate_label(btc_gate_config)} Radius Trend "
                                f"({bg_mode}), step {btc_gate_config['step']:.3f}, "
                                f"dist {btc_gate_config['multi']:.2f}"
                            )
                        if btc_sma_gate_config.get("enabled"):
                            sg_mode = (
                                "long-position" if btc_sma_gate_config["mode"] == "position"
                                else "trend-up"
                            )
                            st.markdown(
                                f"**BTC SMA gate:** {gate_label(btc_sma_gate_config)} SMA cross "
                                f"({sg_mode}), SMA {btc_sma_gate_config['sma_len']}d, "
                                f"rf {btc_sma_gate_config['rf_len']}d/"
                                f"{btc_sma_gate_config['rf_period']}d"
                            )

                    # ---- BACKFILL DETAILS & DATA NOTES (at end of dashboard) ----
                    if backfill_info:
                        st.markdown("---")
                        with st.expander("Backfill details (which ticker covers which dates)"):
                            st.dataframe(pd.DataFrame(backfill_info), use_container_width=True, hide_index=True)

                    lev_note = ""
                    if leverage_params_ui.get("enabled", False):
                        lp = leverage_params_ui
                        if lp.get("method") == "constant":
                            cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                            cost_str = f" Borrowing cost: {cost_pct:.1f}% annual on leveraged portion." if cost_pct > 0 else ""
                            lev_note = (
                                f" Constant {lp.get('mult_constant', 1.25):.2f}x leverage applied "
                                f"to strategy and benchmarks.{cost_str}"
                            )
                        elif lp.get("method") == "holdings_gate":
                            defensives = ", ".join(lp.get("defensive_tickers", []))
                            cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                            cost_str = f" Borrowing cost: {cost_pct:.1f}% annual on leveraged portion." if cost_pct > 0 else ""
                            lev_note = (
                                f" Dynamic leverage: {lp.get('mult_leveraged', 1.25):.2f}x when no "
                                f"defensive assets ({defensives}) are in the final portfolio, "
                                f"1.0x otherwise.{cost_str}"
                            )
                        elif lp.get("method") == "vt_trend":
                            lev_note = (
                                " Dynamic leverage scales monthly returns by a multiplier "
                                f"({lp['mult_risk_off']:.2f}x to {lp['mult_risk_on']:.2f}x) "
                                f"based on {lp['ticker']} trend regime."
                            )
                    gate_note = ""
                    if btc_gate_config.get("enabled"):
                        bg_mode = (
                            "long-position state" if btc_gate_config["mode"] == "position"
                            else "trend-up state"
                        )
                        gate_note = (
                            f" Bitcoin trend gate: {btc_gate_config['ticker']} is only "
                            f"eligible for the basket when the "
                            f"{gate_signal_ticker(btc_gate_config)} Radius Trend "
                            f"{bg_mode} is in the market (Radius Step "
                            f"{btc_gate_config['step']:.3f}, Start Points Distance "
                            f"{btc_gate_config['multi']:.2f}, computed on daily bars). "
                            f"This is a close approximation of the TradingView "
                            f"indicator and may differ slightly due to data-feed alignment."
                        )
                    sma_gate_note = ""
                    if btc_sma_gate_config.get("enabled"):
                        sg_mode = (
                            "long-position state" if btc_sma_gate_config["mode"] == "position"
                            else "trend-up state"
                        )
                        sma_gate_note = (
                            f" Bitcoin SMA cross gate: {btc_sma_gate_config['ticker']} is "
                            f"only eligible for the basket when the "
                            f"{gate_signal_ticker(btc_sma_gate_config)} SMA cross "
                            f"{sg_mode} is in the market ({btc_sma_gate_config['sma_len']}-day "
                            f"long SMA, {btc_sma_gate_config['rf_len']}-day rising/falling SMA "
                            f"over {btc_sma_gate_config['rf_period']} bars, computed on daily "
                            f"bars as a conversion of the original 8-hour strategy). When both "
                            f"Bitcoin gates are enabled, the ticker must pass both."
                        )
                    if exec_timing == "last_day":
                        exec_note = (
                            "Allocations computed from the close of the trading day before "
                            "the last trading day of each month; trades execute at the last "
                            "trading day's close (no look-ahead). "
                        )
                    else:
                        exec_note = (
                            "Signals computed at month-end close; trades execute at next "
                            "trading day's close (1-day lag). "
                        )
                    st.caption(
                        "All prices are dividend- and split-adjusted (total return). "
                        + exec_note +
                        "Sortino ratio uses BIL as the risk-free rate."
                        + lev_note
                        + gate_note
                        + sma_gate_note
                    )

else:
    st.info("Configure parameters in the sidebar, then click **Run Backtest** to see results.")

    if os.path.exists(RESULTS_FILE):
        st.subheader("Previous Backtest Results")
        st.caption(f"Loaded from {os.path.basename(RESULTS_FILE)}")
        try:
            log_df = pd.read_excel(RESULTS_FILE, engine="openpyxl")
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not read results log: {e}")

        if st.button("Clear Results Log", type="secondary"):
            try:
                os.remove(RESULTS_FILE)
                st.success("Results log cleared. Refresh the page to confirm.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete file (it may be open in Excel): {e}")