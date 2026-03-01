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
# CONSTANTS
# =============================================================================

DEFAULT_FIXED_ASSETS = {
    "TLT":     "Bonds (20+ Year Treasury)",
    "GLD":     "Gold",
    "BIL":     "Cash (1-3 Month T-Bill)",
    "BTAL":    "Defensive Equity Factor",
    "HGER":    "Commodity (All-Weather)",
    "BTC-USD": "Bitcoin",
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
    "6/8-month blend (50/50)":    ([6, 8],    [0.50, 0.50]),
    "1/8-month blend (50/50)":    ([1, 8],    [0.50, 0.50]),
    "2/8-month blend (50/50)":    ([2, 8],    [0.50, 0.50]),
    "2/11-month blend (50/50)":   ([2, 11],   [0.50, 0.50]),
}

BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 10_000

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
            for col in yahoo_result.columns:
                cached[col] = yahoo_result[col]
            save_price_cache(cached.sort_index())
        else:
            save_price_cache(yahoo_result)

    progress.caption(f"Cache: {len(unique_tickers)} tickers loaded ({len(local_tickers)} from local files).")

    available = [t for t in unique_tickers if t in result.columns]
    return result[available]


def build_backfilled_series(daily_prices, primary_ticker, backfill_tickers):
    """
    Build a single price series for primary_ticker, extending it backward
    using backfill tickers (in priority order) where the primary has no data.

    The backfill uses return-splicing: we compute daily returns from the
    backfill ETF and chain them onto the primary's earliest known price,
    so the price levels connect smoothly.

    Returns the spliced series and a list of segment descriptions.
    """
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

        # Fill in the earlier dates (exclude the overlap date itself)
        fill_dates = bf_scaled.index[bf_scaled.index < earliest_combined]
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


def run_backtest(daily_prices, monthly_prices, params):
    """
    Run the adaptive portfolio backtest with the given parameters.

    Execution timing: signals are computed from month-end closing prices.
    Trades execute at the NEXT trading day's close (1-day lag) to avoid
    look-ahead bias from same-day signal + execution.
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

    min_lookback = max(main_lookbacks)
    valid_months = monthly_prices.index[min_lookback:]

    # Build sorted array of all trading dates for execution-day lookups
    all_trading_dates = daily_prices.index.sort_values()

    def next_trading_day(date):
        """Return the first trading day strictly AFTER the given date."""
        future = all_trading_dates[all_trading_dates > date]
        if len(future) == 0:
            return None
        return future[0]

    results = []
    trade_log = []
    daily_portfolio_values = {}  # {date: portfolio_value} for daily drawdown
    cumulative_multiplier = 1.0  # tracks growth across holding periods

    for i, rebal_date in enumerate(valid_months):
        if i >= len(valid_months) - 1:
            break

        next_month_end = valid_months[i + 1]

        # Build basket
        basket = []

        for t in fixed_tickers:
            if t in monthly_prices.columns:
                ps = monthly_prices[t].loc[:rebal_date].dropna()
                if len(ps) > min_lookback:
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
                eq_subset = monthly_prices[eq_available].loc[:rebal_date]
                eq_mom = compute_blended_momentum(eq_subset, equity_lookbacks, equity_lookback_weights)
                latest_eq = eq_mom.iloc[-1].dropna().sort_values(ascending=False)
                selected_equities = latest_eq.head(equity_top_n).index.tolist()

            basket.extend(selected_equities)

        if len(basket) == 0:
            continue

        basket_monthly = monthly_prices[basket].loc[:rebal_date].copy()
        basket_daily = daily_prices[basket].copy()

        basket_mom = compute_blended_momentum(basket_monthly, main_lookbacks, main_lookback_weights)
        latest_basket = basket_mom.iloc[-1].dropna().sort_values(ascending=False)

        n_select = min(final_top_n, len(latest_basket))
        final_holdings = latest_basket.head(n_select).index.tolist()

        if len(final_holdings) == 0:
            continue

        weights = compute_risk_parity_weights(
            basket_daily, final_holdings, rebal_date, vol_months
        )

        # Execution prices: use the next trading day AFTER each month-end
        # to model realistic 1-day execution lag
        exec_date_entry = next_trading_day(rebal_date)
        exec_date_exit = next_trading_day(next_month_end)

        if exec_date_entry is None or exec_date_exit is None:
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

        # --- Daily equity tracking for this holding period ---
        holding_mask = (
            (daily_prices.index >= exec_date_entry)
            & (daily_prices.index <= exec_date_exit)
        )
        holding_days = daily_prices.index[holding_mask]

        # Entry prices for each held asset (computed once per period)
        entry_px = {}
        for t in final_holdings:
            ep = daily_prices[t].loc[:exec_date_entry].dropna()
            if len(ep) > 0 and ep.iloc[-1] != 0:
                entry_px[t] = ep.iloc[-1]

        for day in holding_days:
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
            daily_portfolio_values[day] = (
                cumulative_multiplier * (1 + day_weighted_return) * INITIAL_CAPITAL
            )

        # Update cumulative multiplier at period end
        cumulative_multiplier *= (1 + port_return)

        results.append({
            "date": next_month_end,
            "portfolio_return": port_return,
        })

        holdings_str_parts = []
        for t in final_holdings:
            if t in holding_details:
                w = weights.get(t, 0.0)
                r = holding_details[t]["return"]
                holdings_str_parts.append(f"{t}: {w:.1%} (ret: {r:+.2%})")

        trade_log.append({
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "hold_month": next_month_end.strftime("%Y-%m"),
            "exec_entry": exec_date_entry.strftime("%Y-%m-%d"),
            "exec_exit": exec_date_exit.strftime("%Y-%m-%d") if i < len(valid_months) - 2 else "",
            "basket": ", ".join(basket),
            "equities_selected": ", ".join(selected_equities) if selected_equities else "N/A",
            "final_holdings": " | ".join(holdings_str_parts),
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


def compute_current_signals(daily_prices, monthly_prices, params):
    """
    Compute model signals as of the most recent month-end.
    Returns a dict with signal_date, asset details table, equity selections,
    final holdings, and weights.
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

    min_lookback = max(main_lookbacks)
    signal_date = monthly_prices.index[-1]

    # Build basket (same logic as backtest)
    basket = []
    for t in fixed_tickers:
        if t in monthly_prices.columns:
            ps = monthly_prices[t].loc[:signal_date].dropna()
            if len(ps) > min_lookback:
                basket.append(t)

    selected_equities = []
    if equity_top_n > 0 and len(equity_tickers) > 0:
        eq_available = []
        for t in equity_tickers:
            if t in monthly_prices.columns:
                ps = monthly_prices[t].loc[:signal_date].dropna()
                min_eq_history = max(max(equity_lookbacks), min_lookback)
                if len(ps) > min_eq_history:
                    eq_available.append(t)

        eq_scores = {}
        if len(eq_available) >= equity_top_n:
            eq_subset = monthly_prices[eq_available].loc[:signal_date]
            eq_mom = compute_blended_momentum(eq_subset, equity_lookbacks, equity_lookback_weights)
            latest_eq = eq_mom.iloc[-1].dropna().sort_values(ascending=False)
            selected_equities = latest_eq.head(equity_top_n).index.tolist()
            eq_scores = latest_eq.to_dict()

        basket.extend(selected_equities)

    if len(basket) == 0:
        return None

    # Compute individual lookback scores for every asset in the full universe
    all_assets = list(set(fixed_tickers + equity_tickers))
    all_assets = [t for t in all_assets if t in monthly_prices.columns]

    asset_details = []
    for t in all_assets:
        ps = monthly_prices[t].loc[:signal_date].dropna()
        if len(ps) < min_lookback + 1:
            continue

        row = {"Asset": t, "In Basket": t in basket}

        # Individual main lookback scores
        for lb in sorted(set(main_lookbacks)):
            ret = compute_total_return(monthly_prices[[t]].loc[:signal_date], lb)
            val = ret.iloc[-1].values[0] if len(ret) > 0 else np.nan
            row[f"{lb}mo Return"] = val

        # Blended main score
        basket_sub = monthly_prices[[t]].loc[:signal_date]
        blended = compute_blended_momentum(basket_sub, main_lookbacks, main_lookback_weights)
        blended_val = blended.iloc[-1].values[0] if len(blended) > 0 else np.nan
        row["Blended Score"] = blended_val

        # Annualized volatility
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
    basket_monthly = monthly_prices[basket].loc[:signal_date].copy()
    basket_mom = compute_blended_momentum(basket_monthly, main_lookbacks, main_lookback_weights)
    latest_basket = basket_mom.iloc[-1].dropna().sort_values(ascending=False)

    n_select = min(final_top_n, len(latest_basket))
    final_holdings = latest_basket.head(n_select).index.tolist()

    # Risk parity weights
    weights = compute_risk_parity_weights(
        daily_prices, final_holdings, signal_date, vol_months
    )

    # Mark selected and add weights to details
    for row in asset_details:
        t = row["Asset"]
        row["Selected"] = t in final_holdings
        row["Weight"] = weights.get(t, 0.0) if t in final_holdings else 0.0
        row["Equity Pick"] = t in selected_equities

    # Sort by blended score descending
    asset_details.sort(key=lambda x: x.get("Blended Score", -999), reverse=True)

    return {
        "signal_date": signal_date,
        "asset_details": asset_details,
        "selected_equities": selected_equities,
        "final_holdings": final_holdings,
        "weights": weights,
        "basket": basket,
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

def plot_equity_and_drawdown(strat_returns, bench_returns, balanced_returns=None):
    common_idx = strat_returns.index.intersection(bench_returns.index)
    strat = strat_returns.loc[common_idx]
    bench = bench_returns.loc[common_idx]

    strat_cum = INITIAL_CAPITAL * (1 + strat).cumprod()
    bench_cum = INITIAL_CAPITAL * (1 + bench).cumprod()

    # 60/40 if available
    bal_cum = None
    if balanced_returns is not None:
        bal = balanced_returns.reindex(common_idx).dropna()
        if len(bal) > 0:
            bal_cum = INITIAL_CAPITAL * (1 + bal).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Adaptive Strategy vs. SPY vs. 60/40", fontsize=14, fontweight="bold", y=0.97)

    ax1 = axes[0]
    ax1.plot(strat_cum.index, strat_cum.values, label="Strategy", linewidth=1.8, color="#1a5276")
    ax1.plot(bench_cum.index, bench_cum.values, label="SPY", linewidth=1.2, color="#aab7b8", alpha=0.8)
    if bal_cum is not None:
        ax1.plot(bal_cum.index, bal_cum.values, label="60/40", linewidth=0.9, color="#d4a574", alpha=0.55, linestyle="--")
    ax1.set_ylabel(f"Growth of ${INITIAL_CAPITAL:,}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2 = axes[1]
    wealth = (1 + strat).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    ax2.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4, label="Strategy")

    wealth_b = (1 + bench).cumprod()
    peak_b = wealth_b.cummax()
    dd_b = (wealth_b - peak_b) / peak_b
    ax2.fill_between(dd_b.index, dd_b.values, 0, color="#aab7b8", alpha=0.3, label="SPY")

    if balanced_returns is not None:
        bal = balanced_returns.reindex(common_idx).dropna()
        if len(bal) > 0:
            wealth_bal = (1 + bal).cumprod()
            peak_bal = wealth_bal.cummax()
            dd_bal = (wealth_bal - peak_bal) / peak_bal
            ax2.fill_between(dd_bal.index, dd_bal.values, 0, color="#d4a574", alpha=0.2, label="60/40")

    ax2.set_ylabel("Drawdown")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    return fig


def plot_daily_drawdown(daily_equity, daily_equity_bench=None, daily_equity_balanced=None):
    """Plot daily drawdown chart from daily equity curves."""
    if daily_equity is None or len(daily_equity) == 0:
        return None

    peak = daily_equity.cummax()
    dd = (daily_equity - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Daily Drawdown (close-to-close)", fontsize=14, fontweight="bold")
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4, label="Strategy")

    if daily_equity_bench is not None and len(daily_equity_bench) > 0:
        peak_b = daily_equity_bench.cummax()
        dd_b = (daily_equity_bench - peak_b) / peak_b
        ax.fill_between(dd_b.index, dd_b.values, 0, color="#aab7b8", alpha=0.3, label="SPY")

    if daily_equity_balanced is not None and len(daily_equity_balanced) > 0:
        peak_bal = daily_equity_balanced.cummax()
        dd_bal = (daily_equity_balanced - peak_bal) / peak_bal
        ax.fill_between(dd_bal.index, dd_bal.values, 0, color="#d4a574", alpha=0.2, label="60/40")

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
        label="Strategy", color="#2e4057", edgecolor="white", linewidth=0.5
    )
    bars_bench = ax.bar(
        x + bar_width / 2, bench_yearly.values, bar_width,
        label="SPY", color="#62c4b2", edgecolor="white", linewidth=0.5
    )

    # Add value labels on each bar
    for bar in bars_strat:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{h:.1%}", ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=7, fontweight="bold", color="#2e4057"
        )
    for bar in bars_bench:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{h:.1%}", ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=7, fontweight="bold", color="#62c4b2"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

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

    ax.plot(strat_36.index, strat_36.values, label="Strategy (36mo)", linewidth=2.0, color="#1a5276")
    ax.plot(strat_24.index, strat_24.values, label="Strategy (24mo)", linewidth=1.0, color="#5dade2", alpha=0.5)
    ax.plot(bench_36.index, bench_36.values, label="SPY (36mo)", linewidth=1.8, color="#e67e22", linestyle="--")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.0, color="#999", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylabel("Sortino Ratio")
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

    ax.plot(strat_36.index, strat_36.values, label="Strategy", linewidth=2.0, color="#1a5276")
    ax.plot(bench_36.index, bench_36.values, label="SPY", linewidth=1.8, color="#e67e22", linestyle="--")

    ax.axhline(0, color="black", linewidth=0.5)
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

    ax.plot(strat_hr.index, strat_hr.values, label="Strategy", linewidth=2.0, color="#1a5276")
    ax.plot(bench_hr.index, bench_hr.values, label="SPY", linewidth=1.8, color="#e67e22", linestyle="--")

    ax.axhline(0.5, color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.5, label="50% line")
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
    ax1.bar(uw_strat.index, uw_strat.values, width=25, color="#1a5276", alpha=0.7, label="Strategy")
    ax1.bar(uw_bench.index, uw_bench.values, width=25, color="#e67e22", alpha=0.35, label="SPY")
    ax1.set_ylabel("Months Since\nEquity High")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(dd_strat.index, dd_strat.values, 0, color="#1a5276", alpha=0.4, label="Strategy")
    ax2.fill_between(dd_bench.index, dd_bench.values, 0, color="#e67e22", alpha=0.25, label="SPY")
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
                    where=(excess.values >= 0), color="#27ae60", alpha=0.4, interpolate=True)
    ax.fill_between(excess.index, excess.values, 0,
                    where=(excess.values < 0), color="#c0392b", alpha=0.4, interpolate=True)
    ax.plot(excess.index, excess.values, color="#2c3e50", linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Excess Annualized Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

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
    index=2,
)
main_lookbacks, main_lookback_weights = MAIN_LOOKBACK_OPTIONS[main_lookback_label]

final_top_n = st.sidebar.radio(
    "How many assets to hold in the final portfolio?",
    options=[3, 4, 5, 6],
    index=1,
    horizontal=True,
)

vol_months = st.sidebar.slider(
    "Risk parity volatility window (months)",
    min_value=3,
    max_value=24,
    value=12,
    step=1,
)

# ---- RUN BACKTEST (second position) ----
st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# ---- PERIOD (third position) ----
st.sidebar.markdown("---")
st.sidebar.header("Backtest Period")

backtest_start = st.sidebar.text_input(
    "Backtest start (YYYY-MM)",
    value="2002-01",
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
    ("HGER",    "QCI"),
    ("BTAL",    "VMNFX"),
    ("BIL",     "VFISX"),
    ("TLT",     "VUSTX"),
    ("GLD",     "GOLD"),
    ("QQQ",     "FSPTX"),
    ("IWM",     "OTCFX"),
    ("EMXC",    "EEM, VEIEX"),
    ("DXJ",     "FJPNX"),
    ("",        ""),
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

st.title("Adaptive Portfolio Strategy Dashboard")

st.markdown("---")


if run_button:
    if len(selected_fixed) == 0 and equity_top_n == 0:
        st.error("You need at least some assets in the basket. Select fixed assets or enable equity sub-selection.")
    else:
        # Gather ALL tickers needed (including backfills)
        all_tickers_needed = list(selected_fixed) + list(selected_equity) + [BENCHMARK_TICKER, "BIL", "VBIAX"]
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
                    spliced, segments = build_backfilled_series(daily_prices, primary, chain)
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
                    "backtest_start": backtest_start,
                    "backtest_end": backtest_end,
                    "backfill_summary": backfill_summary,
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

                    # 60/40 benchmark (VBIAX)
                    balanced_returns = None
                    balanced_metrics = {}
                    if "VBIAX" in monthly_prices.columns:
                        balanced_monthly = monthly_prices["VBIAX"].pct_change()
                        balanced_returns = balanced_monthly.reindex(common_dates)

                    # Risk-free rate from BIL for Sharpe/Sortino calculation
                    rf_monthly = None
                    if "BIL" in monthly_prices.columns:
                        rf_monthly = monthly_prices["BIL"].pct_change().loc[common_dates]

                    # Build daily equity curves for benchmarks (for daily drawdown)
                    spy_daily_equity = pd.Series(dtype=float)
                    balanced_daily_equity = pd.Series(dtype=float)
                    if len(daily_equity) > 0:
                        de_start, de_end = daily_equity.index[0], daily_equity.index[-1]
                        if BENCHMARK_TICKER in daily_prices.columns:
                            spy_daily = daily_prices[BENCHMARK_TICKER].loc[de_start:de_end].dropna()
                            if len(spy_daily) > 0:
                                spy_daily_equity = INITIAL_CAPITAL * (spy_daily / spy_daily.iloc[0])
                        if "VBIAX" in daily_prices.columns:
                            bal_daily = daily_prices["VBIAX"].loc[de_start:de_end].dropna()
                            if len(bal_daily) > 0:
                                balanced_daily_equity = INITIAL_CAPITAL * (bal_daily / bal_daily.iloc[0])

                    strat_metrics = compute_metrics(strat_returns, rf_monthly=rf_monthly, daily_equity=daily_equity)
                    bench_metrics = compute_metrics(bench_returns, rf_monthly=rf_monthly, daily_equity=spy_daily_equity)
                    if balanced_returns is not None:
                        balanced_metrics = compute_metrics(balanced_returns.dropna(), rf_monthly=rf_monthly, daily_equity=balanced_daily_equity)

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

                    # Ordered with key metrics first; Sharpe and Total Months removed
                    metrics_rows = [
                        ("Sortino Ratio", fmt_ratio(strat_metrics.get("Sortino Ratio", 0)), fmt_ratio(bench_metrics.get("Sortino Ratio", 0)), fmt_ratio(balanced_metrics.get("Sortino Ratio", 0)), True),
                        ("CAGR", fmt_pct(strat_metrics.get("CAGR", 0)), fmt_pct(bench_metrics.get("CAGR", 0)), fmt_pct(balanced_metrics.get("CAGR", 0)), True),
                        ("Max Drawdown", fmt_pct(strat_metrics.get("Max Drawdown", 0)), fmt_pct(bench_metrics.get("Max Drawdown", 0)), fmt_pct(balanced_metrics.get("Max Drawdown", 0)), True),
                        ("Max DD (Daily)", fmt_pct(strat_metrics.get("Max Drawdown (Daily)", 0)), fmt_pct(bench_metrics.get("Max Drawdown (Daily)", 0)), fmt_pct(balanced_metrics.get("Max Drawdown (Daily)", 0)), True),
                        ("Calmar Ratio", fmt_ratio(strat_metrics.get("Calmar Ratio", 0)), fmt_ratio(bench_metrics.get("Calmar Ratio", 0)), fmt_ratio(balanced_metrics.get("Calmar Ratio", 0)), False),
                        ("Annualized Vol", fmt_pct(strat_metrics.get("Annualized Vol", 0)), fmt_pct(bench_metrics.get("Annualized Vol", 0)), fmt_pct(balanced_metrics.get("Annualized Vol", 0)), False),
                        ("Best Year", fmt_pct(strat_metrics.get("Best Year", 0)), fmt_pct(bench_metrics.get("Best Year", 0)), fmt_pct(balanced_metrics.get("Best Year", 0)), False),
                        ("Worst Year", fmt_pct(strat_metrics.get("Worst Year", 0)), fmt_pct(bench_metrics.get("Worst Year", 0)), fmt_pct(balanced_metrics.get("Worst Year", 0)), False),
                        ("Win Rate", fmt_pct(strat_metrics.get("Win Rate (monthly)", 0)), fmt_pct(bench_metrics.get("Win Rate (monthly)", 0)), fmt_pct(balanced_metrics.get("Win Rate (monthly)", 0)), False),
                        ("Total Return", fmt_pct(strat_metrics.get("Total Return", 0)), fmt_pct(bench_metrics.get("Total Return", 0)), fmt_pct(balanced_metrics.get("Total Return", 0)), False),
                        (f"Growth of ${INITIAL_CAPITAL:,}", fmt_dollar(strat_metrics.get(f"Growth of ${INITIAL_CAPITAL:,}", 0)), fmt_dollar(bench_metrics.get(f"Growth of ${INITIAL_CAPITAL:,}", 0)), fmt_dollar(balanced_metrics.get(f"Growth of ${INITIAL_CAPITAL:,}", 0)), False),
                        ("Period", strat_metrics.get("Period", ""), bench_metrics.get("Period", ""), balanced_metrics.get("Period", ""), False),
                    ]

                    # Build HTML table with bold top 3 rows and 60/40 column
                    html_parts = [
                        '<style>',
                        '.metrics-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px; }',
                        '.metrics-table th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #ddd; background-color: #f8f9fa; }',
                        '.metrics-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }',
                        '.metrics-table tr:hover { background-color: #f5f5f5; }',
                        '.bold-row td { font-weight: 700; }',
                        '</style>',
                        '<table class="metrics-table">',
                        '<thead><tr><th>Metric</th><th>Strategy</th><th>SPY</th><th>60/40 (VBIAX)</th></tr></thead>',
                        '<tbody>',
                    ]
                    for metric, strat_val, bench_val, bal_val, is_bold in metrics_rows:
                        row_class = ' class="bold-row"' if is_bold else ""
                        html_parts.append(f'<tr{row_class}><td>{metric}</td><td>{strat_val}</td><td>{bench_val}</td><td>{bal_val}</td></tr>')
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

                    # ---- CURRENT SIGNALS ----
                    signals = compute_current_signals(daily_prices, monthly_prices, params)
                    if signals is not None:
                        st.subheader("Current Model Signals")
                        sig_date = signals["signal_date"]
                        st.caption(f"As of {sig_date.strftime('%m/%d/%Y')} (most recent month-end close)")

                        # Holdings summary
                        holdings_parts = []
                        for t in signals["final_holdings"]:
                            w = signals["weights"].get(t, 0)
                            holdings_parts.append(f"{w:.2%} {t}")
                        st.markdown("**Current holdings:** " + " | ".join(holdings_parts))

                        if signals["selected_equities"]:
                            st.markdown(
                                "**Equity picks this month:** "
                                + ", ".join(signals["selected_equities"])
                            )

                        # Details table
                        details = signals["asset_details"]
                        vol_months_label = f"{params['vol_months']}mo Vol (ann)"

                        # Build lookback column headers
                        lb_cols = sorted(set(params["main_lookbacks"]))
                        lb_headers = [f"{lb}mo Return" for lb in lb_cols]

                        sig_html = [
                            '<style>',
                            '.sig-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px; }',
                            '.sig-table th { text-align: left; padding: 6px 10px; border-bottom: 2px solid #ddd; background-color: #f8f9fa; }',
                            '.sig-table td { padding: 6px 10px; border-bottom: 1px solid #eee; }',
                            '.sig-table tr:hover { background-color: #f5f5f5; }',
                            '.sig-selected { background-color: #eaf4e8; }',
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

                    # ---- CHARTS ----
                    st.subheader("Equity Curve and Drawdown")
                    fig = plot_equity_and_drawdown(strat_returns, bench_returns, balanced_returns)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Daily drawdown chart
                    if len(daily_equity) > 0:
                        fig_dd = plot_daily_drawdown(
                            daily_equity,
                            daily_equity_bench=spy_daily_equity if len(spy_daily_equity) > 0 else None,
                            daily_equity_balanced=balanced_daily_equity if len(balanced_daily_equity) > 0 else None,
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

                    # ---- BACKFILL DETAILS & DATA NOTES (at end of dashboard) ----
                    if backfill_info:
                        st.markdown("---")
                        with st.expander("Backfill details (which ticker covers which dates)"):
                            st.dataframe(pd.DataFrame(backfill_info), use_container_width=True, hide_index=True)

                    st.caption(
                        "All prices are dividend- and split-adjusted (total return). "
                        "Signals computed at month-end close; trades execute at next trading day's close (1-day lag). "
                        "Sortino ratio uses BIL as the risk-free rate."
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