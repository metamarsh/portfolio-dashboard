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

Trading costs (November 1, 2026): every rebalance is charged the sidebar's
"Trading cost per side" (default 0.05%) times one-way turnover, measured
against the drifted weights, so strategy returns are net of costs. Set it
to 0 to reproduce the earlier gross results exactly.

If Yahoo returns no prices for a ticker the cache already holds (an
outage), the cached series is used and a warning is shown. A ticker with
no prices anywhere stops the run with an error instead of dropping out
silently.
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
import plotly.graph_objects as go
import json
import os
import re
import sys
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
# GLOBAL STYLING (Asymmetric Edge design language; page facelift November 17, 2026)
# =============================================================================
# The page follows asymmetricedge.io and the public Chart Dashboard: IBM Plex
# Serif 600 for titles and section headings, Inter for everything else, the
# warm neutral set (ink, body, rules, wash) and the brand purple accent. The
# Streamlit theme in .streamlit/config.toml carries the same fonts and colors
# into the sidebar's controls; change both together. Charts are Plotly and
# read these tokens through ae_layout() below. The matplotlib rcParams further
# down are kept for the plot_* functions that research scripts still load.

# ---- Shared design tokens ----
AE_INK = "#1C1B19"              # Titles, figures, table text
AE_BODY = "#55524D"             # Body text, labels, captions
AE_AXIS = "#6B6B7B"             # Chart axis and tick text
AE_RULE = "#E6E3DC"             # Dividers and table rules
AE_CONTROL = "#D9D5CC"          # Pills, tags, control borders
AE_CARD = "#E2DED6"             # Card borders
AE_WASH = "#F5F4F0"             # Light fill: table heads, notes, hover
AE_TINT = "#EFEDF6"             # Purple tint: held rows, "now" chips
AE_GAIN = "#1F8A4C"             # Gains
AE_LOSS = "#B8432F"             # Losses
AE_PLOT_GRID = "#EEEDE9"        # Chart gridlines
AE_SERIF = "'IBM Plex Serif', Georgia, 'Times New Roman', serif"
AE_SANS = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
# Older names, still read by the matplotlib defaults below
AE_TEXT_PRIMARY = AE_INK
AE_TEXT_SECONDARY = AE_BODY
AE_TEXT_MUTED = AE_AXIS
AE_GRID = AE_PLOT_GRID
AE_ZERO_LINE = AE_CONTROL
AE_SPINE = AE_CONTROL

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=Inter:wght@400;500;600;700&display=swap');

    /* Trim the busiest chrome but keep the run/stop toolbar and menu usable */
    footer {visibility: hidden;}
    [data-testid="stDeployButton"] {display: none;}

    html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"] {
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif;
        color: #1C1B19;
    }
    .stApp { background-color: #ffffff; }
    .block-container { max-width: 1180px; padding-top: 2.6rem; }

    /* Headings: IBM Plex Serif 600, as on asymmetricedge.io */
    h1, h2, h3 {
        font-family: 'IBM Plex Serif', Georgia, serif !important;
        font-weight: 600 !important;
        color: #1C1B19 !important;
        letter-spacing: -0.01em;
    }
    h2, h3 { margin-top: 0.25rem !important; margin-bottom: 0.35rem !important; }
    [data-testid="stCaptionContainer"] { color: #55524D; }
    hr { border-color: #E6E3DC !important; }

    /* Brand bar */
    .ae-brand-header { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.8rem; }
    .ae-brand-header .ae-brand-name {
        font-family: 'IBM Plex Serif', Georgia, serif; font-size: 1.65rem; font-weight: 600;
        color: #1C1B19; letter-spacing: -0.01em;
    }
    .ae-brand-header .ae-brand-tag {
        font-size: 0.9rem; color: #55524D; border-left: 2px solid #4B3F8F; padding-left: 0.6rem;
    }
    .ae-brand-sub {
        font-size: 0.85rem; color: #55524D; margin: 0.35rem 0 0.4rem 0;
        padding-bottom: 1rem; border-bottom: 1px solid #E6E3DC;
    }

    /* Pills (if used): brand-purple active state */
    button[data-testid="stBaseButton-pills"][aria-checked="true"],
    button[data-variant="pills"][aria-checked="true"] {
        background-color: #EFEDF6 !important;
        border-color: #4B3F8F !important;
        color: #4B3F8F !important;
        font-weight: 600 !important;
    }

    /* Button labels keep the button's own color (the rule above would turn them ink) */
    button [data-testid="stMarkdownContainer"], button [data-testid="stMarkdownContainer"] p { color: inherit; }

    /* Results tabs: plain text tabs standing on a muted indigo rule, the selected one a solid
       block of the same color (mockups H, P1 and M1 of November 20, 2026), the bar held at the top
       while a tab scrolls. Selected by role, which both Streamlit's newer tabs and the older
       BaseWeb ones carry. The selected tab is marked by its fill, so the underline each Streamlit
       build draws is hidden: BaseWeb's highlight and border elements, the newer build's indicator
       inside each tab, and the rule under the newer build's bar. */
    [data-testid="stTabs"] [role="tablist"] {
        position: sticky; top: 3.75rem; z-index: 60; background: #ffffff; gap: 6px;
        padding: 6px 0 0 0; border-bottom: 3px solid #4B3F8F;
    }
    [data-testid="stTabs"] [role="tablist"]::after,
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"],
    [data-testid="stTabs"] [role="tab"] .react-aria-SelectionIndicator { display: none !important; }
    [data-testid="stTabs"] [role="tab"] {
        height: auto; padding: 8px 14px; background: transparent;
        border: none; border-radius: 8px 8px 0 0;
    }
    [data-testid="stTabs"] [role="tab"]:hover { background: #EFEDF6; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]:hover { background: #4B3F8F; padding: 8px 18px; }
    [data-testid="stTabs"] [role="tab"] p { font-size: 15px; font-weight: 600; color: #1C1B19; }
    [data-testid="stTabs"] [role="tab"]:hover p { color: #4B3F8F; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]:hover p { font-weight: 700; color: #ffffff; }
    [data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.2rem; }

    /* Bordered containers (chart cards), expanders and alerts in the warm set */
    [class*="st-key-aecard"] { border-color: #E2DED6 !important; border-radius: 10px !important; }
    [data-testid="stExpander"] details { border-color: #E2DED6; border-radius: 10px; }
    [data-testid="stExpander"] summary p { font-weight: 600; color: #1C1B19; }
    [data-testid="stAlert"] { border-radius: 10px; }

    /* ---- Results page building blocks (ae-*) ---- */
    .ae-status { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 18px; font-size: 13.5px; color: #55524D; margin: 0.2rem 0 0.4rem 0; }
    .ae-status .ae-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #1F8A4C; margin-right: 7px; vertical-align: 1px; }
    .ae-head { display: flex; flex-direction: column; gap: 8px; margin-bottom: 0.4rem; }
    .ae-eyebrow { font-size: 13px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #4B3F8F; }
    .ae-title { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 30px; line-height: 1.18; font-weight: 600; letter-spacing: -0.01em; color: #1C1B19; text-wrap: balance; }
    .ae-desc { max-width: 760px; font-size: 16px; line-height: 1.6; color: #55524D; }
    /* On this page: a segmented control with no label, in a muted indigo (mockups P1 and M1 of
       November 20, 2026). The first link is solid until a link is followed; then the one followed
       is (ae_head writes those rules, one per link). */
    .ae-otp { display: flex; flex-wrap: wrap; align-items: center; margin-top: 10px; }
    .ae-otp .ae-otp-seg { display: inline-flex; flex-wrap: wrap; gap: 3px; padding: 4px; background: #EFEDF6; border: 1px solid #D6D1E8; border-radius: 10px; }
    .ae-otp a, .ae-otp a:visited { display: inline-flex; align-items: center; min-height: 35px; padding: 0 16px; border-radius: 7px; font-size: 13.5px; font-weight: 600; color: #3A3170 !important; text-decoration: none; }
    .ae-otp a:hover, .ae-otp a:focus-visible { background: #E0DCEE; }
    .ae-otp .ae-otp-seg a:first-child { background: #4B3F8F; color: #ffffff !important; font-weight: 700; }
    .ae-h3 { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 8px 20px; margin: 1.4rem 0 0.5rem 0; scroll-margin-top: 8rem; }
    .ae-h3-t { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 22px; line-height: 1.2; font-weight: 600; color: #1C1B19; }
    .ae-h3-s { font-size: 14px; line-height: 1.55; color: #55524D; max-width: 780px; margin-top: 4px; }
    .ae-anchor { scroll-margin-top: 8rem; }
    .ae-pill { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px; font-size: 13px; font-weight: 600; white-space: nowrap; }
    .ae-tag { display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 999px; border: 1px solid #D9D5CC; background: #ffffff; color: #55524D; font-size: 13px; font-weight: 500; white-space: nowrap; }
    .ae-now { display: inline-flex; padding: 3px 10px; border-radius: 999px; background: #EFEDF6; color: #4B3F8F; font-size: 12.5px; font-weight: 600; white-space: nowrap; }
    .ae-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; }
    .ae-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 0.6rem 0 0.4rem 0; }
    .ae-tile { border: 1px solid #E2DED6; border-radius: 10px; padding: 16px 18px; display: flex; flex-direction: column; gap: 9px; background: #ffffff; }
    .ae-tile-l { font-size: 14px; font-weight: 500; color: #55524D; }
    .ae-tile-v { font-size: 34px; line-height: 1; font-weight: 700; color: #4B3F8F; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
    .ae-tile-c { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; font-size: 13px; color: #55524D; }
    .ae-tile-r { padding-top: 9px; border-top: 1px solid #E6E3DC; font-size: 12.5px; line-height: 1.45; color: #55524D; }
    .ae-tile-r b { color: #1C1B19; font-weight: 600; }
    .ae-card { border: 1px solid #E2DED6; border-radius: 10px; padding: 18px 20px; display: flex; flex-direction: column; gap: 12px; background: #ffffff; min-width: 0; }
    .ae-card-l { font-size: 12.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #55524D; }
    .ae-grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }
    .ae-tbl { overflow-x: auto; border: 1px solid #E2DED6; border-radius: 10px; margin: 0.3rem 0 0.6rem 0; }
    .ae-tbl table { width: 100%; border-collapse: collapse; font-size: 14px; color: #1C1B19; margin: 0; }
    .ae-tbl thead th { background: #F5F4F0; text-align: right; padding: 10px 14px; font-size: 13px; font-weight: 600; color: #55524D; border-bottom: 1px solid #D9D5CC; white-space: nowrap; }
    .ae-tbl thead th:first-child, .ae-tbl thead th.l { text-align: left; }
    .ae-tbl tbody td { padding: 9px 14px; border-bottom: 1px solid #E6E3DC; text-align: right; font-variant-numeric: tabular-nums; vertical-align: middle; }
    .ae-tbl tbody td:first-child, .ae-tbl tbody td.l { text-align: left; }
    .ae-tbl tbody tr:last-child td { border-bottom: none; }
    .ae-tbl tbody tr:hover td { background: #F5F4F0; }
    .ae-tbl tbody tr.ae-held td { background: #EFEDF6; }
    .ae-tbl .ae-k { color: #55524D; font-weight: 500; white-space: nowrap; }
    .ae-tbl tr.ae-bold td { font-weight: 700; }
    .ae-tbl tr.ae-bold td.ae-k { color: #1C1B19; font-weight: 600; }
    .ae-tbl td.ae-strat { color: #1C1B19; font-weight: 600; }
    .ae-tbl tr.ae-bold td.ae-strat { color: #4B3F8F; font-weight: 700; }
    .ae-tbl .ae-date { display: block; font-size: 11.5px; font-weight: 400; color: #6B6B7B; margin-top: 2px; }
    .ae-tbl .ae-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }
    .ae-tbl .ae-asset { font-weight: 700; color: #1C1B19; }
    .ae-tbl td:first-child { white-space: nowrap; }
    .ae-tbl td .ae-tag { font-size: 11.5px; padding: 1px 7px; margin-left: 6px; }
    .ae-tbl td.ae-nw, .ae-tbl th.ae-nw { white-space: nowrap; }
    .ae-tbl .ae-asset-n { color: #55524D; font-size: 12.5px; margin-left: 6px; }
    .ae-tbl.ae-tbl-fit thead th { white-space: normal; min-width: 4.5em; vertical-align: bottom; }
    .ae-tbl.ae-tbl-fit .ae-score-b { width: 64px; }
    .ae-tbl.ae-tbl-fit tbody td.l { white-space: nowrap; }
    .ae-tip { position: relative; cursor: help; display: inline-flex; align-items: center; }
    .ae-tip-i { display: inline-block; width: 15px; height: 15px; line-height: 15px; margin-left: 7px; border-radius: 50%; border: 1px solid #A99FCB; color: #4B3F8F; font-size: 10px; font-weight: 700; font-style: italic; text-align: center; }
    .ae-tip-box { visibility: hidden; opacity: 0; position: absolute; left: 0; top: 135%; z-index: 1000; width: 300px; padding: 10px 12px; background: #1C1B19; color: #ffffff; font-size: 12.5px; font-weight: 400; line-height: 1.45; text-align: left; white-space: normal; border-radius: 6px; box-shadow: 0 6px 18px rgba(0,0,0,0.18); transition: opacity 0.12s; }
    .ae-tip:hover .ae-tip-box { visibility: visible; opacity: 1; }
    .ae-notes { display: flex; flex-direction: column; border-top: 1px solid #E6E3DC; }
    .ae-note { display: grid; grid-template-columns: minmax(120px, 170px) minmax(0, 1fr); gap: 16px; padding: 11px 0; border-bottom: 1px solid #E6E3DC; }
    .ae-note-k { font-size: 13px; font-weight: 600; color: #1C1B19; }
    .ae-note-v { font-size: 14px; line-height: 1.55; color: #55524D; }
    .ae-hold { display: grid; grid-template-columns: minmax(140px, 210px) minmax(0, 1fr) 64px; gap: 14px; align-items: center; }
    .ae-hold-t { font-size: 15px; font-weight: 700; color: #1C1B19; }
    .ae-hold-n { display: block; font-size: 12.5px; color: #55524D; }
    .ae-hold-b { height: 12px; border-radius: 999px; background: #F5F4F0; overflow: hidden; }
    .ae-hold-b span { display: block; height: 100%; border-radius: 999px; background: #4B3F8F; }
    .ae-hold-w { text-align: right; font-size: 16px; font-weight: 700; color: #1C1B19; font-variant-numeric: tabular-nums; }
    .ae-chg { display: grid; grid-template-columns: 64px minmax(0, 1fr) auto; gap: 12px; align-items: center; padding: 9px 0; border-bottom: 1px solid #E6E3DC; }
    .ae-small { font-size: 12.5px; line-height: 1.5; color: #55524D; }
    .ae-score { display: inline-flex; align-items: center; gap: 10px; justify-content: flex-end; }
    .ae-score-b { position: relative; display: inline-block; width: 84px; height: 8px; background: #F5F4F0; border-radius: 999px; }
    .ae-score-b span { position: absolute; top: 0; bottom: 0; border-radius: 999px; }
    .ae-score-b i { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: #D9D5CC; }
    .ae-chip { display: inline-flex; gap: 6px; align-items: baseline; padding: 3px 9px; border: 1px solid #D9D5CC; border-radius: 6px; font-size: 12.5px; white-space: nowrap; }
    .ae-set { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
    .ae-set div { display: flex; flex-direction: column; gap: 3px; padding: 12px 14px; background: #F5F4F0; border-radius: 8px; }
    .ae-set-k { font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: #55524D; }
    .ae-set-v { font-size: 14px; line-height: 1.45; color: #1C1B19; overflow-wrap: anywhere; }
    .ae-set div.ae-wide { grid-column: 1 / -1; }
    .ae-chart-h { display: flex; flex-direction: column; gap: 4px; margin-bottom: 4px; scroll-margin-top: 8rem; }
    .ae-chart-t { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 18px; font-weight: 600; color: #1C1B19; }
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
    "font.sans-serif": ["Inter", "DM Sans", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
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
    # GLDM replaced GLD as the default gold fund November 3, 2026: the same
    # gold for 0.10% a year against GLD's 0.40%. GLD can still be run by
    # typing it into the custom tickers box.
    "GLDM":    "Gold (SPDR Gold MiniShares)",
    "BIL":     "Cash (1-3 Month T-Bill)",
    "BTAL":    "Defensive Equity Factor",
    "HGER":    "Commodity (All-Weather)",
    "IBIT":    "Bitcoin (Spot ETF)",
    # BTC-USD's checkbox was removed November 13, 2026. BTC-USD is still IBIT's
    # backfill source, and it can be typed into the custom tickers box like
    # any other ticker.
}

# Fixed assets that may start after the backtest start. Instead of blocking the
# run, they join the basket once they have more month-ends of history than the
# longest main lookback, the rule run_backtest already applies to every fixed
# asset. Bitcoin is the case: IBIT, backfilled with BTC-USD, has prices only
# from September 2014, so with the 2/8 lookback it joins in June 2015. Added
# November 3, 2026, when IBIT became a default fixed asset (the live strategy
# holds it).
JOIN_WHEN_AVAILABLE = {"IBIT", "BTC-USD"}

# Equity funds that may start after the backtest start. Emerging markets have
# no usable daily history before October 16, 1991 (EMXC_X, Templeton
# Developing Markets), so from an earlier start (1973-02 by default since
# November 25, 2026) the equity step picks from the other equity funds until
# EMXC has more month-ends than the longest lookback (mid-1992 at 2/8), which
# run_backtest already does for any equity fund. The
# data check lets one through only while enough other equity funds have
# history to fill the equity picks. Added November 18, 2026, with the history
# back to 1980 (Tyler's decision of that day).
JOIN_WHEN_AVAILABLE_EQUITY = {"EMXC"}

# An equity fund's place among the equity candidates before its own history is
# usable, as {fund: (stand-in, first holding month of the fund itself)}. Before
# that month the stand-in is the candidate and the fund is not; from that month
# the fund is and the stand-in is not. They are two separate candidates, never
# both in the pool, with no price splice between them, so each is ranked and
# sized on its own history only. The stand-in loads with the fund
# (download_data) from its local file, and applies only when that file is
# there and the fund is in the equity list.
#   EMXC before 1993: DEVXJ_X, developed markets outside the US and Japan, in
#   dollars, from "DEVXJ X Daily Prices.csv" (December 31, 1969 to December
#   31, 1992). From 1980, month-ends are the real monthly returns of every
#   market in Ken French's country files except Japan (14 markets in 1980, 18
#   by 1989), weighted by prior year-end market values from the IFC Factbook;
#   the days come from local indexes times the Fed's daily exchange rates (54
#   to 87% of the weight from March 1980), shifted each month to add up to the
#   real return; less EMXC's 0.25% a year. Before 1980: French's country files
#   from 1975 (MSCI for Canada in 1975 and 1976) and three MSCI regional
#   indexes (Europe, Pacific ex Japan, Canada) for 1970 to 1974, with daily
#   closes for only 28 to 58% of the weight; good enough for ranking, flagged
#   for sizing (Fourth Equity Seat Build 2026-11-24\Extension to 1970
#   2026-11-25.md). Emerging markets have no usable history
#   before October 16, 1991 and the real MSCI index for 1988 to 1991 could not
#   be had, so Tyler set the hand-off at the nearest year start at which
#   emerging markets are available: January 1993 (November 24, 2026). In the
#   backtest it is a tie with leaving the seat empty (1981 to 1992: Sortino
#   0.607 against 0.642, CAGR 11.02% against 11.10%), so it is evidence about
#   the rules, not about EMXC. A run that starts in 1993 or later is unchanged.
#   Record: C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System\
#   Backtest Engine\History Extension 2026-11-18\Fourth Equity Seat Build
#   2026-11-24\Fourth Equity Seat Build 2026-11-24.md
# params["equity_seat_before"] replaces this rule for one run; {} turns it off,
# which reproduces results from before November 24, 2026.
EQUITY_SEAT_BEFORE = {"EMXC": ("DEVXJ_X", "1993-01")}


def equity_seat_standins(params):
    """The stand-ins that apply to a run: {fund: (stand-in, first holding month
    of the fund)} for each fund in the run's equity list (EQUITY_SEAT_BEFORE,
    or params["equity_seat_before"] when given). A stand-in that is itself in
    the equity list is left alone."""
    rule = params.get("equity_seat_before")
    if rule is None:
        rule = EQUITY_SEAT_BEFORE
    eq = list(params.get("equity_tickers", []))
    return {p: (v[0], v[1]) for p, v in (rule or {}).items() if p in eq and v[0] not in eq}


def equity_pool_for(equity_tickers, as_of, standins, columns):
    """The equity candidates for the holding month after as_of (a rebalance
    month-end or a signal date): each fund, or its stand-in while the holding
    month is before the fund's own first one and the stand-in has prices."""
    if not standins:
        return list(equity_tickers)
    hold = pd.Timestamp(as_of).to_period("M") + 1
    pool = []
    for t in equity_tickers:
        s = standins.get(t)
        if s and s[0] in columns and hold < pd.Period(s[1], "M"):
            pool.append(s[0])
        else:
            pool.append(t)
    return pool

# The warm-up anchor: earliest_backtest_start counts the lookback and the
# volatility window forward from this month. At the standing settings the
# first reported month's risk window opens here (its first daily return is
# dated 1972-02-01, measured from the close of 1972-01-31) and its lookbacks
# open later. The unreported rebalance one month before the start reads
# earlier rows; it sets only the first month's starting weights. 1980-01-02
# until the switch to a 1973 start (November 25, 2026).
HISTORY_START = "1972-02-01"
# The earliest cleared month: a blank start never holds before it, whatever the
# lookback and volatility window allow (November 25, 2026, Tyler's decision).
FIRST_HOLD_FLOOR = "1973-02"


def earliest_backtest_start(main_lookbacks, vol_months):
    """The earliest backtest start, as "YYYY-MM", that the default basket's
    history allows for a main lookback and a volatility window, and never
    before FIRST_HOLD_FLOOR.

    Two things have to be in place by the first month held. The app's data
    check wants each fund's first price a full month before the longest
    lookback begins, which is the longest lookback plus two months after
    HISTORY_START's month. Position sizing wants a full volatility window,
    which is vol_months after it. The later of the two wins, and then the
    later of that and FIRST_HOLD_FLOOR: 1973-02 at 2/8 with a 12-month
    window or any shorter one, 1973-03 at 2/11, 1973-08 with an 18-month
    window. A blank start box means this month (added November 18, 2026).
    A custom fund with a shorter history still has to pass the data check.
    """
    first = pd.Period(HISTORY_START, "M")
    return str(max(pd.Period(FIRST_HOLD_FLOOR, "M"),
                   first + max(int(vol_months), int(max(main_lookbacks)) + 2)))

# Calendar anchor. The trading calendar is every weekday on which an
# exchange-listed ticker printed a close (market_trading_dates), and local
# price files do not vote. Before 1983 the default basket has no Yahoo ticker
# that should be trusted to mark the days, so the S&P 500 index is downloaded
# with every run and votes like any other ticker. It is never traded, ranked
# or shown. Added November 18, 2026. Checked November 25, 2026 on the cached
# prices: before 1980 it is the only ticker with prices, and through 1979 the
# calendar equals the days in the long price files.
CALENDAR_TICKER = "^GSPC"

# Expense-ratio adjustment applied to a backfilled (synthetic) segment, by
# primary ticker. A number applies to every segment of the chain; a
# {backfill ticker: fee} dict sets one per source; a negative fee is a credit.
# Needed where the backfill source is the underlying asset itself, so the
# synthetic era would otherwise carry no fund fees, or a fund whose fee
# differs from the primary's.
# The older segments added November 18, 2026 (TBILL_X, TLT_X, BTAL_X, QQQ_X,
# IWM_X, DXJ_X, EMXC_X, HGER_TR, GOLD_X, IEI_X, SPY_X, AOR_X, AOA_X) already
# carry their fund's fee in the file, so none is listed here and none is
# charged again. Two adjustments are written into the files themselves, as
# the 1981 research panel applied them: QQQ_X holds the Nasdaq-100 price
# index's +0.56% a year credit, and GOLD_X holds GLDM's 0.10% a year
# (Build Engine Segments.py in History Extension 2026-11-18\Engine
# Integration 2026-11-18). Since November 25, 2026 QQQ_X carries QQQ all the
# way to its March 10, 1999 launch: the +0.56% credit runs through February
# 11, 1994 and the Nasdaq-100 price index has no credit after it (QQQ 1970
# Extension 2026-11-25, in History Extension 2026-11-18).
# HGER_FUT, HGER's history before 1991 since November 25, 2026, carries
# HGER's 0.68% a year in its file the same way, so it is not listed either.
BACKFILL_FEE_ANNUAL = {
    "IBIT": 0.0025,   # iShares Bitcoin Trust, 0.25%/yr
    # GLDM (0.10%/yr) before its June 26, 2018 launch: GLD's prices already
    # carry GLD's 0.40%/yr, so they get a 0.30%/yr credit; GOLD (the daily
    # gold price in "GOLD Daily Prices.csv", a TradingView export from 1980;
    # not a fund, so it carries no fee of its own) before GLD's November 18,
    # 2004 launch is charged GLDM's 0.10%/yr. Added November 3, 2026. The
    # export had filed each close under the previous calendar day (two days
    # early after some US holidays from 2009), so its dates were moved one day
    # later on November 7, 2026.
    "GLDM": {"GLD": -0.0030, "GOLD": 0.0010},
    # QQQ before March 1999: RYOCX (Rydex OTC, a Nasdaq-100 index fund, from
    # February 14, 1994) gets a 1.26%/yr credit, its measured shortfall to QQQ
    # over 1999-2026; ^NDX (the Nasdaq-100 price index, no dividends) before
    # it gets 0.56%/yr, QQQ's measured lead over the price index. Added
    # November 3, 2026. Since November 25, 2026 QQQ's default chain is QQQ_X
    # alone, with its credit in the file, so these two rates apply only when
    # the old chain "RYOCX, ^NDX, QQQ_X" is typed into QQQ's row. Typed, it
    # rebuilds the history used before that date.
    "QQQ": {"RYOCX": -0.0126, "^NDX": -0.0056},
    # HGER's 0.68%/yr fee on its index history before its February 10, 2022
    # launch: QCI is the Quantix index's hypothetical back-test and QCI_SYNTH a
    # broad commodity index, neither with a fund fee. Added November 3, 2026.
    # Set per source since November 18, 2026: the segment before 1991
    # (HGER_FUT since November 25, 2026, HGER_TR before it) already carries
    # the 0.68%, so one rate for the whole chain would charge it twice. A
    # custom source typed into HGER's row is charged nothing unless it is
    # listed here.
    "HGER": {"QCI": 0.0068, "QCI_SYNTH": 0.0068},
}

# A backfill source's own prices from before this date are left out of the
# chain, by primary ticker and source, so the next source in the chain covers
# those years (PRIMARY_HISTORY_FROM does the same for a primary). Added
# November 18, 2026, with the history back to 1980:
#   NAESX. Yahoo's mutual fund prices from the 1980s miss distributions: NAESX
#   shows 2.6% a year for 1980 to 1989 against 18.2% for the small-company
#   series fitted to the Russell 2000. No Yahoo fund NAV from before 1990
#   reaches a chain; IWM_X carries IWM there.
#   ^NDX. The Nasdaq-100 itself from its first day (January 31, 1985) and a
#   scaled Nasdaq Composite before it are in QQQ_X, so ^NDX is used from 1990.
#   Since November 25, 2026 QQQ's default chain no longer names ^NDX (QQQ_X
#   runs to March 1999), so this date matters only for a typed "RYOCX, ^NDX,
#   QQQ_X".
#   GOLD (for GLDM only). GOLD_X carries the years before 1990: the same file
#   with its 1984-12-27 bad print repaired and missing exchange days carried
#   forward. Before the file begins on 1980-06-25: from 1974-12-31, spot gold
#   at the New York close worked out from COMEX gold futures (the held
#   contract's close less the carry the futures curve itself priced in;
#   repaired November 25, 2026, because the source's "spot" there was
#   futures closes with a false step at each contract switch); before that,
#   Investing.com's gold spot (the Frankfurt fixing in November and December
#   1969). GLD's own rule, for a GLD typed as a custom ticker, is unchanged.
# The dates are the 1981 research panel's splice dates (its FLOORS).
BACKFILL_SOURCE_FROM = {
    "IWM": {"NAESX": "1990-01-02"},
    "QQQ": {"^NDX": "1990-01-02"},
    "GLDM": {"GOLD": "1990-01-02"},
}

# A primary's own prices from before this date are dropped and replaced by its
# backfill chain, for a fund whose strategy changed or whose first months of
# trading don't reflect its holdings. DXJ was the unhedged WisdomTree Japan
# Total Dividend Fund until the close on March 31, 2010 and has been yen-hedged
# since (WisdomTree's SEC filing of January 2010), so its earlier prices come
# from DXJ_HEDGED. AOR and AOA, the 60/40 and 80/20 benchmarks, launched in
# November 2008 and traded too thinly to track their Vanguard stand-ins until
# about June 2009 (AOR fell 16.2% on December 1, 2008, VSMGX 6.1%), so VSMGX
# and VASGX carry them until then. Added November 3, 2026.
# BTAL traded a few hundred shares a day from its September 2011 launch through
# 2018, often none, so many of its Yahoo closes then are zero-volume marks or
# odd-lot trades: 9 month-end closes sit more than 1% from NAV (December 31,
# 2014 +2.93%), and even with the 2015-04-29 repair the closes make BTAL look
# 14.3% volatile a year against its NAV's 11.4% (2011-2018), which cut its
# inverse-volatility weight. From 2019 its close tracks NAV (no month-end off
# by more than 1%, daily correlation 0.96). So BTAL_NAV, its
# daily NAV from AGF with distributions reinvested ("BTAL NAV Daily Prices.csv"),
# carries it before 2019. Added November 4, 2026. Evidence: C:\Users\tyler\
# OneDrive\PROJECTS\Strategy Backtesting System\Basket Research\ETF Scouting\
# BTAL NAV Fix 2026-11-04\BTAL NAV Check Output.txt
PRIMARY_HISTORY_FROM = {
    "DXJ": "2010-04-01",
    "AOR": "2009-06-01",
    "AOA": "2009-06-01",
    "BTAL": "2019-01-02",
}

# Why each asset's history is built the way it is, shown under "Data
# adjustments" at the bottom of the results page for the assets in a run.
# They describe the default rules (BACKFILL_DEFAULTS, BACKFILL_FEE_ANNUAL,
# PRIMARY_HISTORY_FROM, KNOWN_BAD_PRINTS, BAL_BENCHMARKS); keep them in step
# with those and with the full record in "Data Adjustments.md" in the
# Backtest Engine project folder. Added November 4, 2026.
DATA_ADJUSTMENT_NOTES = {
    "HGER": (
        "Before its February 10, 2022 launch: the Quantix Commodity Index it tracks (a hypothetical "
        "back-test before the index's January 2022 launch), and before 2001 a broad commodity index built "
        "to track Bloomberg's (QCI_SYNTH), with HGER's 0.68% fee charged throughout. The Quantix back-test "
        "compounded far faster than broad commodities (about 10% a year in 2001-2025 against about 1.5%), "
        "so results that lean on HGER before 2022 flatter it; research rechecks them on the broad index alone. "
        "Before 1991: a daily commodity index built from individual futures contracts in up to 16 markets "
        "(10 to 15 in the 1970s) on fixed sector weights, with T-bill interest added and the 0.68% fee taken "
        "out (HGER_FUT, January 2, 1969 to January 2, 1991). It holds no energy before November 1979, so its "
        "1970s rows are farm goods and metals only. The earlier stand-in, T-bills plus 0.66 x "
        "the S&P GSCI Total Return's excess over T-bills (HGER_TR, from January 1980), can still be typed into "
        "HGER's Backfill Rules row."
    ),
    "BTAL": (
        "Its own closes from January 2, 2019. From its September 2011 launch through 2018 it traded a few "
        "hundred shares a day, often none, so its closes were stale marks that strayed from the fund's value "
        "and made it look more volatile than it was; its daily NAV (AGF's, distributions reinvested) is used "
        "instead. Before launch: T-bills + 0.873 x the daily US low-beta factor of "
        "Jensen, Kelly and Pedersen (Journal of Finance 2023; jkpfactors.com, CC BY-NC 4.0), less 1.4% a "
        "year for fund costs. Out of sample it tracks the real fund at a 0.92 monthly correlation, and BTAL's "
        "own index back-test (2001-2011) at 0.97, still the least certain stand-in in the basket. Until "
        "November 18, 2026 a formula on AQR's Betting Against Beta factor (BTAL_SYNTH) carried those years; "
        "it ran about 4 points a year richer over 1981-2011 and got 2008 backwards. The same formula reaches "
        "back to 1969, and over 1969 to 1979 it earned about what T-bills paid. Two years fall outside anything "
        "the real fund has done since 2011: it gained 30% in 1970, and in 1974 it only matched T-bills while US "
        "stocks as a whole fell 28%."
    ),
    "BIL": (
        "Before its May 30, 2007 launch: the 13-week T-bill's total return less BIL's fee (tracking error "
        "0.15% a year against BIL since 2007). It replaced a 1-3 year Treasury fund 5.5 times as volatile, "
        "which had shrunk the cash weight in defensive months. BIL is also the risk-free rate. Before 1990 "
        "the same build, back to January 1969. For 1969 to 1979 its bill rate matches the Federal Reserve's "
        "published rate on every day, and each year lands within 0.2 points of the published T-bill return."
    ),
    "TLT": (
        "Before its July 30, 2002 launch: 20 to 30-year Treasuries priced off the Federal Reserve's "
        "zero-coupon curve, less TLT's 0.15% fee (tracking error 1.5% a year against TLT since 2002). It "
        "replaced a 10-plus-year bond fund with about 15% less duration. Before January 31, 1990: a 30-year "
        "Treasury priced off Yahoo's 30-year yield, scaled to the same duration, less the fee (from February "
        "16, 1977; the Federal Reserve's curve has no 30-year point before November 1985). Before that it is "
        "priced mainly off the 20-year yield, a rougher stand-in cleared for months held from February 1973; "
        "its earlier rows only warm up the first readings."
    ),
    "GLDM": (
        "Before its June 26, 2018 launch: GLD, credited 0.30% a year for GLD's higher fee, and before GLD's "
        "November 18, 2004 launch, the daily gold price less GLDM's 0.10% fee. That price series had filed "
        "each close a day early; its dates were corrected on November 7, 2026, so backtests run before then "
        "that start before 2006 differ. Before 1990 the same gold price with a 1984 bad print repaired; "
        "from December 31, 1974 to June 24, 1980, spot gold at the New York close worked out from COMEX gold "
        "futures (each day's contract price less the carry the futures curve priced in); and before that, "
        "Investing.com's gold spot (from November 3, 1969). All less the 0.10% fee. Americans could not own "
        "gold bullion until December 31, 1974, so gold held before 1975 is on paper only."
    ),
    "GLD": (
        "Before its November 18, 2004 launch: the daily gold price, with no fund fee charged. That price "
        "series had filed each close a day early; its dates were corrected on November 7, 2026, so backtests "
        "run before then that start before 2006 differ."
    ),
    "QQQ": (
        "Before its March 10, 1999 launch: the Nasdaq-100 price index, with no credit from February 14, 1994 "
        "and plus 0.56% a year before that. Since November 25, 2026 the index carries 1994 to 1999 in place of "
        "Rydex's Nasdaq-100 index fund (RYOCX), for a cleaner daily record. Before January 31, 1985, when the "
        "Nasdaq-100 began: 1.35 x the Nasdaq Composite's daily move plus 0.56% a year, the least certain "
        "stretch of QQQ's history. The Nasdaq Composite starts on February 5, 1971, and the file follows its "
        "daily moves from February 8; the days before that are a fitted stand-in."
    ),
    "IWM": (
        "Before its May 26, 2000 launch: Vanguard Small-Cap Index (NAESX), which tracked the Russell 2000 "
        "until 2003 (tracking error 2.6% a year). It replaced an active small-cap fund that ran about 5.6 "
        "points a year ahead of IWM. Before 1990: Ken French's small-company portfolios fitted to IWM, less "
        "1.52% a year (Yahoo's NAESX prices from the 1980s miss distributions, so they are not used). The "
        "same build reaches back to January 1970. Small stocks traded thinly in those years, so their daily "
        "prices make them look calmer than their monthly prices do."
    ),
    "DXJ": (
        "DXJ was an unhedged Japan dividend fund until March 31, 2010, so its own prices start April 1, "
        "2010. Before that: its own and the MSCI Japan fund's (EWJ) returns converted to yen-hedged, with a "
        "hedged Nikkei 225 futures proxy before November 1996 (the same conversion tracks the real hedged "
        "MSCI Japan fund at a 0.97 correlation). Six misdated 2008 yen closes in Yahoo's data are repaired. "
        "Before February 1993: the Nikkei 225 with dividends, yen-hedged at the Japan and US short rates "
        "(from January 1969). No such fund existed then, and it compounds 29% a year over 1985 to 1989. "
        "Before 1980 the prices are Nikkei's own closes, but the hedge is only on paper: the yen was fixed at "
        "360 to the dollar until August 1971, and Japan limited money moving in and out all decade."
    ),
    "EMXC": (
        "Before its July 2017 launch: a synthetic MSCI Emerging Markets ex China, EEM less a China sleeve "
        "at China's published index weight, from October 2004 (after launch it picked the same equities as "
        "real EMXC in 103 of 109 months); EEM before that; and before April 2003 Vanguard's EM index fund "
        "(VEIEX), whose index held Hong Kong and Singapore and, until 2000-2001, no Korea, Taiwan, China or "
        "India, a known limit. Before May 4, 1994: Templeton Developing Markets (TEDMX), an actively managed "
        "fund, from October 16, 1991. Nothing earlier. Before 1993 its place among the equity candidates is "
        "taken by DEVXJ_X, developed markets outside the US and Japan: the real monthly returns of 14 to 18 "
        "markets, weighted by market value, with the days from local indexes covering 54 to 87% of the weight, "
        "less 0.25% a year (from January 1980). It reaches back to 1970 on rougher inputs: three MSCI regional "
        "indexes before 1975, and for most of the 1970s daily closes from only the UK and Germany, so most of "
        "each day's move then is estimated. Emerging markets take the seat from January 1993. In the "
        "backtest the stand-in is a tie with leaving the seat empty, so 1981 to 1992 says little about EMXC."
    ),
    "IBIT": (
        "Before its January 11, 2024 launch: Bitcoin's price (BTC-USD) less IBIT's 0.25% fee, on exchange "
        "trading days only. It joins the basket once it has more month-ends of history than the longest "
        "lookback (June 2015 with the 2/8 lookback). Bitcoin is in the basket with hindsight, since it survived "
        "and soared, so its early gains are partly in-sample."
    ),
    "AOR": (
        "The 60/40 benchmark line uses Vanguard's balanced fund VSMGX through May 2009, because AOR traded "
        "too thinly at first to track its holdings (it fell 16.2% on December 1, 2008, against VSMGX's 6.1%). "
        "Before September 30, 1994: 60% US stocks and 40% a 1 to 10-year Treasury ladder, less 0.15%, back to "
        "1969. Before August 16, 1971 the ladder holds 1 to 7-year bonds, since the Federal Reserve's curve has "
        "no longer yields then. It is "
        "US-only, and such mixes have run about 1.0 point a year ahead of the real global fund since 1994."
    ),
    "AOA": (
        "The 80/20 benchmark line uses Vanguard's VASGX through May 2009, because AOA traded too thinly at "
        "first to track its holdings. Before September 30, 1994: 80% US stocks and 20% a 1 to 10-year "
        "Treasury ladder, less 0.15%, back to 1969. Before August 16, 1971 the ladder holds 1 to 7-year bonds, "
        "since the Federal Reserve's curve has no longer yields then. It is US-only, and such mixes have run "
        "about 1.3 points a year ahead "
        "of the real global fund since 1994."
    ),
    "SPY": (
        "Before its January 29, 1993 first close: the S&P 500 with dividends, less SPY's 0.0945% fee (a "
        "0.998 monthly correlation with SPY since 1993). It reaches back to 1969, with eight misprinted Yahoo "
        "closes from 1970 to 1975 repaired, and each year from 1969 to 1979 lands within 0.3 points of the "
        "published S&P 500 total return."
    ),
    "MNK_SYNTH": (
        "A synthetic for the CME Micro Nikkei 225 future, not a fund: the full-size dollar Nikkei future's "
        "price moves (the Nikkei 225 index before 2004) plus modeled carry, calibrated to MNK's actual rolls "
        "in 2025-2026 (+52.35% against MNK's +52.07% over that year)."
    ),
    "IEI": (
        "Used by the G1 cash rule. Before its January 11, 2007 first close: a 3 to 7-year Treasury ladder "
        "priced off the Federal Reserve's curve, less IEI's 0.15% fee, from January 1969 (tracking error "
        "0.46% a year against IEI since 2007). Before August 16, 1971 its 7-year bond is less certain, though "
        "the ladder as a whole checks out. Until November 18, 2026 Vanguard Intermediate-Term Treasury "
        "(VFITX) carried those years, as in the November 7, 2026 test of the rule; type VFITX as the G1 "
        "backfill to reproduce it."
    ),
}

DEFAULT_EQUITY_ETFS = {
    "QQQ":  "US Large-Cap Growth",
    "IWM":  "US Small-Cap Blend",
    "DXJ":  "Japan Hedged Equity",
    "EMXC": "EM ex-China",
    # Synthetic total-return series for the CME Micro Dollar-Denominated
    # Nikkei 225 future, from the local "MNK SYNTH Daily Prices.csv".
    # Rebuild it with "Build Synthetic MNK.py" to extend the data.
    "MNK_SYNTH": "Japan Nikkei 225 Futures (MNK, synthetic)",
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
    # 2/7 added November 4, 2026 at Tyler's request: the best region in the
    # main lookback re-check. Listed after 2/8 so the selectbox default
    # (index=3, 2/8) is unchanged.
    "2/7-month blend (50/50)":    ([2, 7],    [0.50, 0.50]),
    "2/11-month blend (50/50)":   ([2, 11],   [0.50, 0.50]),
}

BENCHMARK_TICKER = "SPY"
# SPY's history before its first close (January 29, 1993): SPY_X, the S&P 500
# with dividends less SPY's 0.0945% fee. Applied on every run, like the
# balanced benchmarks' chains, unless a Backfill Rules row is typed for SPY.
# Added November 18, 2026.
BENCHMARK_BACKFILL = ["SPY_X"]
INITIAL_CAPITAL = 10_000

# Trading cost per side, charged on one-way turnover at every rebalance
# (adopted November 1, 2026, from the Signal Flip Tests). 0.0 reproduces the
# earlier gross results exactly.
DEFAULT_COST_PER_SIDE = 0.0005

# Position sizing: how the final holdings are weighted (added November 13,
# 2026, after Tyler decided on November 12, 2026 to switch to ERC).
#   "erc"          equal risk contribution, the sidebar's default. The holdings
#                  in ERC_KEEP (BIL) keep the weight inverse volatility gives
#                  them, and the other holdings split the rest so each carries
#                  the same share of that group's variance, from the covariance
#                  of their daily returns over the rows compute_risk_parity_weights
#                  reads (the vol_months calendar months to the signal date, on
#                  the days every holding has a price). A month with too short
#                  a sample (erc_min_returns: 126 returns at 12 months), a zero
#                  variance or no solver answer keeps the inverse-volatility
#                  weights. It is ERC12m in the ERC Weights Test:
#                  C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System\Basket Research\ETF Scouting\ERC Weights Test 2026-11-12\
#   "inverse_vol"  inverse volatility (simple risk parity), the method until
#                  then: compute_risk_parity_weights, unchanged.
# run_backtest and compute_current_signals read params["sizing"]. A caller that
# leaves it out (research scripts and tools written before the option) gets
# inverse volatility, so nothing outside the app changes method until it asks.
SIZING_OPTIONS = {
    "Equal risk contribution (ERC)": "erc",
    "Inverse volatility (simple risk parity)": "inverse_vol",
}
SIZING_LABELS = {v: k for k, v in SIZING_OPTIONS.items()}
SIZING_FILE_TAGS = {"erc": "ERC", "inverse_vol": "Inverse vol"}
SIZING_DEFAULT = "erc"            # the sidebar's starting choice
SIZING_IF_UNSET = "inverse_vol"   # for callers that pass no params["sizing"]
ERC_KEEP = ("BIL",)               # keep their inverse-volatility weight, outside the ERC split

# G1 cash rule (added November 7, 2026, off by default). In a cash month (BIL
# among the final holdings), move part of BIL's weight to an intermediate
# Treasury fund while Treasury yields are falling and the fund's main score is
# above BIL's. These defaults are the rule tested on November 7, 2026 as
# "G1 f0.5 ab" (Suggestive there, not adopted): half of BIL's weight to IEI,
# with VFITX before IEI's January 2007 launch, when the 5-year yield (^FVX) is
# below its level 63 trading days earlier. The record:
# C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System\Basket Research\ETF Scouting\Unconventional Ideas Test 2026-11-07\
# The sidebar's G1 Cash Rule section turns it on and changes any setting.
# Since November 18, 2026 (Tyler's decision) IEI_X carries IEI before its
# January 11, 2007 first close, in VFITX's place: a 3 to 7-year Treasury
# ladder off the Federal Reserve curve, less IEI's 0.15%, from 1969. Type
# VFITX into the G1 section's backfill box to reproduce the November 7 test.
G1_DEFAULTS = {
    "fund": "IEI",
    "backfill": ["IEI_X"],
    "fraction": 0.5,
    "yield_gate": True,
    "yield_ticker": "^FVX",
    "yield_lookback": 63,
    "trend_gate": True,
}
G1_YIELD_OPTIONS = {
    "^FVX": "5-year Treasury yield (^FVX)",
    "^TNX": "10-year Treasury yield (^TNX)",
    "^IRX": "13-week T-bill yield (^IRX)",
}

# ---- Brand colors (mirrors the Asymmetric Edge newsletter palette) ----
# Series colors are consistent across every chart in the dashboard:
#   Strategy = brand purple, SPY = muted indigo, 60/40 = brand teal.
# Categorical colors reuse teal (positive / risk-on) and purple (negative /
# risk-off), matching the asset class chart convention in app.py.
BRAND_PURPLE = "#4B3F8F"        # Strategy / primary: muted indigo since November 20, 2026 (app.py keeps #3A0CA3)
BRAND_PURPLE_FADE = "#9189BC"   # Lightened indigo for secondary lines (e.g. 24mo Sortino)
BRAND_INDIGO = "#9CA3AF"        # SPY / benchmark (neutral gray, matches app.py S&P 500)
BRAND_BLUE = "#4361EE"          # 80/20 benchmark (matches app.py)
BRAND_CYAN = "#4CC9F0"          # 60/40 benchmark (matches app.py)
BRAND_TEAL = "#30C7B5"          # Positive regime / excess accent (brand teal)
BRAND_STEEL = "#6A9BAD"         # Legacy steel blue-green (retained for reference)
BRAND_RED = "#C0392B"           # Loss / drawdown / risk-off semantic (matches app.py)
BRAND_GRAY = "#A8A8A8"          # Neutral / reference lines
BRAND_TEXT = "#484848"          # Charcoal text
BRAND_HIGHLIGHT_BG = "#EFEDF6"  # Light purple tint for selected table rows

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
#              Sept 1994) onto them to reach the mid-1990s. Before the
#              Vanguard funds begin (September 30, 1994), AOR_X and AOA_X:
#              60/40 and 80/20 mixes of US stocks and a 1 to 10-year
#              Treasury ladder, rebalanced daily, less 0.15% (added
#              November 18, 2026). They are US-only, where the real funds
#              hold international stocks.
# Set to [] to disable balanced benchmarks entirely.
BAL_BENCHMARKS = [
    {"label": "60/40", "ticker": "AOR", "color": BRAND_CYAN, "linestyle": "--", "backfill": ["VSMGX", "AOR_X"]},
    {"label": "80/20", "ticker": "AOA", "color": BRAND_BLUE, "linestyle": "-.", "backfill": ["VASGX", "AOA_X"]},
]

_base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
RESULTS_FILE = os.path.join(_base_dir, "Backtest Results Log.xlsx")
CACHE_DIR = os.path.join(_base_dir, "price_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "daily_prices.csv")
# How far back each ticker was downloaded (see load_cache_meta).
CACHE_META_FILE = os.path.join(CACHE_DIR, "cache_meta.json")
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


def load_cache_meta():
    """How far back each cached ticker was downloaded: {ticker: "YYYY-MM-DD"}.

    Read from price_cache/cache_meta.json, which download_data writes after a
    full download (added November 7, 2026). A fund that launched after that
    date has no earlier history to find, so its series counts as complete
    from that date. Empty if the file is missing or unreadable.
    """
    try:
        with open(CACHE_META_FILE, "r", encoding="utf-8") as fh:
            got = json.load(fh).get("downloaded_from", {})
        return {str(k): str(v) for k, v in got.items()} if isinstance(got, dict) else {}
    except Exception:
        return {}


def save_cache_meta(downloaded_from):
    """Write the download record that load_cache_meta reads."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_META_FILE, "w", encoding="utf-8") as fh:
        json.dump({"downloaded_from": dict(sorted(downloaded_from.items()))}, fh, indent=1)


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

    Each date must be the trading day its close belongs to. An export stamped
    with each session's opening evening (Sunday to Thursday) is a day early,
    as GOLD Daily Prices.csv was until November 7, 2026. WTI Daily Prices.csv,
    which no default uses, still is; GOLD's known limits after 2004 (holiday
    rows, an open last bar) are listed in "Data Adjustments.md".
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


# Known bad prints in the Yahoo price history, as (ticker, date) pairs. Each
# close is replaced at load time by the geometric midpoint of the closes on the
# trading days before and after it. Month-end prices are unaffected by prints
# like these, but the risk parity weights use 12 months of daily returns, so
# one bad day distorts a year of weights. The disk cache keeps Yahoo's raw
# value and the repair runs on every load, so a cache refresh cannot bring a
# print back.
#   BTAL 2015-04-29: Yahoo shows 11.2412 between 16.7317 on April 28 and
#   16.7317 on April 30, a -32.8% day reversed +48.8% the next. Found October
#   31, 2026. It distorted BTAL's weight in 8 hold months, August 2015 to
#   April 2016. Evidence: C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting
#   System\Basket Research\ETF Scouting\BTAL Glitch Check Output.txt
#   Since November 4, 2026, BTAL's own closes before 2019 are replaced by its NAV
#   (PRIMARY_HISTORY_FROM), so this repair matters only to a BTAL run without
#   the default backfill rule.
#   DXJ_HEDGED 2008-01-08, 2008-04-08, 2008-05-08, 2008-07-08, 2008-10-08
#   and 2008-12-08: Yahoo's JPY=X, which DXJ_HEDGED is hedged with before
#   April 2010, files August 2008 closes under the 8th of these months (day
#   and month swapped; the close dated December 8 is August 12's 109.24 yen,
#   against 92.82 and 92.30 on either side). Each shows as a one-day spike
#   reversed the next day, the largest +24.5% then -16.5% on December 8 and
#   9. Found November 3, 2026. They overstated DXJ's 2008 daily volatility
#   (58% a year against 47% repaired); effect on the standing configuration
#   at most 0.003 Sortino. Evidence: C:\Users\tyler\OneDrive\PROJECTS\Strategy
#   Backtesting System\Basket Research\ETF Scouting\DXJ Slot Deep Test
#   2026-11-03\Yen Print Recheck Output.txt
KNOWN_BAD_PRINTS = [
    ("BTAL", "2015-04-29"),
    ("DXJ_HEDGED", "2008-01-08"),
    ("DXJ_HEDGED", "2008-04-08"),
    ("DXJ_HEDGED", "2008-05-08"),
    ("DXJ_HEDGED", "2008-07-08"),
    ("DXJ_HEDGED", "2008-10-08"),
    ("DXJ_HEDGED", "2008-12-08"),
]

# Distributions missing from Yahoo's adjusted closes, as (ticker, ex-date,
# factor, trigger). Yahoo lowers a fund's whole earlier history at each
# distribution it has on file, so one it lacks leaves a false one-day loss on
# the ex-date. The repair multiplies every close before the ex-date by the
# factor: the fund's ex-date NAV over that NAV plus the distribution, which is
# reinvestment at the ex-date NAV. It runs only while the ex-date's own return
# is still below the trigger, so it applies once per load and stops by itself
# if Yahoo ever adds the distribution. The disk cache keeps Yahoo's values.
#   VFITX 1993-12-31: Vanguard's May 20, 1994 prospectus (SEC EDGAR, accession
#   0000912057-94-001821) shows $0.413 a share of realized capital gains for
#   the fiscal year ended January 31, 1994; Yahoo has none on file. Its closes
#   show -3.16% that day, when the 5-year Treasury yield barely moved, and
#   +7.31% for calendar 1993 against Vanguard's published +11.43%. Factor
#   10.71 / (10.71 + 0.413) = 0.96287; repaired, 1993 reads +11.45%. Found
#   November 18, 2026. VFITX carried IEI before January 2007 for the G1 cash
#   rule only, and the print sat before any start the default basket then
#   allowed (1995-02), so no run changed. Evidence: C:\Users\tyler\OneDrive\PROJECTS\
#   Strategy Backtesting System\Backtest Engine\History Extension 2026-11-18\
#   VFITX Print Check\
KNOWN_MISSING_DISTRIBUTIONS = [
    ("VFITX", "1993-12-31", 0.96287, -0.02),
]


def apply_known_price_fixes(prices):
    """Repair the KNOWN_BAD_PRINTS and the KNOWN_MISSING_DISTRIBUTIONS in a
    daily price frame, returning a copy if anything changes. A print is
    skipped if its ticker or date is missing or it has no trading day on
    either side; a distribution is skipped if its ticker or date is missing,
    or the date's return is not below its trigger."""
    if prices is None or prices.empty:
        return prices
    out = prices
    for ticker, date_str in KNOWN_BAD_PRINTS:
        if ticker not in out.columns:
            continue
        series = out[ticker].dropna()
        day = pd.Timestamp(date_str)
        if day not in series.index:
            continue
        i = series.index.get_loc(day)
        if i == 0 or i >= len(series) - 1:
            continue
        if out is prices:
            out = prices.copy()
        out.loc[day, ticker] = float(np.sqrt(series.iloc[i - 1] * series.iloc[i + 1]))
    for ticker, date_str, factor, trigger in KNOWN_MISSING_DISTRIBUTIONS:
        if ticker not in out.columns:
            continue
        series = out[ticker].dropna()
        day = pd.Timestamp(date_str)
        if day not in series.index:
            continue
        i = series.index.get_loc(day)
        if i == 0 or series.iloc[i] / series.iloc[i - 1] - 1.0 >= trigger:
            continue
        if out is prices:
            out = prices.copy()
        early = out.index < day
        out.loc[early, ticker] = out.loc[early, ticker] * factor
    return out


class MissingPriceDataError(RuntimeError):
    """A requested ticker has no prices from Yahoo, the price cache or a local
    index file. Raised so a run stops instead of dropping the ticker silently."""


# download_data's update step (November 7, 2026). The recent download starts
# UPDATE_OVERLAP_DAYS calendar days before each ticker's last cached date.
# Cached days up to PROVISIONAL_DAYS before that date count as settled: over
# them, the recent prices' ratio to the cached ones must agree to within
# RESCALE_TOLERANCE, and their median rescales the cached history when it
# differs from 1 by more than that. A dividend or split moves the ratio far
# more (BIL's monthly dividend, about 3e-3); Yahoo's rounding, a few parts in
# a million. The last PROVISIONAL_DAYS take the recent prices, since a close
# saved during a session, or one Yahoo left blank, may since have changed.
UPDATE_OVERLAP_DAYS = 30
PROVISIONAL_DAYS = 5
RESCALE_TOLERANCE = 1e-5


def download_data(tickers_tuple, start_date_str, force_refresh=False):
    """
    Download adjusted close prices for a list of tickers, using a local
    disk cache to avoid redundant downloads.

    On the first run, all data is fetched from Yahoo Finance and saved
    to price_cache/daily_prices.csv. On later runs, only missing tickers
    and recent dates are downloaded. price_cache/cache_meta.json records
    how far back each ticker was downloaded, so a fund that launched after
    the download start still counts as complete, and the cached history is
    lined up with the recent download so a dividend paid in between isn't
    lost (both November 7, 2026; see the update step). Set
    force_refresh=True to ignore the cache and re-download everything.

    Tickers that match a local index CSV file (e.g., QCI) are loaded
    from disk and never sent to Yahoo Finance.

    If a full download comes back empty for a ticker the cache holds (a
    Yahoo outage), the cached series is kept and a warning is shown. If any
    requested ticker still has no prices after the merge, raises
    MissingPriceDataError before anything is written to the cache.
    """
    unique_tickers = sorted(set(tickers_tuple))
    if not unique_tickers:
        return pd.DataFrame()

    # Separate local index tickers from Yahoo tickers
    local_index_data = load_local_index_data()
    # A fund with a stand-in for its equity seat (EQUITY_SEAT_BEFORE) loads the
    # stand-in's local series with it, so every run that has the fund has both.
    for _seat_fund, (_seat_standin, _seat_until) in EQUITY_SEAT_BEFORE.items():
        if (_seat_fund in unique_tickers and _seat_standin in local_index_data.columns
                and _seat_standin not in unique_tickers):
            unique_tickers.append(_seat_standin)
    local_tickers = [t for t in unique_tickers if t in local_index_data.columns]
    yahoo_tickers = [t for t in unique_tickers if t not in local_index_data.columns]

    # --- Yahoo Finance portion (only for non-local tickers) ---
    cached = pd.DataFrame() if force_refresh else load_price_cache()
    start_date = pd.Timestamp(start_date_str)
    downloaded_from = {} if force_refresh else load_cache_meta()

    needs_full_download = []
    needs_update = []

    for t in yahoo_tickers:
        if cached.empty or t not in cached.columns:
            needs_full_download.append(t)
        else:
            series = cached[t].dropna()
            if series.empty:
                needs_full_download.append(t)
            else:
                # A fund that launched after the download start never has
                # history back to it, so its first date alone marked it
                # incomplete on every run: until November 7, 2026, 20 of the
                # default run's tickers were re-downloaded in full each time.
                # The series is complete from the earliest date it was
                # downloaded from, when that is earlier.
                covered_from = series.index.min()
                if t in downloaded_from:
                    try:
                        covered_from = min(covered_from, pd.Timestamp(downloaded_from[t]))
                    except (ValueError, TypeError):
                        pass
                if covered_from > start_date + pd.Timedelta(days=7):
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

    # The update step (rewritten November 7, 2026). Yahoo's adjusted closes are
    # back-adjusted: each dividend or split lowers a fund's whole earlier
    # history by one factor. Until November 7, 2026 new days were added to the
    # cache without rescaling, so a dividend paid between two downloads was
    # lost at the join. Now each cached series is compared with the recent
    # download over its settled days (see UPDATE_OVERLAP_DAYS). If the ratio
    # agrees there, the cached history is rescaled by it, and the recent
    # prices replace the cached ones over the provisional days and add the new
    # ones (a blank recent close keeps the cached value, as on 2026-09-29, when
    # Yahoo left a close blank for hours). If the two disagree, or share no
    # settled day, the ticker takes a full download instead.
    updated = {}
    if needs_update:
        last_cached = {t: cached[t].last_valid_index() for t in needs_update}
        update_from = (min(last_cached.values())
                       - pd.Timedelta(days=UPDATE_OVERLAP_DAYS)).strftime("%Y-%m-%d")
        delta_dl = pd.DataFrame()
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
        no_recent, redo = [], []
        for t in needs_update:
            old = cached[t].dropna()
            new = delta_dl[t].dropna() if t in delta_dl.columns else pd.Series(dtype=float)
            if new.empty:
                updated[t] = old
                no_recent.append(t)
                continue
            settled_to = last_cached[t] - pd.Timedelta(days=PROVISIONAL_DAYS)
            window = old.index[(old.index >= last_cached[t] - pd.Timedelta(days=UPDATE_OVERLAP_DAYS))
                               & (old.index <= settled_to)]
            shared = window.intersection(new.index)
            ratio = new.loc[shared] / old.loc[shared]
            if len(shared) == 0 or ratio.max() / ratio.min() - 1.0 > RESCALE_TOLERANCE:
                redo.append(t)
                continue
            factor = float(ratio.median())
            if abs(factor - 1.0) > RESCALE_TOLERANCE:
                old = old * factor
            updated[t] = new[new.index > settled_to].combine_first(old)
        if redo:
            print(f"download_data: full download for {', '.join(redo)}, whose recent prices "
                  f"don't line up with the cached ones", file=sys.stderr, flush=True)
            needs_full_download.extend(redo)
        if no_recent:
            # During a Yahoo outage the recent download can come back empty
            # (as full downloads did on November 1, 2026): keep the cache and say so.
            last_dates = sorted({last_cached[t].strftime("%Y-%m-%d") for t in no_recent})
            stale_msg = (
                f"Yahoo Finance returned no recent prices for {', '.join(no_recent)}, so the "
                f"cached series {'is' if len(no_recent) == 1 else 'are'} used as is "
                f"(last cached date {', '.join(last_dates)})."
            )
            st.warning(stale_msg)
            print("WARNING: " + stale_msg, file=sys.stderr, flush=True)

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

    # Cache fallback (November 1, 2026). During a Yahoo outage a full
    # re-download can come back empty, or with blank columns, for tickers the
    # cache already holds. 13 tickers did on November 1, and with no fallback
    # BTAL silently became the raw BAB factor. Keep the cached series instead.
    fell_back = []
    if needs_full_download and not cached.empty:
        for t in needs_full_download:
            got = full_dl[t].dropna() if t in full_dl.columns else pd.Series(dtype=float)
            have = cached[t].dropna() if t in cached.columns else pd.Series(dtype=float)
            if got.empty and not have.empty:
                fell_back.append(t)
        if fell_back:
            kept = cached[fell_back]
            if full_dl.empty:
                full_dl = kept.copy()
            else:
                full_dl = full_dl.drop(columns=fell_back, errors="ignore").join(kept, how="outer")
            last_dates = sorted({cached[t].last_valid_index().strftime("%Y-%m-%d") for t in fell_back})
            fallback_msg = (
                f"Yahoo Finance returned no prices for {', '.join(fell_back)}, so the "
                f"cached series {'is' if len(fell_back) == 1 else 'are'} used instead "
                f"(last cached date {', '.join(last_dates)})."
            )
            st.warning(fallback_msg)
            print("WARNING: " + fallback_msg, file=sys.stderr, flush=True)

    result = pd.DataFrame()

    if updated:
        idx = cached.index
        for s in updated.values():
            idx = idx.union(s.index)
        result = pd.DataFrame({t: s.reindex(idx) for t, s in updated.items()}, index=idx)

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

    # Every requested ticker needs prices by now, from Yahoo, the cache or a
    # local index file. Stop rather than let an empty column drop out of the
    # backtest silently. Nothing has been written to the cache yet.
    missing = [t for t in unique_tickers
               if t not in result.columns or result[t].dropna().empty]
    if missing:
        progress.empty()
        raise MissingPriceDataError(
            f"No price data for {', '.join(missing)} from Yahoo Finance, the price "
            f"cache or a local index file. Check the ticker spelling, or try again "
            f"later if Yahoo is having an outage. The price cache was not changed."
        )

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
        # Record how far back each full download reached (see load_cache_meta).
        for t in needs_full_download:
            if t in fell_back or t not in full_dl.columns or full_dl[t].dropna().empty:
                continue
            had = not cached.empty and t in cached.columns and not cached[t].dropna().empty
            reach = start_date
            if had and t in downloaded_from:
                try:
                    reach = min(reach, pd.Timestamp(downloaded_from[t]))
                except (ValueError, TypeError):
                    pass
            downloaded_from[t] = reach.strftime("%Y-%m-%d")
        save_cache_meta(downloaded_from)

    progress.caption(f"Cache: {len(unique_tickers)} tickers loaded ({len(local_tickers)} from local files).")

    available = [t for t in unique_tickers if t in result.columns]
    return apply_known_price_fixes(result[available])


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

    Local index CSVs ('<TICKER> Daily Prices.csv') are excluded for the same
    reason: they are not exchange-listed, and their dates need not be US
    exchange days, so they would vote non-trading days into the calendar.
    GOLD has closes on US exchange holidays, BAB has Saturday rows until 1952
    (the NYSE's Saturday sessions), and WTI is still stamped a day early,
    Sunday to Thursday, as GOLD was until November 7, 2026.

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
                            fee_annual=0.0, primary_from=None, source_from=None):
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
    fee_annual can also be a {backfill ticker: fee} dict, one fee per source,
    and a negative fee is a credit, for a source whose own fees run higher
    than the primary's -- e.g. GLD (0.40%) standing in for GLDM (0.10%).

    primary_from drops the primary's own prices before that date, so the
    backfill chain covers them too, for a fund whose strategy changed (see
    PRIMARY_HISTORY_FROM).

    source_from, a {backfill ticker: date} dict, drops a source's own prices
    before that date, so the next source in the chain covers the earlier
    years (see BACKFILL_SOURCE_FROM; added November 18, 2026). Left out,
    every source is used from its first price, as before.

    The synthetic segment is also masked to the exchange calendar whenever
    the primary is not itself a 24/7 asset. A 24/7 backfill source such as
    BTC-USD would otherwise hand the ETF weekend closes it could never have
    printed, and would put weekend dates back into the trading calendar.

    A primary with no prices of its own is built from its first available
    source on the same terms: masked to the calendar, and charged the fee
    anchored at that source's last date. IBIT's rule runs this way, from
    BTC-USD, whenever IBIT isn't ticked, because the UI downloads every
    chain ticker. Before November 7, 2026 this case skipped the mask and the
    fee, so BTC-USD's weekday US holidays entered no-IBIT runs' calendar.

    Returns the spliced series and a list of segment descriptions.
    """
    is_crypto_primary = str(primary_ticker).upper().endswith("-USD")
    calendar = None if is_crypto_primary else market_trading_dates(daily_prices)
    if primary_ticker in daily_prices.columns:
        primary = daily_prices[primary_ticker].copy()
    else:
        primary = pd.Series(dtype=float, index=daily_prices.index, name=primary_ticker)
    if primary_from is not None:
        primary = primary.where(primary.index >= pd.Timestamp(primary_from))

    primary_valid = primary.dropna()
    segments = []

    if not primary_valid.empty:
        segments.append((primary_ticker, primary_valid.index.min(), primary_valid.index.max()))

    combined = primary.copy()

    for bf_ticker in backfill_tickers:
        if bf_ticker not in daily_prices.columns:
            continue

        bf_series = daily_prices[bf_ticker].dropna()
        if source_from and bf_ticker in source_from:
            bf_series = bf_series[bf_series.index >= pd.Timestamp(source_from[bf_ticker])]
        if bf_series.empty:
            continue

        combined_valid = combined.dropna()
        if combined_valid.empty:
            # No primary data at all: this backfill is the whole series, with
            # a spliced segment's calendar mask and fee (see the docstring).
            if calendar is not None:
                bf_series = bf_series[bf_series.index.isin(calendar)]
                if bf_series.empty:
                    continue
            fee = fee_annual.get(bf_ticker, 0.0) if isinstance(fee_annual, dict) else fee_annual
            if fee:
                yrs = (bf_series.index.max() - bf_series.index).days / 365.25
                bf_series = bf_series / ((1.0 - fee) ** yrs)
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
        # the synthetic series start HIGHER and grow more slowly (and a
        # credit, a negative fee, the reverse).
        fee = fee_annual.get(bf_ticker, 0.0) if isinstance(fee_annual, dict) else fee_annual
        if fee:
            yrs = (earliest_combined - bf_scaled.index).days / 365.25
            bf_scaled = bf_scaled / ((1.0 - fee) ** yrs)

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


def solve_erc(S, tol=1e-11, itmax=200):
    """
    Equal risk contribution: the weights x > 0, sum(x) = 1, with x_i (Sx)_i
    the same for every fund (S is the covariance matrix). Solved as Spinu's
    convex problem, minimize y'Sy / 2 - sum(ln y_i) / n, by a damped Newton
    iteration; x = y / sum(y). None if it does not converge. Copied line for
    line from erc() in MV252 Harness.py (the Minimum Variance 252-Day Test of
    November 12, 2026, in ETF Scouting), the solver the ERC Weights Test used.
    """
    n = S.shape[0]
    if n == 1:
        return np.array([1.0])
    b = np.full(n, 1.0 / n)
    y = 1.0 / np.sqrt(np.diag(S) * n)              # the exact answer when every correlation is 0

    def f(z):
        return 0.5 * float(z @ S @ z) - float(b @ np.log(z))

    for _ in range(itmax):
        g = S @ y - b / y
        if np.max(np.abs(g * y)) < tol:           # y_i (Sy)_i = 1 / n for every fund
            x = y / y.sum()
            return x if np.all(np.isfinite(x)) and np.all(x > 0) else None
        try:
            d = np.linalg.solve(S + np.diag(b / y ** 2), -g)
        except np.linalg.LinAlgError:
            return None
        f0, slope, t, moved = f(y), float(g @ d), 1.0, False
        g0 = float(np.max(np.abs(g * y)))
        while t > 1e-14:
            yn = y + t * d
            # accept on the usual sufficient decrease, or, once the objective's
            # change is lost in rounding, on a smaller residual
            if np.all(yn > 0) and (f(yn) <= f0 + 1e-4 * t * slope
                                   or float(np.max(np.abs((S @ yn - b / yn) * yn))) < g0):
                y, moved = yn, True
                break
            t *= 0.5
        if not moved:
            return None
    return None


def erc_min_returns(vol_months):
    """The fewest daily returns an ERC sample may have: half of a full window
    (252 returns a year), and at least 20. 126 at the standing 12 months, as
    in the ERC Weights Test."""
    return max(20, int(np.ceil(252 * vol_months / 12 * 0.5)))


def compute_position_weights(daily_prices, selected_tickers, as_of_date, vol_months=12,
                             sizing=SIZING_IF_UNSET):
    """
    The final holdings' weights under the chosen position sizing (see
    SIZING_OPTIONS). Returns (weights, note): weights is a dict in holding
    order, and note says what set them, for the trade log and the signals:
    "Inverse vol", "ERC", or "Inverse vol (ERC fallback: <reason>)".

    "inverse_vol": compute_risk_parity_weights, unchanged.
    "erc": the holdings in ERC_KEEP (BIL) keep the weight
    compute_risk_parity_weights gives them, and the rest of the portfolio is
    split among the other holdings by equal risk contribution (solve_erc) on
    the sample covariance (ddof = 1) of their daily returns over the rows
    compute_risk_parity_weights reads: vol_months calendar months to
    as_of_date, on the days every holding, BIL included, has a price. The
    month keeps the inverse-volatility weights when the sample has fewer than
    erc_min_returns(vol_months) returns ("short sample"), when a variance is
    zero or the covariance is not finite ("zero variance"), or when the
    solver finds no answer ("no convergence"). The same steps as
    new_weights_x in MV252 Harness.py with the ERC Weights Test's ERC12m
    setting.
    """
    iv = compute_risk_parity_weights(daily_prices, selected_tickers, as_of_date, vol_months)
    if sizing == "inverse_vol":
        return iv, "Inverse vol"
    if sizing != "erc":
        raise ValueError(f"unknown position sizing {sizing!r}")
    keep = [t for t in selected_tickers if t in ERC_KEEP]
    funds = [t for t in selected_tickers if t not in keep]
    sleeve = 1.0 - sum(float(iv[t]) for t in keep)
    if not funds or sleeve <= 0:
        return iv, "ERC"
    if len(funds) == 1:
        x = np.array([1.0])
    else:
        lookback_start = as_of_date - pd.DateOffset(months=vol_months)
        subset = daily_prices[selected_tickers].loc[lookback_start:as_of_date].dropna()
        px = subset[funds].to_numpy(dtype=float)
        r = px[1:] / px[:-1] - 1.0
        if len(r) < erc_min_returns(vol_months):
            return iv, "Inverse vol (ERC fallback: short sample)"
        S = np.atleast_2d(np.cov(r.T, ddof=1))
        var = np.diag(S)
        if not np.all(np.isfinite(S)) or np.any(var <= 0):
            return iv, "Inverse vol (ERC fallback: zero variance)"
        x = solve_erc(S)
        if x is None:
            return iv, "Inverse vol (ERC fallback: no convergence)"
    weights = {}
    for t in selected_tickers:                    # keep the holding order
        weights[t] = float(iv[t]) if t in keep else sleeve * float(x[funds.index(t)])
    return weights, "ERC"


def describe_sizing(params, trade_df=None):
    """
    Plain-language descriptions of a run's position sizing, for the results
    page and the results log. Returns a dict with:
      "label"     the method's name, as in the sidebar
      "file_tag"  a short name for download file names
      "final"     the method in the settings summary's "Final" line
      "fallback"  for ERC with a trade log, the months ERC fell back to
                  inverse volatility, e.g. "0 of 377 months" or "2 of 377
                  months (short sample: 1998-03, 1998-04)"; "not an ERC run"
                  otherwise
      "line"      a sentence for the Performance Summary
      "note"      a sentence for the method note at the bottom of the page
    """
    sizing = params.get("sizing", SIZING_IF_UNSET)
    vm = int(params.get("vol_months", 12))
    info = {"label": SIZING_LABELS[sizing], "file_tag": SIZING_FILE_TAGS[sizing], "fallback": "not an ERC run"}
    if sizing == "inverse_vol":
        info["final"] = f"inverse volatility ({vm}mo vol)"
        info["line"] = f"Position sizing: inverse volatility (simple risk parity) on the {vm}-month window."
        info["note"] = (
            f"Position sizing: inverse volatility (simple risk parity), each holding weighted in proportion "
            f"to 1 over the volatility of its daily returns over the {vm} months to the signal date."
        )
        return info
    if trade_df is not None and "sizing" in trade_df.columns:
        prefix = "Inverse vol (ERC fallback: "
        by_reason = {}
        for hm, s in zip(trade_df["hold_month"], trade_df["sizing"].astype(str)):
            if s.startswith(prefix):
                by_reason.setdefault(s[len(prefix):-1], []).append(hm)
        n_fb = sum(len(v) for v in by_reason.values())
        text = f"{n_fb} of {len(trade_df)} months"
        if n_fb:
            text += " (" + "; ".join(
                f"{why}: " + (", ".join(ms) if len(ms) <= 6 else f"{len(ms)} months from {ms[0]} to {ms[-1]}")
                for why, ms in by_reason.items()) + ")"
        info["fallback"] = text
    info["final"] = f"equal risk contribution ({vm}mo window, BIL at inverse volatility)"
    info["line"] = (
        f"Position sizing: equal risk contribution (ERC) on the {vm}-month window, with BIL at its "
        f"inverse-volatility weight. ERC fell back to inverse volatility in {info['fallback']}."
    )
    info["note"] = (
        f"Position sizing: equal risk contribution (ERC). BIL keeps its inverse-volatility weight, and the other "
        f"holdings split the rest so each carries the same share of that group's risk, from the covariance of "
        f"their daily returns over the {vm} months to the signal date. A month with fewer than "
        f"{erc_min_returns(vm)} daily returns, a zero variance or no solver answer uses inverse volatility "
        f"(in this run: {info['fallback']})."
    )
    return info


def apply_g1_cash_rule(weights, final_holdings, signal_date, g1, fund_bil_frame,
                       daily_ffill, main_lookbacks, main_lookback_weights):
    """
    The G1 cash rule (added November 7, 2026; G1_DEFAULTS has the tested
    settings). In a cash month, BIL among the final holdings, move a share
    (g1["fraction"]) of BIL's weight to g1["fund"] when both gates pass:

      yield gate  the yield (g1["yield_series"], daily closes) at the signal
                  date is lower than g1["yield_lookback"] observations
                  earlier. Its last close must fall within 7 days of the
                  signal date.
      trend gate  the fund's main score (the main lookback blend, from
                  fund_bil_frame, the monthly closes of the fund and BIL as
                  the caller ranks the basket) is above BIL's.

    Either gate can be turned off (g1["yield_gate"], g1["trend_gate"]). The
    fund needs a price at the signal date. It is added to the holdings, after
    the ranked ones, and BIL stays in them even if its weight reaches 0, so a
    holdings-gate leverage rule still sees a cash month.

    Matches h_cashq in Ideas Harness.py (Unconventional Ideas Test
    2026-11-07), the rule as tested.

    Returns (weights, final_holdings, info). info["status"] is "in" when the
    rule moved weight, otherwise "no cash", "no data", "yield not down" or
    "fund below BIL"; info also carries the yields, scores and moved weight
    it read.
    """
    info = {"status": "no cash"}
    if "BIL" not in final_holdings or weights.get("BIL", 0.0) <= 0:
        return weights, final_holdings, info
    fund = g1["fund"]
    if g1.get("yield_gate", True):
        ys = g1.get("yield_series")
        lag = int(g1.get("yield_lookback", 63))
        v = ys.loc[:signal_date].dropna() if ys is not None else pd.Series(dtype=float)
        if len(v) == 0 or (signal_date - v.index[-1]).days > 7 or len(v) < lag + 1:
            info["status"] = "no data"
            return weights, final_holdings, info
        info["yield_now"], info["yield_then"] = float(v.iloc[-1]), float(v.iloc[-1 - lag])
        if not (info["yield_now"] - info["yield_then"] < 0):
            info["status"] = "yield not down"
            return weights, final_holdings, info
    if fund_bil_frame is None or fund not in daily_ffill.columns:
        info["status"] = "no data"
        return weights, final_holdings, info
    if g1.get("trend_gate", True):
        mom = compute_blended_momentum(fund_bil_frame, main_lookbacks, main_lookback_weights).iloc[-1]
        info["fund_score"], info["bil_score"] = float(mom[fund]), float(mom["BIL"])
        if not (np.isfinite(info["fund_score"]) and np.isfinite(info["bil_score"])):
            info["status"] = "no data"  # the fund has too little history for a score yet
            return weights, final_holdings, info
        if not info["fund_score"] > info["bil_score"]:
            info["status"] = "fund below BIL"
            return weights, final_holdings, info
    px = daily_ffill[fund].loc[:signal_date]
    if len(px) == 0 or not np.isfinite(px.iloc[-1]):
        info["status"] = "no data"
        return weights, final_holdings, info
    out = dict(weights)
    move = float(g1.get("fraction", 0.5)) * out["BIL"]
    out["BIL"] -= move
    out[fund] = out.get(fund, 0.0) + move
    holdings = list(final_holdings) if fund in final_holdings else list(final_holdings) + [fund]
    info["status"], info["moved"] = "in", move
    return out, holdings, info


def g1_label(g1):
    """One line describing a G1 cash rule setting, for the results log and notes."""
    if not g1 or not g1.get("enabled"):
        return "Off"
    fund = g1.get("fund", G1_DEFAULTS["fund"])
    parts = [f"{fund} gets {float(g1.get('fraction', 0.5)):.0%} of BIL's weight in cash months"]
    gates = []
    if g1.get("yield_gate", True):
        gates.append(f"{g1.get('yield_ticker', G1_DEFAULTS['yield_ticker'])} is below its level "
                     f"{int(g1.get('yield_lookback', 63))} trading days earlier")
    if g1.get("trend_gate", True):
        gates.append(f"{fund}'s main score is above BIL's")
    parts.append(" when " + " and ".join(gates) if gates else ", with no gates")
    return "".join(parts)


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
# DAILY OHLC DOWNLOADS, AND THE BITCOIN SIGNAL FORMULAS KEPT FOR RESEARCH
# =============================================================================
# The Bitcoin trend gate and SMA cross gate were removed from the app on
# November 13, 2026: the November 4, 2026 IBIT keep, gate or drop test found
# that both trail plain IBIT. Their two signal formulas,
# compute_radius_trend_state and compute_sma_cross_state, stay below, unused
# by the app, because research scripts load them from this file (the IBIT
# test's IBIT Harness.py and the Main Lookback Recheck's Lookback Harness.py).

def download_gate_ohlc(ticker, start_date_str, force_refresh=False):
    """
    Download daily OHLC for a single signal ticker. The app uses it for the
    G1 cash rule's yield (^FVX by default), and research scripts for
    Bitcoin's daily highs and lows (BTC-USD). The main price cache keeps
    adjusted close only, so these are fetched separately and cached on their
    own in price_cache/gate_ohlc_<ticker>.csv. Returns a DataFrame with High,
    Low, Close columns (daily), or an empty DataFrame on failure. Neither
    series has dividends or splits, so adjusted and raw prices match.
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


def reject_removed_gates(params):
    """
    The Bitcoin trend gate and SMA cross gate were removed on November 13,
    2026. A caller that still turns one on (params["btc_gate"] or
    params["btc_sma_gate"] with "enabled") gets an error rather than a run
    that silently ignores it; {"enabled": False}, as research scripts pass,
    is fine.
    """
    for k in ("btc_gate", "btc_sma_gate"):
        if (params.get(k) or {}).get("enabled"):
            raise ValueError(
                f"params[{k!r}] is enabled, but the Bitcoin gates were removed from the engine on "
                f"November 13, 2026; run a copy of the engine from before then "
                f"(Adaptive Portfolio Dashboard.py.bak_20261113_gates-removal)"
            )


def resolve_borrow_rate(cost, period_start, period_end):
    """
    Resolve leverage_cost_annual to one annual rate for a single holding period.

    Accepts either:
      * a scalar APR, e.g. 0.062 for 6.2%/yr (legacy behaviour, unchanged), or
      * a pandas Series of daily APRs indexed by date, in which case the rate
        returned is the mean APR in effect from period_start to period_end.

    Series fallbacks, in order: mean over the window; last observation on or
    before period_start; the series mean; then 0.0.
    """
    if cost is None:
        return 0.0

    if isinstance(cost, bool):
        return 0.0

    if isinstance(cost, (int, float)):
        return float(cost)

    if isinstance(cost, pd.Series):
        s = pd.to_numeric(cost, errors="coerce").dropna()
        if s.empty:
            return 0.0
        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                return float(s.mean())
        s = s.sort_index()

        window = s.loc[(s.index >= period_start) & (s.index <= period_end)]
        if len(window) > 0:
            return float(window.mean())

        prior = s.loc[s.index <= period_start]
        if len(prior) > 0:
            return float(prior.iloc[-1])

        return float(s.mean())

    try:
        return float(cost)
    except (TypeError, ValueError):
        return 0.0


def resolve_backtest_end(backtest_end):
    """Read a "YYYY-MM" backtest end as the last calendar day of that month.

    run_backtest keeps the results, the trade log and the daily equity dated
    on or before the end, and pandas reads a bare "YYYY-MM" as the first day
    of the month. Until November 3, 2026 an end of 2026-08 therefore stopped
    the results and the daily equity at July while the trade log kept
    August. A full date (YYYY-MM-DD) and a blank end pass through unchanged.
    """
    if isinstance(backtest_end, str):
        s = backtest_end.strip()
        if len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit():
            try:
                return (pd.Timestamp(s) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return backtest_end


def run_backtest(daily_prices, monthly_prices, params):
    """
    Run the adaptive portfolio backtest with the given parameters.

    Execution timing (params["exec_timing"]):
      "last_day"  (default) - allocations (momentum, position weights, G1,
                  leverage) are computed from data through the close of the
                  trading day BEFORE the last trading day of the month, and
                  trades execute at the last trading day's close. This
                  mirrors real-world execution: take the signal from the
                  penultimate close, enter positions near the close on the
                  final trading day. No look-ahead.
      "first_day" - signals are computed from month-end closing prices and
                  trades execute at the NEXT trading day's close (1-day
                  lag). This is the legacy behavior.

    Trading costs (params["cost_per_side"], default DEFAULT_COST_PER_SIDE):
      every rebalance is charged cost_per_side x one-way turnover, where
      one-way turnover is half the sum of |target weight - drifted weight|
      over every asset, so reweighting held funds counts as well as swaps.
      Drifted weights are the previous period's weights carried through
      that period's returns (exposure weights, so leverage counts); the
      first rebalance buys from cash. The cost comes out of the month the
      rebalance opens. On the daily curve a trade day's value is read
      before its trades, so the cost shows from the next day. 0 reproduces
      the gross results exactly. Matches net_path in Signal Flip
      Harness.py (Signal Flip Tests folder).

    G1 cash rule (params["g1"], off unless "enabled"; added November 7,
    2026): after the position weights, apply_g1_cash_rule may move part of
    BIL's weight to the G1 fund, which then counts as a holding for returns,
    costs, leverage and the trade log. While the rule is on, the trade log
    gains a "g1" column with each month's outcome. With it off, every output
    is exactly what it was before the rule was added.

    Signal day (params["signal_offset"]; added November 16, 2026, for the
    Reality Check panel): k puts the signal close k trading days before the
    month's last trading day T, with the trade at the next close. 1 is
    "last_day" and 0 is "first_day", and both run through those paths
    unchanged. Any other k trades 1 - k trading days from T (earlier for a
    positive k), reads its signal at the close before, and measures each
    lookback from an earlier month-end close to that signal close, as
    "last_day" does. Left out, exec_timing alone decides, as before.
    params["fast_daily"] computes the daily equity curve with arrays, the
    same arithmetic in the same order as the day-by-day loop.

    Position sizing (params["sizing"]; added November 13, 2026):
    compute_position_weights weights the final holdings by equal risk
    contribution ("erc", the app's default) or inverse volatility
    ("inverse_vol"). Left out, it is inverse volatility, as before the
    option, so older callers' results do not move. The trade log's "sizing"
    column says how each month was weighted, including the months ERC fell
    back to inverse volatility.
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
    backtest_end = resolve_backtest_end(params.get("backtest_end", ""))
    leverage_params = params.get("leverage_params", {"enabled": False})
    exec_timing = params.get("exec_timing", "last_day")
    # Signal day and the fast daily curve (see the docstring; November 16, 2026).
    signal_offset = params.get("signal_offset")
    if signal_offset is not None:
        signal_offset = int(signal_offset)
        if signal_offset == 1:
            exec_timing, signal_offset = "last_day", None
        elif signal_offset == 0:
            exec_timing, signal_offset = "first_day", None
        else:
            exec_timing = "offset"
    fast_daily = bool(params.get("fast_daily"))
    cost_per_side = params.get("cost_per_side", DEFAULT_COST_PER_SIDE)
    cost_per_side = float(cost_per_side) if cost_per_side is not None else 0.0
    sizing = params.get("sizing", SIZING_IF_UNSET)
    if sizing not in SIZING_LABELS:
        raise ValueError(f"unknown position sizing {sizing!r}")

    reject_removed_gates(params)

    # G1 cash rule (see the docstring). Off unless enabled.
    g1 = params.get("g1") or {}
    g1_on = bool(g1.get("enabled"))
    g1_fund = g1.get("fund") if g1_on else None
    g1_has_fund = (g1_on and g1_fund in monthly_prices.columns and g1_fund in daily_prices.columns
                   and "BIL" in monthly_prices.columns)

    min_lookback = max(main_lookbacks)
    valid_months = monthly_prices.index[min_lookback:]
    # The equity seat's stand-in before its fund's own history (EQUITY_SEAT_BEFORE).
    seat_standins = equity_seat_standins(params)

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
        last_day = last_trading_day(month_end_date)
        if exec_timing != "offset" or last_day is None:
            return last_day
        # "offset": the trade is 1 - signal_offset trading days from the
        # month's last trading day (earlier for a positive offset).
        pos = int(all_trading_dates.searchsorted(last_day)) + 1 - signal_offset
        if pos < 0 or pos >= len(all_trading_dates):
            return None
        return all_trading_dates[pos]

    # Forward-filled daily prices for signal-date lookups (computed once)
    daily_ffill = daily_prices.ffill()

    results = []
    trade_log = []
    daily_portfolio_values = {}  # {date: portfolio_value} for daily drawdown
    cumulative_multiplier = 1.0  # tracks growth across holding periods
    prev_drift = {}  # exposure weights drifted to the last exit close (cash at the start)

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
        if exec_timing in ("last_day", "offset"):
            signal_date = exec_date_entry - pd.Timedelta(days=1)
        else:
            signal_date = rebal_date

        def signal_monthly_frame(cols):
            """Monthly closes through rebal_date, with the current month's
            value replaced by the close as of signal_date (last_day mode)."""
            frame = monthly_prices[cols].loc[:rebal_date].copy()
            if exec_timing in ("last_day", "offset") and len(frame) > 0:
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
                    basket.append(t)

        selected_equities = []
        if equity_top_n > 0 and len(equity_tickers) > 0:
            eq_available = []
            for t in equity_pool_for(equity_tickers, rebal_date, seat_standins, monthly_prices.columns):
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

        weights, sizing_note = compute_position_weights(
            basket_daily, final_holdings, signal_date, vol_months, sizing
        )

        # G1 cash rule: may move part of BIL's weight to the G1 fund.
        g1_status = "off"
        if g1_on:
            weights, final_holdings, g1_info = apply_g1_cash_rule(
                weights, final_holdings, signal_date, g1,
                signal_monthly_frame([g1_fund, "BIL"]) if g1_has_fund else None,
                daily_ffill, main_lookbacks, main_lookback_weights,
            )
            g1_status = g1_info["status"]

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

        # Borrowing cost on leveraged portion (only when actually leveraged).
        # leverage_cost_annual may be a scalar APR (0.062 == 6.2%/yr) or a
        # pandas Series of daily APRs, in which case this period is charged the
        # average rate actually in effect between entry and exit.
        lev_cost_annual = resolve_borrow_rate(
            leverage_params.get("leverage_cost_annual", 0.0),
            exec_date_entry, exec_date_exit
        )
        monthly_borrow_cost = 0.0
        if lev_mult > 1.0 and lev_cost_annual > 0:
            monthly_borrow_cost = (lev_mult - 1.0) * (lev_cost_annual / 12.0)
            port_return -= monthly_borrow_cost

        # --- Trading costs ---
        # One-way turnover between this rebalance's target exposures and the
        # previous holdings drifted to this trade's close, charged at
        # cost_per_side (see the docstring). Skipped at zero cost so gross
        # results reproduce bit for bit.
        target_exposure = {t: lev_mult * weights.get(t, 0.0) for t in final_holdings}
        turnover = 0.5 * sum(
            abs(target_exposure.get(t, 0.0) - prev_drift.get(t, 0.0))
            for t in sorted(set(target_exposure) | set(prev_drift))
        )
        trading_cost = cost_per_side * turnover if cost_per_side > 0 else 0.0
        cost_factor = 1.0 - trading_cost
        port_return_gross = port_return
        if cost_per_side > 0:
            port_return = (1.0 + port_return_gross) * cost_factor - 1.0

        # Carry this period's exposures to its exit close for the next rebalance
        growth = 1.0 + port_return_gross
        if growth > 0:
            prev_drift = {
                t: target_exposure[t] * (1.0 + holding_details.get(t, {}).get("return", 0.0)) / growth
                for t in final_holdings
            }
        else:
            prev_drift = {}

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

        if fast_daily and n_holding_days > 0:
            # The loop below with arrays (November 16, 2026): the same
            # arithmetic in the same order, read from the forward-filled
            # prices, which hold each day's last price at or before it.
            _dwr = np.zeros(n_holding_days)
            for t in final_holdings:
                if t not in entry_px:
                    continue
                _dwr = _dwr + weights.get(t, 0.0) * (
                    (daily_ffill[t].reindex(holding_days).to_numpy(dtype=float) / entry_px[t]) - 1.0
                )
            _acc = np.zeros(n_holding_days)
            if monthly_borrow_cost > 0:
                _acc = monthly_borrow_cost * np.arange(1, n_holding_days + 1) / n_holding_days
            _dcf = np.full(n_holding_days, cost_factor)
            _dcf[0] = 1.0
            _vals = cumulative_multiplier * _dcf * (1 + _dwr * lev_mult - _acc) * INITIAL_CAPITAL
            daily_portfolio_values.update(zip(holding_days, _vals.tolist()))
            holding_days = holding_days[:0]   # the loop below then has nothing left to do

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
            # A trade day's value is read before its trades, so the trading
            # cost shows from the next day (as in net_path).
            day_cost_factor = cost_factor if day_idx > 0 else 1.0
            daily_portfolio_values[day] = (
                cumulative_multiplier * day_cost_factor
                * (1 + day_weighted_return * lev_mult - accrued_cost) * INITIAL_CAPITAL
            )

        # Update cumulative multiplier at period end
        cumulative_multiplier *= (1 + port_return)

        results.append({
            "date": next_month_end,
            "portfolio_return": port_return,
            "leverage_mult": lev_mult,
            "borrow_cost": monthly_borrow_cost,
            "turnover": turnover,
            "trading_cost": trading_cost,
        })

        holdings_str_parts = []
        for t in final_holdings:
            if t in holding_details:
                w = weights.get(t, 0.0)
                r = holding_details[t]["return"]
                holdings_str_parts.append(f"{t}: {w:.1%} (ret: {r:+.2%})")

        trade_log.append({
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "hold_month": next_month_end.strftime("%Y-%m"),
            "exec_entry": exec_date_entry.strftime("%Y-%m-%d"),
            "exec_exit": exec_date_exit.strftime("%Y-%m-%d") if i < len(valid_months) - 2 else "",
            "basket": ", ".join(basket),
            "equities_selected": ", ".join(selected_equities) if selected_equities else "N/A",
            "g1": g1_status,
            "sizing": sizing_note,
            "final_holdings": " | ".join(holdings_str_parts),
            "leverage": f"{lev_mult:.2f}x",
            "lev_regime": lev_regime,
            "turnover": f"{turnover:.1%}",
            "trading_cost": f"{trading_cost:.3%}",
            "portfolio_return": f"{port_return:+.2%}",
        })
        if not g1_on:
            del trade_log[-1]["g1"]  # the trade log keeps its columns with G1 off

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
    sizing = params.get("sizing", SIZING_IF_UNSET)
    reject_removed_gates(params)

    # G1 cash rule (see run_backtest), applied to this signal too.
    g1 = params.get("g1") or {}
    g1_fund = g1.get("fund") if g1.get("enabled") else None

    min_lookback = max(main_lookbacks)

    # Build basket
    basket = []
    for t in fixed_tickers:
        if t in monthly_prices_local.columns:
            ps = monthly_prices_local[t].loc[:signal_date].dropna()
            if len(ps) > min_lookback:
                basket.append(t)

    selected_equities = []
    if equity_top_n > 0 and len(equity_tickers) > 0:
        eq_available = []
        for t in equity_pool_for(equity_tickers, signal_date, equity_seat_standins(params),
                                 monthly_prices_local.columns):
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
        all_assets = list(set(fixed_tickers + equity_tickers + ([g1_fund] if g1_fund else [])))
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

    weights, sizing_note = compute_position_weights(
        daily_prices, final_holdings, signal_date, vol_months, sizing
    )

    g1_info = None
    if g1_fund:
        g1_frame = None
        if g1_fund in monthly_prices_local.columns and "BIL" in monthly_prices_local.columns:
            g1_frame = monthly_prices_local[[g1_fund, "BIL"]].loc[:signal_date]
        weights, final_holdings, g1_info = apply_g1_cash_rule(
            weights, final_holdings, signal_date, g1, g1_frame,
            daily_prices.ffill(), main_lookbacks, main_lookback_weights,
        )

    if full_details:
        for row in asset_details:
            t = row["Asset"]
            row["Selected"] = t in final_holdings
            row["Weight"] = weights.get(t, 0.0) if t in final_holdings else 0.0
            row["Equity Pick"] = t in selected_equities
            if g1_fund:
                row["G1 Fund"] = t == g1_fund

        asset_details.sort(key=lambda x: x.get("Blended Score", -999), reverse=True)

    out = {
        "signal_date": signal_date,
        "asset_details": asset_details,
        "selected_equities": selected_equities,
        "final_holdings": final_holdings,
        "weights": weights,
        "basket": basket,
        "sizing_note": sizing_note,
    }
    if g1_fund:
        out["g1_info"] = g1_info
    return out


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
    for t in equity_pool_for(equity_tickers, signal_date, equity_seat_standins(params),
                             monthly_prices_local.columns):
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
        prev_exec_date        the day that month's rebalance trades: its last
                              trading day, or the day params["signal_offset"]
                              picks (None in "first_day" mode, and when that
                              day has not happened yet)
        prev_data_date        the close the prev signal is computed from:
                              the trading day before prev_exec_date
        prev_holdings         list of tickers from the prev month-end signal
        prev_weights          dict of {ticker: weight} from the prev month-end signal
        sizing_note           how the live signal was weighted: "ERC", "Inverse vol"
                              or "Inverse vol (ERC fallback: <reason>)"
        prev_sizing_note      the same for the prev month-end signal
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
    #
    # In "last_day" mode (Tyler's real process) the signal for that
    # rebalance is taken from the close of the trading day BEFORE the last
    # trading day of the month, exactly as run_backtest does: signal cutoff
    # = execution day minus one calendar day, with the month's monthly bar
    # replaced by the forward-filled close as of that cutoff. The result
    # therefore matches the backtest trade log for that month.
    prev_month_end_date = None
    prev_holdings = []
    prev_weights = {}
    prev_sizing_note = None
    prev_exec_date = None
    prev_data_date = None
    completed = monthly_prices.index[monthly_prices.index <= last_daily]
    prev_candidates = [d for d in completed if d != signal_date]
    if len(prev_candidates) > 0:
        prev_month_end_date = prev_candidates[-1]
        prev_signal_date = prev_month_end_date
        prev_monthly = monthly_prices
        # The trade day is params["signal_offset"] when set (any of the Trade
        # execution timing choices, November 17, 2026), else the exec_timing
        # choice: 1 is the last trading day ("last_day"), 0 the first trading
        # day of next month ("first_day"). Any other k trades 1 - k trading days
        # from the month's last trading day, as run_backtest does; a day that
        # has not happened yet leaves the month-end close reference below.
        _k = params.get("signal_offset")
        _k = int(_k) if _k is not None else (0 if params.get("exec_timing", "last_day") == "first_day" else 1)
        if _k != 0:
            cal = market_trading_dates(daily_prices)
            in_month = cal[(cal.year == prev_month_end_date.year)
                           & (cal.month == prev_month_end_date.month)
                           & (cal <= prev_month_end_date)]
            if len(in_month) > 0:
                prev_exec_date = in_month[-1]
                if _k != 1:
                    _pos = int(cal.searchsorted(prev_exec_date)) + 1 - _k
                    prev_exec_date = cal[_pos] if 0 <= _pos < len(cal) else None
            if prev_exec_date is not None:
                prev_signal_date = prev_exec_date - pd.Timedelta(days=1)
                asof = daily_prices.ffill().loc[:prev_signal_date]
                if len(asof) > 0:
                    prev_monthly = monthly_prices.loc[:prev_month_end_date].copy()
                    prev_monthly.iloc[-1] = asof.iloc[-1].reindex(prev_monthly.columns)
                    prev_monthly.index = pd.DatetimeIndex(
                        list(prev_monthly.index[:-1]) + [prev_signal_date])
                _before = cal[cal <= prev_signal_date]
                prev_data_date = _before.max() if len(_before) else None
        prev = _compute_signal_at(
            daily_prices, prev_monthly, params, prev_signal_date,
            full_details=False
        )
        if prev is not None:
            prev_holdings = prev["final_holdings"]
            prev_weights = prev["weights"]
            prev_sizing_note = prev["sizing_note"]

    return {
        **live,
        "equity_details": equity_details,
        "prev_month_end_date": prev_month_end_date,
        "prev_exec_date": prev_exec_date,
        "prev_data_date": prev_data_date,
        "prev_holdings": prev_holdings,
        "prev_weights": prev_weights,
        "prev_sizing_note": prev_sizing_note,
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

    # Ulcer Index and Martin ratio (Peter Martin's Ulcer Performance Index),
    # from the same month-end drawdowns as Calmar: the root mean square of
    # the drawdown in every month, and the return above the risk-free rate
    # divided by it. Added November 8, 2026.
    ulcer_index = float(np.sqrt(np.mean(drawdown.values ** 2)))
    martin = (cagr - rf_annual) / ulcer_index if ulcer_index > 0 else 0.0

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

    # Worst/best week from week-ending (Friday) closes of the daily equity
    # curve, the same method the Chart Dashboard uses. The first partial
    # week drops out because it has no prior Friday close.
    worst_week = best_week = np.nan
    worst_week_date = best_week_date = None
    if daily_equity is not None and len(daily_equity) > 1:
        _de = daily_equity.copy()
        _de.index = pd.to_datetime(_de.index)
        weekly_pct = _de.resample("W-FRI").last().dropna().pct_change().dropna()
        if len(weekly_pct) > 0:
            worst_week = weekly_pct.min()
            best_week = weekly_pct.max()
            worst_week_date = weekly_pct.idxmin()
            best_week_date = weekly_pct.idxmax()

    # Worst/best month from the monthly return series (month-end index, so
    # the date labels the month the return belongs to).
    worst_month = r.min()
    best_month = r.max()
    worst_month_date = r.idxmin()
    best_month_date = r.idxmax()

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
        "Worst Week": worst_week,
        "Worst Week Date": worst_week_date,
        "Best Week": best_week,
        "Best Week Date": best_week_date,
        "Worst Month": worst_month,
        "Worst Month Date": worst_month_date,
        "Best Month": best_month,
        "Best Month Date": best_month_date,
        "Calmar Ratio": calmar,
        "Ulcer Index": ulcer_index,
        "Martin Ratio": martin,
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
        "Position Sizing": SIZING_LABELS[params.get("sizing", SIZING_IF_UNSET)],
        "ERC Fallback Months": params.get("erc_fallback_summary", "not an ERC run"),
        "Execution": params.get("exec_timing_label", params.get("exec_timing", "last_day")),
        "Trading Cost (per side)": f"{params.get('cost_per_side', DEFAULT_COST_PER_SIDE):.2%}",
        "Leverage": lev_summary,
        "G1 Cash Rule": g1_label(params.get("g1")),
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
        "Martin": strat_metrics.get("Martin Ratio", 0),
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
        if "Position Sizing" not in existing.columns:
            # Every run logged before November 13, 2026 used inverse
            # volatility, the only position sizing the engine had until then.
            existing["Position Sizing"] = SIZING_LABELS["inverse_vol"]
            existing["ERC Fallback Months"] = "not an ERC run"
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    # Keep the position sizing columns beside the volatility window.
    cols = list(combined.columns)
    if "Vol Window (months)" in cols:
        for c in ("ERC Fallback Months", "Position Sizing"):
            if c in cols:
                cols.remove(c)
                cols.insert(cols.index("Vol Window (months)") + 1, c)
        combined = combined[cols]

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
# RESULTS PAGE: CHARTS AND PAGE PIECES  (page facelift, November 17, 2026)
# =============================================================================
# The results page is six tabs (Overview, Reality Check, Signals, Returns and
# Drawdowns, Strategy Health, Trades and Data). Charts are Plotly, drawn with
# the tokens above; the rolling figures come from the compute_rolling_*
# functions, unchanged. Period and frequency buttons live inside the Plotly
# figures (updatemenus), because a Streamlit widget would rerun the script and
# clear the results, which exist only on the run that pressed Run Backtest.
# The matplotlib plot_* functions above are no longer drawn by the app.

import html as _ae_html

AE_ASSET_NAMES = {
    "TLT": "Long-term Treasuries", "GLDM": "Gold", "GLD": "Gold", "BIL": "T-bills",
    "BTAL": "Defensive equity", "HGER": "Broad commodities", "IBIT": "Bitcoin", "BTC-USD": "Bitcoin",
    "QQQ": "Nasdaq-100", "IWM": "Russell 2000", "DXJ": "Japan, yen-hedged",
    "DEVXJ_X": "Developed markets ex US and Japan (EMXC's seat before 1993)",
    "EMXC": "Emerging markets ex China", "MNK_SYNTH": "Nikkei 225 futures (synthetic)",
    "IEI": "Intermediate Treasuries",
}
AE_BENCH_NAMES = {"SPY": "S&P 500 (SPY)", "AOA": "80/20 (AOA)", "AOR": "60/40 (AOR)"}


def ae_esc(s):
    return _ae_html.escape(str(s))


def ae_local_now():
    """Wall-clock time in Mountain time. Streamlit Community Cloud runs on UTC, so the
    deployed app would otherwise stamp the page seven hours ahead (as the Chart Dashboard
    learned; its local_now() does the same)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Denver")).replace(tzinfo=None)
    except Exception:
        return datetime.now()


def ae_md(html_text):
    """Draw one HTML block. Lines are joined, so Markdown never reads a blank line or an indent."""
    st.markdown(html_text, unsafe_allow_html=True)


def ae_neg(text):
    """A typographic minus for display text (never for code or CSS)."""
    return str(text).replace("-", "−")


def ae_asset_name(t):
    return AE_ASSET_NAMES.get(t) or DEFAULT_FIXED_ASSETS.get(t) or DEFAULT_EQUITY_ETFS.get(t) or ""


def ae_pill(text, tone="neutral", glyph=None):
    bg, fg = RC_TONES.get(tone, RC_TONES["neutral"])
    g = f'<span aria-hidden="true">{ae_esc(glyph)}</span>' if glyph else ""
    return f'<span class="ae-pill" style="background:{bg};color:{fg}">{g}<span>{ae_esc(text)}</span></span>'


def ae_tag(text):
    return f'<span class="ae-tag">{ae_esc(text)}</span>'


def ae_now(text):
    return f'<span class="ae-now">{ae_esc(text)}</span>'


def ae_head(eyebrow, title, desc_html="", links=None):
    """A tab's opening: eyebrow, serif title, description and its On this page row."""
    parts = ['<div class="ae-head">']
    if eyebrow:
        parts.append(f'<div class="ae-eyebrow">{ae_esc(eyebrow)}</div>')
    parts.append(f'<div class="ae-title">{ae_esc(title)}</div>')
    if desc_html:
        parts.append(f'<div class="ae-desc">{desc_html}</div>')
    if links:
        # No label. The link last followed is drawn solid (CSS :target, one rule per link, written
        # here because the links differ by tab); until one is followed, the first link is.
        ok = [a for a, _ in links if a[:1].isalpha() and a.replace("-", "").replace("_", "").isalnum()]
        rules = "".join(f'body:has(#{a}:target) .ae-otp .ae-otp-seg a[href="#{a}"]'
                        '{background:#4B3F8F;color:#ffffff !important;font-weight:700}' for a in ok)
        rules += "".join(f'body:has(#{a}:target) .ae-otp .ae-otp-seg:has(a[href="#{a}"]) a:first-child'
                         '{background:transparent;color:#3A3170 !important;font-weight:600}'
                         for a in ok if a != links[0][0])
        parts.append((f"<style>{rules}</style>" if rules else "")
                     + '<div class="ae-otp"><span class="ae-otp-seg">'
                     + "".join(f'<a href="#{a}">{ae_esc(t)}</a>' for a, t in links) + "</span></div>")
    parts.append("</div>")
    return "".join(parts)


def ae_h3(title, sub_html="", right_html="", anchor=None):
    aid = f' id="{anchor}"' if anchor else ""
    sub = f'<div class="ae-h3-s">{sub_html}</div>' if sub_html else ""
    right = f'<div class="ae-row">{right_html}</div>' if right_html else ""
    return (f'<div class="ae-h3"{aid}><div style="flex:1 1 420px;min-width:0">'
            f'<div class="ae-h3-t">{ae_esc(title)}</div>{sub}</div>{right}</div>')


def ae_gap_pill(s_val, b_val, kind, higher_better=True, digits=2):
    """The gap between the strategy and a benchmark as a colored pill."""
    if s_val is None or b_val is None or not np.isfinite(s_val) or not np.isfinite(b_val):
        return ""
    if kind == "dd":                      # drawdowns: values are negative; shallower is better
        d = (abs(b_val) - abs(s_val)) * 100
        if abs(d) < 0.05:
            return ae_pill("level", "neutral")
        return ae_pill(f"{abs(d):.1f} pts {'shallower' if d > 0 else 'deeper'}", "good" if d > 0 else "bad")
    if kind == "pct":
        d = (s_val - b_val) * 100
        txt = f"{d:+.1f} pts"
    else:
        d = s_val - b_val
        txt = f"{d:+.{digits}f}"
    if abs(d) < (0.05 if kind == "pct" else 0.005):
        return ae_pill("level", "neutral")
    good = (d > 0) == higher_better
    return ae_pill(ae_neg(txt), "good" if good else "bad")


# ---- Plotly ----

def ae_chart_config(name):
    """Plotly toolbar settings: PNG downloads at 3x, named for the chart and the time."""
    now = ae_local_now()
    stamp = f"{now:%Y-%m-%d} {now.hour % 12 or 12}.{now:%M} {'a.m.' if now.hour < 12 else 'p.m.'}"
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": f"{name} {stamp}", "scale": 3},
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    }


def ae_plot(fig, name, key):
    st.plotly_chart(fig, width="stretch", theme=None, config=ae_chart_config(name), key=key)


def ae_layout(fig, height=340, yfmt=None, ytitle=None, log=False, xhover="%b %Y", top=12, legend=True,
              yrange=None, tickprefix=None):
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=AE_SANS, size=13, color=AE_AXIS),
        margin=dict(l=8, r=12, t=top, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AE_CARD, font=dict(family=AE_SANS, size=13, color=AE_INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                    font=dict(size=13, color=AE_INK), itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    fig.update_xaxes(showgrid=False, linecolor=AE_CONTROL, ticks="outside", tickcolor=AE_CONTROL,
                     hoverformat=xhover, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikecolor="#B4B8BF", spikedash="solid")
    fig.update_yaxes(gridcolor=AE_PLOT_GRID, zeroline=False, tickformat=yfmt, automargin=True,
                     title=dict(text=ytitle, font=dict(size=12.5, color=AE_BODY)) if ytitle else None,
                     type="log" if log else "linear", range=yrange, tickprefix=tickprefix)
    return fig


def ae_buttons(labels, args_list, y=1.02):
    """Plotly's own buttons (no Streamlit rerun), styled like the site's pills."""
    return [dict(
        type="buttons", direction="right", active=0, showactive=True,
        x=0, xanchor="left", y=y, yanchor="bottom", pad=dict(r=6, t=0, b=4, l=0),
        bgcolor="#ffffff", bordercolor=AE_CONTROL, borderwidth=1,
        font=dict(family=AE_SANS, size=12.5, color=AE_BODY),
        buttons=[dict(label=lab, method="update", args=a) for lab, a in zip(labels, args_list)],
    )]


# ---- Extra comparison ticker (November 21, 2026) ----
# One more line beside the benchmarks, typed into the sidebar's "Compare with another ticker" box, which is
# blank at startup. Blank, nothing here runs and the page is what it was. Filled, the ticker's prices load
# on their own (cmp_load), outside the backtest's price table, so they cannot touch the run's trading days,
# signals or results. The ticker then joins balanced_data as one more entry, marked "custom", and so reaches
# the Growth, Drawdowns and Annual returns charts and the Performance Summary by the path the 60/40 and
# 80/20 take. Build and checks:
# C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System\Backtest Engine\Comparison Ticker 2026-11-21\
# Its color (added later the same day): a preset from CMP_PALETTE, picked in the sidebar, or any hex code
# typed there, which wins when it is a valid one. Every preset sits apart from the strategy's indigo, the
# two blues and the S&P 500's gray. The one color is used wherever the ticker is drawn: its lines, bars,
# legend swatches, the drawdown chart's worst-point dot and label, and its Performance Summary header.
# The starting color is Charcoal (Tyler, November 21, 2026: a dark gray, not a bright color). It is darker
# than the S&P 500's gray, and on the drawdown chart it is a solid line where the S&P 500's is dashed.
CMP_PALETTE = {
    "Charcoal": "#2B2F36", "Emerald": "#1F9D6B", "Teal": "#0E8C8C", "Olive": "#7A8B2A", "Gold": "#C99A0B",
    "Burnt orange": "#D9622B", "Crimson": "#C0392B", "Rose": "#E0527A", "Magenta": "#C2278F", "Brown": "#8A5A3B",
}
CMP_COLOR = CMP_PALETTE["Charcoal"]   # the starting color: the palette's first


def cmp_hex(raw):
    """A typed hex color code as "#RRGGBB" in upper case, or None when it is not one. Takes "#1f9d6b",
    "1F9D6B" and the three-digit form "#2a7"."""
    s = str(raw or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9A-Fa-f]{3}", s):
        s = "".join(ch * 2 for ch in s)
    return "#" + s.upper() if re.fullmatch(r"[0-9A-Fa-f]{6}", s) else None


def cmp_clean(raw):
    """The typed ticker in upper case, less anything a ticker never holds. Blank stays blank."""
    return re.sub(r"[^A-Za-z0-9.\-^=_]", "", str(raw or "")).upper()


def cmp_load(ticker, daily_prices, start_date_str):
    """(the ticker's closes on the run's trading days, None), or (None, why the ticker was left out).

    A ticker the run already holds (a basket fund, a stand-in) is read from the run's price table, with
    whatever earlier history the Backfill Rules gave it. Any other is downloaded by download_data, through
    the price cache, then set on the run's trading days: each day takes the ticker's last close on or before
    it, between the ticker's first and last close. A forced refresh is never passed on, because a forced
    download replaces the price cache with what it fetched."""
    missing = f"no prices were found for {ticker} (check the spelling)"
    if ticker in daily_prices.columns:
        s = daily_prices[ticker].dropna()
    else:
        try:
            got = download_data((ticker,), start_date_str)
        except MissingPriceDataError:
            return None, missing
        except Exception as exc:
            return None, f"{ticker}'s prices could not be loaded ({exc})"
        if got is None or got.empty or ticker not in got.columns:
            return None, missing
        s = got[ticker].dropna()
    if s.empty:
        return None, missing
    cal = market_trading_dates(daily_prices)
    days = cal[(cal >= s.index[0]) & (cal <= s.index[-1])]
    s = s.reindex(s.index.union(days)).ffill().reindex(days).dropna()
    if len(s) < 2:
        return None, f"{ticker} has no prices on this run's trading days"
    return s, None


def ae_series_set(strat_returns, bench_returns, balanced_data, lev_tag=""):
    """(name, monthly returns, color, width, dash) for the strategy and its benchmarks, in legend order.
    The extra comparison ticker, when one was typed, comes last."""
    out = [("Strategy", strat_returns, BRAND_PURPLE, 2.6, None)]
    for e in sorted((e for e in balanced_data if not e.get("custom")), key=lambda e: 0 if e["label"] == "80/20" else 1):
        r = e["returns"].dropna() if e.get("returns") is not None else None
        if r is not None and len(r):
            out.append((AE_BENCH_NAMES.get(e["ticker"], e["label"]) + lev_tag, r, e["color"], 1.7, None))
    out.append((AE_BENCH_NAMES.get(BENCHMARK_TICKER, BENCHMARK_TICKER) + lev_tag, bench_returns, BRAND_INDIGO, 1.8, None))
    for e in balanced_data:
        if e.get("custom"):
            r = e["returns"].dropna() if e.get("returns") is not None else None
            if r is not None and len(r):
                out.append((AE_BENCH_NAMES.get(e["ticker"], e["label"]) + lev_tag, r, e["color"], 1.9, None))
    return out


def ae_fig_growth(strat_returns, bench_returns, balanced_data, lev_tag=""):
    """Growth of $10,000 on a log scale, with Plotly buttons for the last 20, 10 and 5 years."""
    sets = ae_series_set(strat_returns, bench_returns, balanced_data, lev_tag)
    idx = strat_returns.index
    periods = [("Since " + idx[0].strftime("%b %Y"), None)]
    for yrs in (20, 10, 5):
        if len(idx) > yrs * 12 + 6:
            periods.append((f"{yrs} years", yrs))
    fig = go.Figure()
    vis_all = []
    lo_all, hi_all = np.inf, 0.0
    leg_chars = 0
    for p_i, (label, yrs) in enumerate(periods):
        for rank, (name, r, color, width, dash) in reversed(list(enumerate(sets))):
            r = r.reindex(idx).dropna()
            if yrs is None:
                start = r.index[0] - pd.offsets.MonthEnd(1)
                w = pd.concat([pd.Series([1.0], index=[start]), (1 + r).cumprod()])
            else:
                cut = idx[-(yrs * 12) - 1]
                rr = r[r.index > cut]
                # A line that begins inside the period (an extra comparison ticker with a short
                # history) starts at its own first month, not at the period's first day.
                start = cut
                if len(rr) and rr.index[0] - pd.offsets.MonthEnd(1) > cut:
                    start = rr.index[0] - pd.offsets.MonthEnd(1)
                w = pd.concat([pd.Series([1.0], index=[start]), (1 + rr).cumprod()])
            v = INITIAL_CAPITAL * w
            n_years = max((len(w) - 1) / 12.0, 1e-9)
            cagr = (v.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1
            sub = f"${v.iloc[-1]:,.0f} · {ae_neg(f'{cagr:.1%}')}"
            grp = f"{name}|{p_i}"
            # Legend drops the ticker ("80/20 (AOA)" -> "80/20"); hover and the
            # Performance Summary header keep it.
            short = re.sub(r" \([A-Z]+\)", "", name)
            label = f"<b>{ae_esc(short)}</b>" if rank == 0 else ae_esc(short)
            leg_chars = max(leg_chars, len(short), len(re.sub(r"<[^>]+>", "", sub)))
            lo_all, hi_all = min(lo_all, float(v.min())), max(hi_all, float(v.max()))
            fig.add_trace(go.Scatter(
                x=v.index, y=v.values, mode="lines", line=dict(color=color, width=width, dash=dash),
                name=ae_esc(name), legendgroup=grp, showlegend=False,
                hovertemplate=f"{ae_esc(name)}: %{{y:$,.0f}}<extra></extra>", visible=(p_i == 0),
            ))
            vis_all.append(p_i)
            # Legend-only twin: Plotly draws a line segment for a line trace, so a
            # point-free marker trace in the same group supplies the square swatch
            # (matching the Performance Summary header) and a click still hides the line.
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=10, color=color, line=dict(width=0)),
                legendrank=rank + 1, legendgroup=grp, hoverinfo="skip",
                name=f"{label}<br><span style='font-size:12px;color:{AE_BODY}'>{sub}</span>",
                visible=(p_i == 0),
            ))
            vis_all.append(p_i)
    args = [[{"visible": [v == p_i for v in vis_all]}, {"xaxis.autorange": True, "yaxis.autorange": True}]
            for p_i in range(len(periods))]
    ae_layout(fig, height=440, log=True, top=46)
    ticks = []
    e = int(np.floor(np.log10(max(lo_all, 1.0))))
    while 10 ** e <= hi_all * 1.0001:
        for m in (1, 2, 5):
            if lo_all * 0.98 <= m * 10 ** e <= hi_all * 1.02:
                ticks.append(m * 10 ** e)
        e += 1

    def _usd(t):
        return f"${t / 1e6:g}M" if t >= 1e6 else (f"${t / 1e3:g}K" if t >= 1e3 else f"${t:g}")

    fig.update_yaxes(tickvals=ticks, ticktext=[_usd(t) for t in ticks])
    if len(periods) > 1:
        fig.update_layout(updatemenus=ae_buttons([p[0] for p in periods], args))
    # Equal-width legend items, sized to the longest label, so spacing is even.
    fig.update_layout(legend=dict(y=-0.1, valign="top", itemsizing="constant", groupclick="togglegroup",
                                  entrywidth=max(120, 7 * leg_chars + 16), entrywidthmode="pixels"))
    return fig


def ae_drawdown(w):
    return w / w.cummax() - 1


# The drawdown chart's lines look like the Chart Dashboard's Drawdowns Over Time chart (app.py DD_LINE, DD_DASH,
# DD_WIDTH, DD_TEXT). Each entry: line color, dash, width, label color.
AE_DD_LOOK = {
    "Strategy": ("#4B3F8F", None, 2.4, "#4B3F8F"),
    "80/20": ("#4361EE", "7px,4px", 2.0, "#3451D1"),
    "60/40": ("#4CC9F0", "3px,4px", 2.6, "#0B7AA3"),
    "S&P 500": ("#6B7280", "11px,4px,3px,4px", 2.0, "#4B5563"),
}
# The strategy's fill is lightest at 0% and darkest at its deepest point (gradient stop 0 is the lowest value).
AE_DD_FILL = [[0.0, "rgba(75,63,143,0.26)"], [1.0, "rgba(75,63,143,0.04)"]]


def ae_dd_look(name, color):
    for key, look in AE_DD_LOOK.items():
        if name.startswith(key):
            return look
    return color, None, 1.6, color


def ae_dd_worst_labels(worst, x_lo, x_hi, plot_h=250.0, gap_px=34.0):
    """Where each worst-point label sits (the Chart Dashboard's rule). A label starts level with its dot and moves
    down when the label above it is close in time and less than gap_px higher. worst: dicts with v (depth,
    negative) and t (date); each gets frac (position across the chart) and ly (label height)."""
    if not worst:
        return worst
    ppu = plot_h / (1.1 * max(0.05, -min(w["v"] for w in worst)))   # pixels per 100% of drawdown
    span = max(1.0, (x_hi - x_lo).total_seconds())
    prev = None
    for w in sorted(worst, key=lambda w: -w["v"]):
        w["frac"] = (w["t"] - x_lo).total_seconds() / span
        w["ly"] = w["v"]
        if prev is not None and abs(w["frac"] - prev["frac"]) < 0.09 and (prev["ly"] - w["ly"]) * ppu < gap_px:
            w["ly"] = prev["ly"] - gap_px / ppu
        prev = w
    return worst


def ae_fig_drawdowns(strat_returns, bench_returns, balanced_data, daily_equity, spy_daily_equity, lev_tag=""):
    """Drawdowns from the last high, daily (default) or month-end, switched by Plotly buttons. The strategy is a
    solid purple line over a fill that deepens with depth, the benchmarks are dashed, and each portfolio's
    worst point is a ringed dot with its depth and date."""
    sets = ae_series_set(strat_returns, bench_returns, balanced_data, lev_tag)
    daily = {}
    if daily_equity is not None and len(daily_equity):
        daily["Strategy"] = daily_equity
        for e in balanced_data:
            de = e.get("daily_equity")
            if de is not None and len(de):
                daily[AE_BENCH_NAMES.get(e["ticker"], e["label"]) + lev_tag] = de
        if spy_daily_equity is not None and len(spy_daily_equity):
            daily[AE_BENCH_NAMES.get(BENCHMARK_TICKER, BENCHMARK_TICKER) + lev_tag] = spy_daily_equity
    modes = (["daily"] if daily else []) + ["monthly"]
    fig = go.Figure()
    tags = []
    for m in modes:
        worst, x_lo, x_hi = [], None, None
        for rank, (name, r, color, width, dash) in reversed(list(enumerate(sets))):
            if m == "daily":
                if name not in daily:
                    continue
                dd = ae_drawdown(daily[name].dropna())
            else:
                rr = r.dropna()
                w = pd.concat([pd.Series([1.0], index=[rr.index[0] - pd.offsets.MonthEnd(1)]), (1 + rr).cumprod()])
                dd = ae_drawdown(w)
            look_color, look_dash, look_width, look_text = ae_dd_look(name, color)
            fill = (dict(fill="tozeroy", fillgradient=dict(type="vertical", colorscale=AE_DD_FILL))
                    if name == "Strategy" else {})
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values, mode="lines", line=dict(color=look_color, width=look_width, dash=look_dash),
                legendrank=rank + 1, legendgroup=name, **fill,
                name=f"<b>{ae_esc(name)}</b><br><span style='font-size:12px;color:{AE_BODY}'>worst {ae_neg(f'{dd.min():.1%}')}</span>",
                hovertemplate=f"{ae_esc(name)}: %{{y:.1%}}<extra></extra>", visible=(m == modes[0]),
            ))
            tags.append(m)
            x_lo = dd.index[0] if x_lo is None else min(x_lo, dd.index[0])
            x_hi = dd.index[-1] if x_hi is None else max(x_hi, dd.index[-1])
            if np.isfinite(dd.min()) and dd.min() < -0.0005:
                worst.append(dict(name=name, v=float(dd.min()), t=pd.Timestamp(dd.idxmin()),
                                  color=look_color, text=look_text))
        # Each portfolio's worst point: a ringed dot, with its depth and date beside it. The dot and label share
        # the line's legend group, so a legend click hides or shows all three together.
        for w in ae_dd_worst_labels(worst, x_lo, x_hi):
            v, t = w["v"], w["t"]
            label = (f"<b>{ae_neg(f'{v:.1%}')}</b><br>"
                     f"<span style='font-size:11px;color:{AE_AXIS}'>{t:%b} {t.day}, {t.year}</span>")
            fig.add_trace(go.Scatter(
                x=[t], y=[v], mode="markers", legendgroup=w["name"], showlegend=False, hoverinfo="skip",
                marker=dict(size=10, color=w["color"], line=dict(color="#ffffff", width=2)), visible=(m == modes[0]),
            ))
            tags.append(m)
            fig.add_trace(go.Scatter(
                x=[t], y=[w["ly"]], mode="markers+text", legendgroup=w["name"], showlegend=False, hoverinfo="skip",
                marker=dict(size=14, color="rgba(255,255,255,0)"), text=[label],
                textposition="middle left" if w["frac"] > 0.86 else "middle right",
                textfont=dict(size=12.5, color=w["text"]), cliponaxis=False, visible=(m == modes[0]),
            ))
            tags.append(m)
    ae_layout(fig, height=380, yfmt=".0%", xhover="%b %d, %Y" if modes[0] == "daily" else "%b %Y",
              top=46 if len(modes) > 1 else 12)
    fig.update_yaxes(zeroline=True, zerolinecolor=AE_CONTROL, zerolinewidth=1)
    fig.update_layout(legend=dict(itemwidth=44))
    if len(modes) > 1:
        args = [[{"visible": [t == m for t in tags]}, {"xaxis.hoverformat": "%b %d, %Y" if m == "daily" else "%b %Y"}]
                for m in modes]
        fig.update_layout(updatemenus=ae_buttons(["Daily", "Month-end"], args))
    return fig


# ---- Drawdowns in context (moved here from the Chart Dashboard, November 21, 2026) ----
# Every drawdown of 10% or more in the backtest just run, as a ladder of bars, with each portfolio's worst drawdown
# on the same scale beneath (the strategy, the 80/20 and the S&P 500 over the run's period). Tyler, November 21,
# 2026: it belongs here, built from the run; it left the public Chart Dashboard the same day.
# The shaded zone is a fixed study, not this run: the worst daily drawdowns of the standing settings (2/8 main
# lookback, top 4, 3 of 4 equities on 1/3, 12-month window, with IBIT, net of 0.05% a side) and their eight nearby
# settings (short leg 1 to 3, long leg 7 to 9) from the start of the file's band window (band_window) through its
# backtest_end, under the run's position sizing, shaded to at least 20%. When the band window starts before
# DDC_SINCE_WINDOW, that window's range is printed beside it, and a window's optional band_note_chart and
# band_note_text are printed when the file has them (ddc_band; November 25, 2026). It comes from
# drawdown_context.json beside this file (first built by Basket Research\ETF Scouting\Drawdown Context 1981
# 2026-11-21\Build Drawdown Context.py, rebuilt since from dated copies of its scripts) and is left off with
# leverage on, since the study is unlevered.
# When the Chart Dashboard's published series sits beside this folder (..\Portfolio Hist Val Reconstruction\
# asymmetric_edge_series.csv, on Tyler's PC), a "Live account now" bar adds the live account's decline from its
# high; the deployed copy has no such file and draws without it. Drawdowns are measured on exchange days only: the
# daily curve also carries the price table's weekend and holiday rows, which would put a peak on a Sunday.
DDC_FILE = os.path.join(_base_dir, "drawdown_context.json")
DDC_LIVE_FILE = os.path.join(os.path.dirname(_base_dir), "Portfolio Hist Val Reconstruction", "asymmetric_edge_series.csv")
DDC_MIN = 0.10            # drawdowns this deep or deeper get a bar
DDC_MAX_ROWS = 16         # at most this many bars, the deepest
DDC_BAND_TO = 20.0        # the shading is drawn to at least this depth, in percent (Tyler, November 21, 2026)
DDC_ROW = 44              # px per bar row
DDC_LEFT = 232            # px for the row labels
DDC_BACKTEST = "#B3ACD8"  # this backtest's drawdowns: a light tint of the strategy's indigo
DDC_BAND = "#F1EFF8"
DDC_BAND_EDGE = "#C9C3E3"
DDC_BENCH = [("80/20", "AOA"), ("S&P 500", "SPY")]
DDC_SIZING_WORDS = {"erc": "equal risk contribution", "inverse_vol": "inverse volatility"}
DDC_SINCE_WINDOW = "1981-01"  # printed beside a band window that starts before it (November 25, 2026)


def ddc_episodes(e):
    """A value series' drawdowns as (peak, trough, recovered or None, depth), as the drawdown studies count them:
    a drawdown opens on the first close below the running high, its peak is the close before, and it ends on the
    first close back at or above that high."""
    dd = e / e.cummax() - 1
    out, inep, prev = [], False, None
    peak = trough = None
    depth = 0.0
    for t, v in dd.items():
        if v < 0 and not inep:
            inep, peak, trough, depth = True, (prev if prev is not None else t), t, v
        elif inep:
            if v < depth:
                trough, depth = t, v
            if v >= 0:
                out.append((peak, trough, t, float(depth)))
                inep = False
        prev = t
    if inep:
        out.append((peak, trough, None, float(depth)))
    return out


def ddc_day(d):
    d = pd.Timestamp(d)
    return f"{d:%b} {d.day}, {d.year}"


def ddc_months(a, b):
    """Months from a to b, half up."""
    return int(np.floor((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4375 + 0.5))


def ddc_years(p, t):
    p, t = pd.Timestamp(p).year, pd.Timestamp(t).year
    return str(p) if p == t else f"{p}–{str(t)[-2:]}"


def ddc_row_label(primary, secondary, live=False):
    color = BRAND_PURPLE if live else AE_INK
    p = f"<b>{ae_esc(primary)}</b>" if live else ae_esc(primary)
    return (f"<span style='color:{color}'>{p}</span><br>"
            f"<span style='font-size:12px;color:{AE_AXIS}'>{ae_esc(secondary)}</span>")


def ddc_band(sizing, leverage_on):
    """The fixed study's shaded range for this position sizing: (dict(lo, hi, text), note), or (None, note).
    Everything that can change is read from drawdown_context.json: the band window (band_window), its first
    month, the study's last month (backtest_end), this sizing's range, the DDC_SINCE_WINDOW window's range when
    the band window starts before it, and the band window's optional words on its episode (band_note_chart on
    the chart, band_note_text under the heading), printed only when the file has them. Wording of November 25,
    2026 (Tyler, 9:23 p.m.)."""
    if leverage_on:
        return None, "The shaded zone is left off with leverage on: the study behind it is unlevered."
    try:
        with open(DDC_FILE, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except Exception:
        return None, ""
    words = DDC_SIZING_WORDS.get(sizing, sizing)
    band_window = data.get("band_window")
    if not band_window:
        return None, "The shaded zone is not drawn: drawdown_context.json names no band window."
    try:
        windows = data["methods"][sizing]["windows"]
    except Exception:
        return None, f"The shaded zone is not drawn: drawdown_context.json has no windows for {words}."
    if band_window not in windows:
        return None, f"The shaded zone is not drawn: drawdown_context.json has no {band_window} window for {words}."
    try:
        win = windows[band_window]
        vals = [float(v) * 100.0 for v in win["band"]]
        start = pd.Timestamp(win["start"])
        end = pd.Timestamp(data["backtest_end"])
    except Exception:
        return None, ""
    since = None
    if DDC_SINCE_WINDOW in windows and start < pd.Period(DDC_SINCE_WINDOW, "M").to_timestamp():
        try:
            s_vals = [float(v) * 100.0 for v in windows[DDC_SINCE_WINDOW]["band"]]
            since = (pd.Timestamp(windows[DDC_SINCE_WINDOW]["start"]), -max(s_vals), -min(s_vals))
        except Exception:
            since = None

    def pct(x):
        return ae_neg(f"-{x:.1f}%")

    lo, hi = -max(vals), -min(vals)
    to_t = ae_neg(f"-{DDC_BAND_TO:.0f}%")
    note_chart = str(win.get("band_note_chart") or "").strip()
    note_text = str(win.get("band_note_text") or "").strip()
    text = (f"<b>Shaded:</b> the worst daily drawdowns<br>of the standing settings and 8 nearby<br>"
            f"in backtests, {start:%b %Y} to {end:%b %Y}:<br>{pct(lo)} to {pct(hi)}"
            + (f", shown to {to_t}" if hi < DDC_BAND_TO else "")
            + (f", {ae_esc(note_chart)}" if note_chart else "") + "."
            + (f"<br>Since {since[0]:%Y}: {pct(since[1])} to {pct(since[2])}." if since else "")
            + "<br>Context, not a limit on losses.")
    note = (f"Shaded: the deepest fall from a high to a later low, on daily closes, in backtests of the standing "
            f"settings (2/8) and their eight nearby settings from {start:%B %Y} through {end:%B %Y}, with Bitcoin, "
            f"under {words}: {pct(lo)} to {pct(hi)}"
            + (f", shaded to {to_t}" if hi < DDC_BAND_TO else "") + "."
            + (f" {note_text}" if note_text else "")
            + (f" Counting only falls that began in {since[0]:%Y} or later, the range is {pct(since[1])} to "
               f"{pct(since[2])}." if since else "")
            + " A fixed study that does not change with this run's settings. For the years before a fund existed, "
              "its returns come from a stand-in (an index, an older fund or a market price), and the stand-ins are "
              "weakest before 1980. The shading is context from the past, not a limit on future losses.")
    return dict(lo=lo, hi=max(hi, DDC_BAND_TO), text=text), note


def ddc_live():
    """The live account's decline from its high, from the Chart Dashboard's published series when it sits beside
    this folder: (depth in %, peak date, last date), or None (as on the deployed copy)."""
    try:
        s = pd.read_csv(DDC_LIVE_FILE, index_col=0, parse_dates=True)["value"].dropna()
    except Exception:
        return None
    if len(s) < 2:
        return None
    top = s.cummax().iloc[-1]
    return float((s.iloc[-1] / top - 1.0) * 100.0), s[s >= top].index[-1], s.index[-1]


def ae_fig_drawdown_context(daily_equity, balanced_data, spy_daily_equity, cal, period_start, lev_tag="",
                            band=None, band_note="", live=None):
    """Drawdowns in context for the run just made: every drawdown of 10% or more (the DDC_MAX_ROWS deepest) with
    the live account's decline when known, over each portfolio's worst on one scale, and the fixed study's band.
    Returns (figure, the heading's sub-line as HTML)."""
    from plotly.subplots import make_subplots

    def on_cal(s):
        s = s.dropna()
        s = s[s.index >= period_start]
        return s[s.index.isin(cal)] if cal is not None and len(cal) else s

    e = on_cal(daily_equity)
    start, end = e.index[0], e.index[-1]
    eps = ddc_episodes(e)
    big = sorted([x for x in eps if x[3] <= -DDC_MIN], key=lambda x: x[3])

    def rec_text(p, r):
        return f"back to peak in {ddc_months(p, r)} months" if r is not None else f"not back to peak by {ddc_day(end)}"

    def hover(name, p, t, r, d, what):
        back = (f"Back to peak {ddc_day(r)}, {ddc_months(p, r)} months" if r is not None
                else f"Not back to peak by {ddc_day(end)}")
        return (f"<b>{ae_esc(name)}</b> ({what})<br>Peak {ddc_day(p)}, low {ddc_day(t)}<br>{back}<br>"
                f"Depth {ae_neg(f'{d:.2%}')}")

    top = [dict(mag=-d * 100.0, color=DDC_BACKTEST, text=ae_neg(f"{d:.1%}"),
                label=ddc_row_label(ddc_years(p, t), rec_text(p, r)), hover=hover(ddc_years(p, t), p, t, r, d, "backtest"))
           for p, t, r, d in big[:DDC_MAX_ROWS]]
    if live is not None:
        depth, peak, last = live
        mag = max(0.0, -depth)
        top.append(dict(mag=mag, color=BRAND_PURPLE, now=True, text=f"<b>{ae_neg(f'-{mag:.1f}%')}</b>",
                        label=ddc_row_label("Live account now", "at a new high" if mag < 0.005
                                            else f"from the {ddc_day(peak)} high", live=True),
                        hover=(f"<b>Live account now</b> (Fidelity reconstruction)<br>High {ddc_day(peak)}<br>"
                               f"Close {ddc_day(last)}<br>Depth {ae_neg(f'-{mag:.2f}%')}")))
    if not top:
        top.append(dict(mag=0.0, color=DDC_BACKTEST, text="", label=ddc_row_label("None", "no drawdown of 10% or more"),
                        hover="No drawdown of 10% or more"))
    top.sort(key=lambda r: r["mag"])

    bottom = []
    if eps:
        p, t, r, d = min(eps, key=lambda x: x[3])
        nm = "Asymmetric Edge (backtest)" + lev_tag
        bottom.append(dict(mag=-d * 100.0, color=DDC_BACKTEST, text=ae_neg(f"{d:.1%}"),
                           label=ddc_row_label(nm, f"{ddc_years(p, t)}, {rec_text(p, r)}"), hover=hover(nm, p, t, r, d, "backtest")))
    for key, ticker in DDC_BENCH:
        series = spy_daily_equity if ticker == "SPY" else next(
            (b.get("daily_equity") for b in balanced_data if b.get("ticker") == ticker), None)
        if series is None or not len(series):
            continue
        s = on_cal(series)
        s = s[s.index <= end]
        b_eps = ddc_episodes(s) if len(s) > 1 else []
        if not b_eps:
            continue
        p, t, r, d = min(b_eps, key=lambda x: x[3])
        nm = AE_BENCH_NAMES.get(ticker, ticker) + lev_tag
        bottom.append(dict(mag=-d * 100.0, color=AE_DD_LOOK[key][0], text=ae_neg(f"{d:.1%}"),
                           label=ddc_row_label(nm, f"{ddc_years(p, t)}, {rec_text(p, r)}"),
                           hover=hover(nm, p, t, r, d, "total return")))

    n1, n2 = len(top), max(len(bottom), 1)
    M_T, GAP, M_B, M_R = 58, 76, 96, 28
    plot_h = (n1 + n2) * DDC_ROW + GAP
    height = M_T + plot_h + M_B
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[n1, n2], vertical_spacing=GAP / plot_h)
    for row, rows in ((1, top), (2, bottom)):
        ys = [f"r{row}_{i}" for i in range(len(rows))]
        fig.add_trace(go.Bar(
            x=[r["mag"] for r in rows], y=ys, orientation="h", width=0.58,
            marker=dict(color=[r["color"] for r in rows], cornerradius=6),
            text=[r["text"] for r in rows], textposition="outside", cliponaxis=False,
            textfont=dict(size=14, color=[BRAND_PURPLE if r.get("now") else AE_INK for r in rows]),
            hovertext=[r["hover"] for r in rows], hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        ), row=row, col=1)
        fig.update_yaxes(tickmode="array", tickvals=ys, ticktext=[r["label"] for r in rows], autorange="reversed",
                         showgrid=False, ticks="", ticklabelstandoff=12, tickfont=dict(size=15, color=AE_INK),
                         fixedrange=True, row=row, col=1)

    deepest = max([r["mag"] for r in top + bottom] + ([band["hi"]] if band else []))
    step = 10 if deepest > 25 else 5
    xmax = int(np.ceil((deepest + 2.0) / step)) * step
    if band:
        # Leave room right of the band for its label.
        xmax = max(xmax, int(np.ceil(band["hi"] / 0.54 / step)) * step)
    ticks = list(range(0, xmax + 1, step))
    for row in (1, 2):
        fig.update_xaxes(range=[0, xmax], tickvals=ticks, showgrid=True, gridcolor=AE_PLOT_GRID, zeroline=True,
                         zerolinecolor=AE_CONTROL, zerolinewidth=1.5, fixedrange=True, row=row, col=1)
    fig.update_xaxes(ticktext=[ae_neg(f"-{t}%") if t else "0%" for t in ticks], tickfont=dict(size=13, color=AE_AXIS),
                     title=dict(text="Decline from peak", font=dict(size=13, color=AE_BODY), standoff=8), row=2, col=1)
    if band:
        lo, hi = band["lo"], band["hi"]
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=lo, x1=hi, y0=0, y1=1,
                      fillcolor=DDC_BAND, line=dict(width=0), layer="below")
        for xv in (lo, hi):
            fig.add_shape(type="line", xref="x", yref="y domain", x0=xv, x1=xv, y0=0, y1=1,
                          line=dict(color=DDC_BAND_EDGE, width=1, dash="3px,3px"), layer="below")
        fig.add_annotation(x=hi, y="r1_0", xref="x", yref="y", xanchor="left", xshift=10, yanchor="middle",
                           align="left", showarrow=False, text=band["text"],
                           font=dict(size=12, color=AE_BODY, family=AE_SANS))

    title_top = (f"Every drawdown of 10% or more, {start:%b %Y} to {end:%b %Y}"
                 + (", and the live account now" if live is not None else ""))
    title_bottom = "Worst drawdown over the same period: the strategy and the market"
    for y, text in ((fig.layout.yaxis.domain[1], title_top), (fig.layout.yaxis2.domain[1], title_bottom)):
        fig.add_annotation(x=0, y=y, xref="paper", yref="paper", xanchor="left", yanchor="bottom", xshift=-(DDC_LEFT - 4),
                           yshift=14, showarrow=False, text=f"<b>{ae_esc(text)}</b>",
                           font=dict(size=15, color=AE_INK, family=AE_SANS))
    keys = [("This backtest", DDC_BACKTEST)] + ([("Live account", BRAND_PURPLE)] if live is not None else [])
    for name, color in keys:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name + " ",
                                 marker=dict(symbol="square", size=14, color=color, line=dict(color=color, width=1)),
                                 hoverinfo="skip"), row=1, col=1)
    fig.update_layout(
        height=height, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=AE_SANS, size=13, color=AE_AXIS),
        margin=dict(t=M_T, b=M_B, l=DDC_LEFT, r=M_R), hovermode="closest",
        legend=dict(orientation="h", xref="container", yref="container", x=0.5, xanchor="center",
                    y=14 / height, yanchor="bottom", font=dict(size=13, color=AE_BODY),
                    itemsizing="constant", itemclick=False, itemdoubleclick=False),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AE_CARD, font=dict(family=AE_SANS, size=13, color=AE_INK)),
    )

    parts = []
    if big:
        p, t, r, d = big[0]
        n = len(big)
        parts.append(f"{n} drawdown{'s' if n != 1 else ''} of 10% or more from {start:%B %Y} to {end:%B %Y}"
                     + (f" (the {DDC_MAX_ROWS} deepest shown)" if n > DDC_MAX_ROWS else "")
                     + f". The deepest, {ae_neg(f'{d:.1%}')} in {ddc_years(p, t)}, "
                     + (f"took {ddc_months(p, r)} months to get back to its peak." if r is not None
                        else "had not recovered by the end of the backtest."))
    else:
        parts.append(f"No drawdown of 10% or more from {start:%B %Y} to {end:%B %Y}.")
    if live is not None and big:
        mag = max(0.0, -live[0])
        deeper = sum(1 for x in big if -x[3] * 100.0 > mag)
        parts.append(f"The live account is {mag:.1f}% below its {ddc_day(live[1])} high (Fidelity reconstruction, "
                     f"through {ddc_day(live[2])}); {deeper} of this backtest's {len(big)} went deeper.")
    if band_note:
        parts.append(band_note)
    return fig, ae_esc(" ".join(parts))


def ae_yearly(r):
    r = r.dropna()
    return (1 + r).groupby(r.index.year).prod() - 1


def ae_fig_annual(strat_returns, bench_returns, balanced_data, lev_tag=""):
    """Calendar-year returns: the strategy, the 80/20, the S&P 500 (60/40 one legend click away)."""
    sets = ae_series_set(strat_returns, bench_returns, balanced_data, lev_tag)
    years = ae_yearly(strat_returns).index
    fig = go.Figure()
    for name, r, color, width, dash in sets:
        y = ae_yearly(r).reindex(years)
        fig.add_trace(go.Bar(
            x=[str(v) for v in years], y=y.values, name=ae_esc(name), marker=dict(color=color),
            hovertemplate=f"{ae_esc(name)}: %{{y:.1%}}<extra></extra>",
            visible="legendonly" if name.startswith("60/40") else True,
        ))
    ae_layout(fig, height=380, yfmt=".0%", xhover=None)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    fig.update_xaxes(type="category", showspikes=False, tickangle=0, nticks=16)
    fig.update_yaxes(zeroline=True, zerolinecolor=AE_CONTROL, zerolinewidth=1)
    return fig


def ae_fig_lines(lines, height=300, yfmt=None, ref=None, yrange=None, fills=None, tickfmt=None):
    """A small rolling chart. lines: (name, series, color, width, dash); fills: (series, color) areas."""
    fig = go.Figure()
    for s, color in (fills or []):
        s = s.dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor=color, hoverinfo="skip", showlegend=False))
    for name, s, color, width, dash in lines:
        s = s.dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=ae_esc(name),
                                 line=dict(color=color, width=width, dash=dash),
                                 hovertemplate=f"{ae_esc(name)}: %{{y:{yfmt or '.2f'}}}<extra></extra>"))
    ae_layout(fig, height=height, yfmt=tickfmt or yfmt, yrange=yrange)
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color="#9CA3AF", width=1, dash="dot"))
    fig.update_layout(legend=dict(y=-0.16, font=dict(size=11.5)))
    return fig


def ae_fig_underwater(uw_s, uw_b, lev_tag=""):
    fig = go.Figure()
    spy = AE_BENCH_NAMES.get(BENCHMARK_TICKER, BENCHMARK_TICKER) + lev_tag
    fig.add_trace(go.Bar(x=uw_b.index, y=uw_b.values, name=ae_esc(spy), marker=dict(color=BRAND_INDIGO, opacity=0.55),
                         hovertemplate=f"{ae_esc(spy)}: %{{y}} months<extra></extra>"))
    fig.add_trace(go.Bar(x=uw_s.index, y=uw_s.values, name="Strategy", marker=dict(color=BRAND_PURPLE, opacity=0.9),
                         hovertemplate="Strategy: %{y} months<extra></extra>"))
    ae_layout(fig, height=300)
    fig.update_layout(barmode="overlay", bargap=0, legend=dict(y=-0.16, font=dict(size=11.5)))
    return fig


def ae_fig_leverage(trade_df, leverage_params):
    """The leverage multiplier each month, colored by regime (when dynamic leverage is on)."""
    if trade_df.empty or "leverage" not in trade_df.columns or not leverage_params.get("enabled", False):
        return None, ""
    if leverage_params.get("method") == "constant":
        return None, ""
    df = trade_df.copy()
    df["date"] = pd.to_datetime(df["hold_month"])
    df["lev_val"] = df["leverage"].str.replace("x", "").astype(float)
    if leverage_params.get("method", "holdings_gate") == "holdings_gate":
        title = f"Dynamic leverage (holdings gate: {', '.join(leverage_params.get('defensive_tickers', ['BIL', 'BTAL']))})"
        groups = [("leveraged", "Leveraged (no defensives)", BRAND_TEAL)]
        other = ("Neutral (defensives held)", BRAND_GRAY)
    else:
        title = (f"Dynamic leverage ({leverage_params.get('ticker', 'VT')} {leverage_params.get('long_sma', 200)}/"
                 f"{leverage_params.get('short_sma', 50)}-day SMA trend)")
        groups = [("risk-on", "Risk-on", BRAND_TEAL), ("risk-off", "Risk-off", BRAND_RED)]
        other = ("Neutral", BRAND_GRAY)
    fig = go.Figure()
    known = [g[0] for g in groups]
    for key, label, color in groups:
        d = df[df["lev_regime"] == key]
        fig.add_trace(go.Bar(x=d["date"], y=d["lev_val"], name=label, marker=dict(color=color, opacity=0.75),
                             hovertemplate=f"{label}: %{{y:.2f}}x<extra></extra>"))
    d = df[~df["lev_regime"].isin(known)]
    fig.add_trace(go.Bar(x=d["date"], y=d["lev_val"], name=other[0], marker=dict(color=other[1], opacity=0.75),
                         hovertemplate=f"{other[0]}: %{{y:.2f}}x<extra></extra>"))
    ae_layout(fig, height=280, yrange=[0, max(df["lev_val"].max() * 1.15, 1.5)])
    fig.update_layout(barmode="overlay", bargap=0.1)
    fig.add_hline(y=1.0, line=dict(color="#9CA3AF", width=1, dash="dot"))
    return fig, title


def ae_holding_chips(text):
    """'HGER: 41.9% (ret: +0.91%) | ...' from the trade log, as chips."""
    out = []
    for part in str(text).split(" | "):
        m = re.match(r"\s*(\S+): ([\d.]+%) \(ret: ([+\-]?[\d.]+%)\)", part)
        if not m:
            if part.strip():
                out.append(f'<span class="ae-chip">{ae_esc(part.strip())}</span>')
            continue
        t, w, r = m.groups()
        col = AE_GAIN if not r.startswith("-") else AE_LOSS
        out.append(f'<span class="ae-chip"><span style="font-weight:700;color:{AE_INK}">{ae_esc(t)}</span>'
                   f'<span style="color:{AE_BODY}">{ae_esc(w)}</span>'
                   f'<span style="color:{col};font-weight:600">{ae_esc(ae_neg(r))}</span></span>')
    return '<div class="ae-row" style="gap:5px">' + "".join(out) + "</div>"


def ae_score_bar(v, maxabs):
    """A blended score with a small bar that grows right from zero (purple) or left (gray)."""
    if not isinstance(v, (int, float, np.floating)) or not np.isfinite(v):
        return "-"
    w = min(abs(v) / maxabs, 1.0) * 50 if maxabs > 0 else 0
    left = 50 if v >= 0 else 50 - w
    col = BRAND_PURPLE if v >= 0 else "#C9C3B8"
    return (f'<span class="ae-score"><span style="font-weight:600;color:{AE_INK}">{ae_esc(ae_neg(f"{v:.2%}"))}</span>'
            f'<span class="ae-score-b" aria-hidden="true"><span style="left:{left:.1f}%;width:{w:.1f}%;background:{col}"></span><i></i></span></span>')


# =============================================================================
# REALITY CHECK  (added November 16, 2026)
# =============================================================================
# A results section of its own, below the Performance Summary. It repeats the
# backtest on the settings one step away from the ones in the sidebar and says
# how much of the tested result survives: the main lookback legs (each leg one
# month either way), the trade day (params["signal_offset"] in run_backtest),
# position sizing, the volatility window, the holdings count and the equity
# lookback. Every number it shows is a backtest from run_backtest itself, with
# the run's own prices, backfills, costs, leverage and G1 settings.
#
#   Neighbors' average   the lookbacks one step away on each leg, on the trade
#                        day and the days either side of it (up to 27 runs)
#   Typical nearby run   the median of the same lookbacks on 21 trade days, the
#                        19th-to-last trading day of the month to the 2nd
#                        trading day of next month (up to 189 runs; the Full
#                        check)
#   Realistic range      the span between those two
#
# The panel names a run by its trade day (rc_trade_noun), the day the month-end
# trade is made, and any of the 21 can be picked in the sidebar (November 17,
# 2026). Trade days other than the sidebar's original two run on the engine's own
# calendar: earlier months at their month-end closes, the current month at the
# signal close. A second basket can be run beside the run's own (Basket B, below;
# opt-in). Finished runs are saved in reality_check_cache\ under the settings and
# a fingerprint of the prices, so a backtest that has run once is not run
# again until the prices behind it change.
#
# The mockup Tyler approved, and the script that pulled its numbers:
# C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System\Backtest Engine\Robustness Panel 2026-11-16\

import hashlib as _rc_hashlib
import html as _rc_html
import pickle as _rc_pickle
import time as _rc_time

RC_CACHE_DIR = os.path.join(_base_dir, "reality_check_cache")
RC_CACHE_FILE = os.path.join(RC_CACHE_DIR, "reality_check_runs.pkl")
RC_CACHE_MAX = 4000                         # saved runs kept (the oldest go first)
RC_STRIP_DAYS = [5, 4, 3, 2, 1, 0, -1]      # the trade days charted, 5th to last ... 2nd of next (signal days T-5 to T+1)
RC_WIDE_DAYS = list(range(19, -2, -1))      # the Full check's 21 trade days (signal days T-19 to T+1)
RC_MILD = 0.05                              # a setting more than this above its neighbors is a mild peak
RC_SPIKE = 0.15                             # and more than this, a spike
RC_MODE_OPTIONS = {"Off": "off", "Quick check": "quick", "Full check": "full"}
# On a PC the neighbor runs go to a pool of worker processes (reality_check_worker.py),
# so the Quick check is the starting choice there. Anywhere else (the deployed
# app, which has one small processor) they would run one at a time, so the
# section starts off and is switched on by hand.
RC_ON_PC = os.name == "nt"
RC_MODE_DEFAULT = "quick" if RC_ON_PC else "off"
RC_ENGINE_FILE = os.path.abspath(__file__) if "__file__" in dir() else None
RC_POOL_MIN = 3                             # fewer new runs than this are not worth a pool
RC_POOL_MAX = 12                            # worker processes at most
RC_SIZING_NAMES = {"erc": "Equal risk contribution", "inverse_vol": "Inverse volatility"}
# Inverse volatility was the sizing method until November 2026, and the app
# called it risk parity then; the panel says so wherever it names it.
RC_SIZING_NOTES = {"inverse_vol": "the method used until November 2026, previously called risk parity"}
RC_TONES = {"good": ("#DDF3E4", "#0F6B3A"), "warn": ("#FFF0C7", "#7A5300"),
            "bad": ("#FBDDDD", "#A32222"), "neutral": ("#EEEEF2", "#55556A")}
RC_GLYPHS = {"good": "\u2713", "warn": "!", "bad": "\u2715", "neutral": "\u2013", "up": "\u2191", "down": "\u2193"}


# ---- settings one step away -------------------------------------------------

def rc_day_name(k):
    """Signal day k trading days before the month's last trading day T."""
    return f"T-{k}" if k > 0 else ("T" if k == 0 else f"T+{-k}")


def rc_current_offset(params):
    """The run's own signal day: 1 for "last_day" (signal T-1), 0 for "first_day" (signal T)."""
    k = params.get("signal_offset")
    if k is not None:
        return int(k)
    return 0 if params.get("exec_timing", "last_day") == "first_day" else 1


# ---- trade days (added November 17, 2026) -----------------------------------
# The sidebar and the panel name a run by the day the month-end trade is made,
# its trade day. Inside, a run is still keyed by k = params["signal_offset"]:
# the signal is read at the close k trading days before the month's last
# trading day T, and the trade is made at the next close. So k >= 1 trades on
# the k-th trading day from the end of the month (1 is the last trading day)
# and k <= 0 on the (1 - k)th trading day of the next month (0 is the first).

def rc_ordinal(n):
    """1 -> "1st", 2 -> "2nd", 11 -> "11th", 21 -> "21st"."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rc_cap(s):
    """The text with its first letter capitalized."""
    return s[:1].upper() + s[1:]


def rc_trade_noun(k):
    """Signal day k as its trade day: "last trading day", "5th-to-last trading day",
    "first trading day of next month", "2nd trading day of next month"."""
    k = int(k)
    if k == 1:
        return "last trading day"
    if k > 1:
        return f"{rc_ordinal(k)}-to-last trading day"
    if k == 0:
        return "first trading day of next month"
    return f"{rc_ordinal(1 - k)} trading day of next month"


def rc_trade_phrase(k):
    """The trade day in a sentence: "the 5th-to-last trading day"."""
    return "the " + rc_trade_noun(k)


def rc_trade_short(k):
    """The label under a chart bar, which the panel wraps onto two short lines."""
    k = int(k)
    if k == 1:
        return "Last day"
    if k > 1:
        return f"{rc_ordinal(k)} to last"
    return "1st of next" if k == 0 else f"{rc_ordinal(1 - k)} of next"


def rc_exec_label(k):
    """The sidebar's name for a trade day. The first two are the app's original labels, word for word."""
    k = int(k)
    if k == 1:
        return "Last trading day of month (at close)"
    if k == 0:
        return "First trading day of next month (at close)"
    return rc_cap(rc_trade_noun(k)) + (" of month" if k > 1 else "") + " (at close)"


def rc_trade_each(k):
    """The trade day for the results footnote: "5th-to-last trading day of each month"."""
    return rc_trade_noun(k) + (" of each month" if int(k) > 1 else "")


def rc_strip_days(k0):
    """The seven signal days the Trade timing card charts: T-5 to T+1 (5th to last ... 2nd of next), as
    always, when the run's own day is among them; otherwise seven days centered on it, kept inside the 21."""
    if k0 in RC_STRIP_DAYS:
        return list(RC_STRIP_DAYS)
    top = max(min(RC_WIDE_DAYS[0], k0 + 3), RC_WIDE_DAYS[-1] + 6)
    return [top - j for j in range(7)]


def rc_lb_name(lbs):
    return "/".join(str(int(x)) for x in lbs)


def rc_lookback_cells(lbs):
    """The lookback and the ones a month away on each leg (one or two legs)."""
    lbs = [int(x) for x in lbs]
    if len(lbs) == 1:
        return [(x,) for x in (lbs[0] - 1, lbs[0], lbs[0] + 1) if x >= 1]
    if len(lbs) == 2:
        s, l = lbs
        return [(a, b) for a in (s - 1, s, s + 1) for b in (l - 1, l, l + 1) if 1 <= a < b]
    return [tuple(lbs)]


def rc_spec(params, lb=None, k=None, sizing=None, vol=None, topn=None, eq=None):
    """One run, as the six settings the panel varies."""
    return (tuple(int(x) for x in (params["main_lookbacks"] if lb is None else lb)),
            int(rc_current_offset(params) if k is None else k),
            sizing or params.get("sizing", SIZING_IF_UNSET),
            int(params["vol_months"] if vol is None else vol),
            int(params["final_top_n"] if topn is None else topn),
            tuple(int(x) for x in (params["equity_lookbacks"] if eq is None else eq)))


def rc_spec_params(params, spec):
    """The run's params with one spec's settings in place (leg weights unchanged)."""
    lb, k, sizing, vol, topn, eq = spec
    p = dict(params)
    p.update(main_lookbacks=list(lb), signal_offset=int(k), sizing=sizing, vol_months=int(vol),
             final_top_n=int(topn), equity_lookbacks=list(eq), fast_daily=True)
    return p


def rc_plan(params, extras=()):
    """Every run the panel reads, by group. extras: "topn" adds the holdings count, "eq" the equity lookback."""
    k0 = rc_current_offset(params)
    cells = rc_lookback_cells(params["main_lookbacks"])
    near_days = [d for d in (k0 + 1, k0, k0 - 1) if RC_WIDE_DAYS[-1] <= d <= RC_WIDE_DAYS[0]]
    sizing0 = params.get("sizing", SIZING_IF_UNSET)
    v0 = int(params["vol_months"])
    n0 = int(params["final_top_n"])
    plan = {
        "center": rc_spec(params),
        "cells": [rc_spec(params, lb=c) for c in cells],
        "block": [rc_spec(params, lb=c, k=d) for c in cells for d in near_days],
        "strip": [rc_spec(params, k=d) for d in rc_strip_days(k0)],
        "sizing": {s: [rc_spec(params, lb=c, sizing=s) for c in cells] for s in SIZING_LABELS if s != sizing0},
        "vol": {v: [rc_spec(params, lb=c, vol=v) for c in cells]
                for v in sorted({min(24, max(3, x)) for x in (v0 - 6, v0 + 6, v0 + 12)} - {v0})},
        "topn": ({n: rc_spec(params, topn=n) for n in (n0 - 1, n0 + 1) if n >= 1} if "topn" in extras else {}),
        "eq": {},
        "wide": [rc_spec(params, lb=c, k=d) for c in cells for d in RC_WIDE_DAYS],
    }
    eq0 = tuple(int(x) for x in params.get("equity_lookbacks", []))
    if ("eq" in extras and params.get("equity_top_n", 0) > 0 and len(params.get("equity_tickers", [])) > 0
            and 1 <= len(eq0) <= 2):
        for c in rc_lookback_cells(eq0):
            if sum(1 for x, y in zip(c, eq0) if x != y) == 1:
                plan["eq"][c] = rc_spec(params, eq=c)
    return plan


def rc_wanted(plan, mode):
    """The runs a check needs, in running order, without repeats."""
    specs = [plan["center"]] + plan["block"] + plan["strip"] + plan["cells"]
    for group in ("sizing", "vol"):
        for lst in plan[group].values():
            specs += lst
    specs += list(plan["topn"].values()) + list(plan["eq"].values())
    if mode == "full":
        specs += plan["wide"]
    seen, out = set(), []
    for s in specs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---- saved runs -------------------------------------------------------------

def _rc_plain(v):
    """A setting as text for the saved-run key (a price or yield series by its content)."""
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return "series:" + _rc_hashlib.sha1(pd.util.hash_pandas_object(v, index=True).values.tobytes()).hexdigest()[:16]
    if isinstance(v, dict):
        return {str(k): _rc_plain(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_rc_plain(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 12)
    if isinstance(v, (np.integer, int, bool, str)) or v is None:
        return v
    return str(v)


def rc_price_cut(daily_prices, params):
    """The last market day of the last finished month: the month before the one the prices end in, which is
    the last month run_backtest trades (or the backtest end, if that is earlier). It does not depend on the
    run's own signal day, so runs on different signal days share their saved neighbors."""
    dates = market_trading_dates(daily_prices)
    last = dates[-1]
    prior = dates[dates < pd.Timestamp(year=last.year, month=last.month, day=1)]
    cut = prior[-1] if len(prior) else last
    end = resolve_backtest_end(params.get("backtest_end", ""))
    if end:
        try:
            before = dates[dates <= pd.Timestamp(end)]
            if len(before) and before[-1] < cut:
                cut = before[-1]
        except (ValueError, TypeError):
            pass
    return cut


def rc_fingerprint(daily_prices, params, end_date):
    """The prices a run reads, by their daily returns through three market days after end_date.

    Returns, not prices, so a dividend rescale of a whole history leaves it
    unchanged; cut just after the last finished month (rc_price_cut), so new
    days inside an unfinished month leave it unchanged too. The three days
    after the cut are in it because signal days after the month-end trade
    on them; early in a month, fewer exist, and their count is part of it.
    """
    cols = set(params.get("fixed_tickers", [])) | set(params.get("equity_tickers", [])) | {"BIL", BENCHMARK_TICKER}
    # A run that reaches back before an equity seat's hand-off reads the stand-in too (EQUITY_SEAT_BEFORE),
    # so its saved runs from before the stand-in existed are not reused.
    _start = str(params.get("backtest_start", "") or "")
    cols |= {s for s, until in equity_seat_standins(params).values() if _start < until}
    g1 = params.get("g1") or {}
    if g1.get("enabled") and g1.get("fund"):
        cols.add(g1["fund"])
    lev = params.get("leverage_params") or {}
    if lev.get("ticker"):
        cols.add(lev["ticker"])
    cols = sorted(c for c in cols if c in daily_prices.columns)
    end_date = pd.Timestamp(end_date)
    dates = market_trading_dates(daily_prices)
    after = min(3, int((dates > end_date).sum()))
    later = dates[dates > end_date]
    hash_end = later[after - 1] if after > 0 else end_date
    rets = daily_prices[cols].loc[:hash_end].ffill().pct_change().round(8) + 0.0
    digest = _rc_hashlib.sha1(pd.util.hash_pandas_object(rets, index=True).values.tobytes()).hexdigest()[:20]
    return f"{digest}-{','.join(cols)}-{after}"


def rc_base_key(params, fingerprint):
    """Everything a neighbor run shares with the main run, as text."""
    base = {
        "fixed": list(params.get("fixed_tickers", [])),
        "equity": list(params.get("equity_tickers", [])),
        "equity_top_n": params.get("equity_top_n"),
        "equity_w": list(params.get("equity_lookback_weights", [])),
        "main_w": list(params.get("main_lookback_weights", [])),
        "cost": params.get("cost_per_side", DEFAULT_COST_PER_SIDE),
        "start": params.get("backtest_start"),
        "end": resolve_backtest_end(params.get("backtest_end", "")),
        "leverage": params.get("leverage_params") or {},
        "g1": params.get("g1") or {},
        "prices": fingerprint,
    }
    return json.dumps(_rc_plain(base), sort_keys=True)


def rc_key(base_key, spec):
    return _rc_hashlib.sha1((base_key + "|" + repr(spec)).encode("utf-8")).hexdigest()


def rc_cache_load():
    try:
        with open(RC_CACHE_FILE, "rb") as fh:
            cache = _rc_pickle.load(fh)
        return cache if isinstance(cache, dict) else {}
    except Exception:
        return {}


def rc_cache_save(cache):
    """Write the saved runs (oldest dropped past RC_CACHE_MAX). Never raises."""
    try:
        if len(cache) > RC_CACHE_MAX:
            keep = sorted(cache, key=lambda k: cache[k].get("t", 0.0))[-RC_CACHE_MAX:]
            for k in set(cache) - set(keep):
                del cache[k]
        os.makedirs(RC_CACHE_DIR, exist_ok=True)
        tmp = RC_CACHE_FILE + ".tmp"
        with open(tmp, "wb") as fh:
            _rc_pickle.dump(cache, fh, protocol=4)
        os.replace(tmp, RC_CACHE_FILE)
        return True
    except Exception:
        return False


# ---- running the neighbors --------------------------------------------------

def rc_pool_workers():
    """Worker processes to use: none off a PC, else the processor count less two, up to RC_POOL_MAX."""
    if not RC_ON_PC or RC_ENGINE_FILE is None:
        return 1
    return max(1, min(RC_POOL_MAX, (os.cpu_count() or 2) - 2))


def rc_run_parallel(daily_prices, monthly_prices, rf_monthly, jobs, workers, on_done=None):
    """Run (key, params) jobs in worker processes; returns {key: entry} for the ones that finished.

    Each worker loads this file's engine code once (reality_check_worker.py) and reads the run's prices from
    a temporary pickle, which is deleted afterward. Raises if the pool cannot start; the caller then runs
    the jobs one at a time.
    """
    import concurrent.futures
    import tempfile
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)
    import reality_check_worker as worker
    tmp = tempfile.NamedTemporaryFile(prefix="reality_check_", suffix=".pkl", delete=False)
    out = {}
    try:
        _rc_pickle.dump({"daily": daily_prices, "monthly": monthly_prices, "rf": rf_monthly}, tmp, protocol=4)
        tmp.close()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=worker.init, initargs=(RC_ENGINE_FILE, tmp.name)
        ) as pool:
            # Streamlit runs this page as the __main__ module, and a new worker process imports its
            # parent's __main__ by file name, which would run the whole page again in every worker.
            # The file name is hidden while the workers start (they start as the jobs are submitted).
            main_mod = sys.modules.get("__main__")
            hidden = main_mod is not None and "__file__" in getattr(main_mod, "__dict__", {})
            if hidden:
                main_file = main_mod.__dict__.pop("__file__")
            try:
                futures = [pool.submit(worker.run, job) for job in jobs]
            finally:
                if hidden:
                    main_mod.__dict__["__file__"] = main_file
            for fut in concurrent.futures.as_completed(futures):
                key, entry = fut.result()
                out[key] = entry
                if on_done is not None:
                    on_done(len(out))
    finally:
        try:
            tmp.close()
            os.remove(tmp.name)
        except OSError:
            pass
    return out


def rc_entry(results_df, daily_equity, rf_monthly):
    """What the panel keeps of one run: its monthly returns and three headline figures."""
    if results_df is None or len(results_df) == 0:
        return {"r": pd.Series(dtype=float), "S": np.nan, "C": np.nan, "D": np.nan, "t": _rc_time.time()}
    r = results_df["portfolio_return"].dropna().astype(float)
    m = compute_metrics(r, rf_monthly=rf_monthly, daily_equity=daily_equity)

    def num(key):
        v = m.get(key, np.nan)
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else np.nan

    return {"r": r, "S": num("Sortino Ratio"), "C": num("CAGR"), "D": num("Max Drawdown (Daily)"),
            "t": _rc_time.time()}


def rc_compute(daily_prices, monthly_prices, params, results_df, daily_equity, rf_monthly, mode,
               cache=None, progress=None, parallel=True, extras=()):
    """Run (or reuse) every backtest the check needs.

    Returns (plan, runs, stats): runs maps a spec to its entry; stats counts
    the runs needed, reused and run now, and the seconds the new ones took.
    The main run is the plan's center and is never run again. In a Quick
    check, Full-check runs already saved are read too, so the typical nearby
    run shows whenever it is known.
    """
    t0 = _rc_time.time()
    plan = rc_plan(params, extras)
    wanted = rc_wanted(plan, mode)
    base = rc_base_key(params, rc_fingerprint(daily_prices, params, rc_price_cut(daily_prices, params)))
    runs = {plan["center"]: rc_entry(results_df, daily_equity, rf_monthly)}
    if cache is not None:
        cache[rc_key(base, plan["center"])] = runs[plan["center"]]
    todo = []
    for spec in wanted:
        if spec in runs:
            continue
        key = rc_key(base, spec)
        if cache is not None and key in cache:
            runs[spec] = cache[key]
        else:
            todo.append((spec, key))
    pending, used_workers = list(todo), 1
    workers = min(rc_pool_workers(), len(pending)) if parallel else 1
    if workers >= 2 and len(pending) >= RC_POOL_MIN:
        try:
            got = rc_run_parallel(
                daily_prices, monthly_prices, rf_monthly,
                [(key, rc_spec_params(params, spec)) for spec, key in pending], workers,
                (lambda n: progress(n, len(todo))) if progress is not None else None,
            )
            used_workers = workers
        except Exception:
            got = {}
        for spec, key in pending:
            if key in got:
                runs[spec] = got[key]
                if cache is not None:
                    cache[key] = got[key]
        pending = [(spec, key) for spec, key in pending if key not in got]
    n_before = len(todo) - len(pending)
    for i, (spec, key) in enumerate(pending):
        if progress is not None:
            progress(n_before + i, len(todo))
        try:
            res, _trades, deq = run_backtest(daily_prices, monthly_prices, rc_spec_params(params, spec))
            runs[spec] = rc_entry(res, deq, rf_monthly)
        except Exception:
            runs[spec] = rc_entry(None, None, rf_monthly)
        if cache is not None:
            cache[key] = runs[spec]
    if mode != "full" and cache is not None:
        for spec in plan["wide"]:
            if spec not in runs:
                key = rc_key(base, spec)
                if key in cache:
                    runs[spec] = cache[key]
    stats = {"need": len(wanted), "fresh": len(todo), "reused": len(wanted) - len(todo),
             "secs": _rc_time.time() - t0, "base": base, "workers": used_workers}
    return plan, runs, stats


# ---- reading the runs -------------------------------------------------------

def rc_sortino(r, rf_monthly):
    """compute_metrics' Sortino on a slice of monthly returns."""
    r = r.dropna()
    n = len(r)
    if n == 0:
        return np.nan
    cagr = float((1.0 + r).prod()) ** (12.0 / n) - 1.0
    if rf_monthly is not None:
        rf = rf_monthly.reindex(r.index).fillna(0)
        excess = r - rf
        rf_annual = float((1.0 + rf).prod()) ** (12.0 / n) - 1.0
    else:
        excess, rf_annual = r, 0.0
    dd = float(np.sqrt(np.mean(np.minimum(excess.values, 0.0) ** 2)) * np.sqrt(12))
    return (cagr - rf_annual) / dd if dd > 0 else 0.0


def _rc_f2(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"


def _rc_f1(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.1f}"


def _rc_pct(x):
    """A fraction as a percent to one decimal, with a true minus sign."""
    if x is None or not np.isfinite(x):
        return "n/a"
    mag = f"{abs(x) * 100:.1f}"
    return ("\u2212" if x < 0 and float(mag) != 0 else "") + mag + "%"


def _rc_signed(x, digits=2, scale=1.0, unit=""):
    if x is None or not np.isfinite(x):
        return "n/a"
    mag = f"{abs(x * scale):.{digits}f}"
    return ("+" if (x >= 0 or float(mag) == 0) else "\u2212") + mag + unit


def rc_pill(tone, glyph, text):
    return {"tone": tone, "glyph": glyph, "text": text}


def rc_verdict(v, nb, mild=RC_MILD, spike=RC_SPIKE):
    """A setting against its neighbors' average: (level, pill)."""
    if not (np.isfinite(v) and np.isfinite(nb)) or nb == 0:
        return "na", rc_pill("neutral", "neutral", "Not enough runs")
    rel = (v - nb) / abs(nb)
    if rel > spike:
        return "bad", rc_pill("bad", "bad", "Spike")
    if rel > mild:
        return "warn", rc_pill("warn", "warn", "Mild peak")
    if rel < -mild:
        return "dip", rc_pill("neutral", "down", "Dip")
    return "good", rc_pill("good", "good", "Robust")


def _rc_level_rel(t, n, mild, spike):
    if not (np.isfinite(t) and np.isfinite(n)) or n == 0:
        return "na"
    r = (t - n) / abs(n)
    return "bad" if r > spike else ("warn" if r > mild else ("dip" if r < -mild else "good"))


def _rc_level_pts(d, hi, lo):
    if not np.isfinite(d):
        return "na"
    return "bad" if d > hi else ("warn" if d > lo else ("dip" if d < -lo else "good"))


def _rc_gap_pill(d, lvl, digits=2, scale=1.0, unit=""):
    if lvl == "na":
        return rc_pill("neutral", "neutral", "Not enough runs")
    if lvl == "bad":
        return rc_pill("bad", "bad", "Spike: " + _rc_signed(d, digits, scale, unit))
    if lvl == "warn":
        return rc_pill("warn", "warn", "Mild peak: " + _rc_signed(d, digits, scale, unit))
    if lvl == "dip":
        return rc_pill("good", "up", "Neighbors do better: " + _rc_signed(-d, digits, scale, unit))
    return rc_pill("good", "good", "Holds up: " + _rc_signed(d, digits, scale, unit))


def rc_sizing_name(s, with_note=False):
    name = RC_SIZING_NAMES.get(s, str(s))
    if with_note and s in RC_SIZING_NOTES:
        return f"{name} ({RC_SIZING_NOTES[s]})"
    return name


def rc_analyze(params, plan, runs, tested, rf_monthly=None, mild=RC_MILD, spike=RC_SPIKE):
    """Everything the panel shows, from the finished runs.

    tested = (Sortino, CAGR, worst daily drawdown) of the main run, as the
    Performance Summary shows them.
    """
    k0 = rc_current_offset(params)
    cur_lb = tuple(int(x) for x in params["main_lookbacks"])
    pair = rc_lb_name(cur_lb)
    tS, tC, tD = [float(x) if x is not None else np.nan for x in tested]

    def val(spec, f="S"):
        e = runs.get(spec)
        return float(e[f]) if e is not None and np.isfinite(e[f]) else np.nan

    def finite(xs):
        return [x for x in xs if np.isfinite(x)]

    def avg(xs):
        xs = finite(xs)
        return float(np.mean(xs)) if xs else np.nan

    def med(xs):
        xs = finite(xs)
        return float(np.median(xs)) if xs else np.nan

    block = plan["block"]
    wide_have = [s for s in plan["wide"] if s in runs and np.isfinite(runs[s]["S"])]
    has_wide = len(plan["wide"]) > 0 and len(wide_have) >= 0.8 * len(plan["wide"])
    nS, nC, nD = (avg([val(s, f) for s in block]) for f in ("S", "C", "D"))
    mS, mC, mD = ((med([val(s, f) for s in wide_have]) if has_wide else np.nan) for f in ("S", "C", "D"))
    lvS = _rc_level_rel(tS, nS, mild, spike)

    def span(a, b, fmt, first_max=False):
        if not np.isfinite(a):
            return "n/a"
        if not np.isfinite(b):
            return "about " + fmt(a)
        lo, hi = (max(a, b), min(a, b)) if first_max else (min(a, b), max(a, b))
        return fmt(lo) + " to " + fmt(hi)

    def note(a, b, fmt):
        s = "Neighbors' average " + fmt(a)
        return s + (" \u00b7 typical nearby run " + fmt(b) if np.isfinite(b) else " \u00b7 typical nearby run: Full check")

    tiles = [
        {"name": "Sortino", "tested": _rc_f2(tS), "range": span(nS, mS, _rc_f1),
         "pill": _rc_gap_pill(tS - nS, lvS), "note": note(nS, mS, _rc_f2)},
        {"name": "Return a year", "tested": _rc_pct(tC), "range": span(nC, mC, _rc_pct),
         "pill": _rc_gap_pill(tC - nC, _rc_level_pts(tC - nC, 0.015, 0.005), 1, 100.0, " pts"),
         "note": note(nC, mC, _rc_pct)},
        {"name": "Worst drawdown", "tested": _rc_pct(tD), "range": span(nD, mD, _rc_pct, first_max=True),
         "pill": _rc_gap_pill(tD - nD, _rc_level_pts(tD - nD, 0.03, 0.01), 1, 100.0, " pts"),
         "note": note(nD, mD, _rc_pct)},
    ]
    plan_around = span(nS, mS, _rc_f1) if has_wide else _rc_f1(nS)
    n_near = max(0, len(finite([val(s) for s in block])) - 1)
    near_line = (f"Its {n_near} nearest neighbors average {_rc_f2(nS)} (the lookback a month either way, "
                 f"the month-end trade a day earlier or later).")
    if lvS in ("bad", "warn"):
        summary = f"As tested, Sortino is {_rc_f2(tS)}. {near_line} Plan around {plan_around}."
    elif lvS == "dip":
        summary = f"As tested, Sortino is {_rc_f2(tS)}. {near_line} There is likely a steadier choice nearby."
    elif lvS == "good":
        summary = f"As tested, Sortino is {_rc_f2(tS)}. {near_line} This result holds up when the settings move."
    else:
        summary = f"As tested, Sortino is {_rc_f2(tS)}. The neighbor runs did not finish, so there is no comparison."

    # across time
    center_r = runs[plan["center"]]["r"]
    n_months = len(center_r)
    wins = [("Full period", None)] + [(f"Last {y} years", y * 12) for y in (20, 10, 5) if n_months > y * 12]
    time_rows = []
    for label, months in wins:
        def ws(spec):
            e = runs.get(spec)
            if e is None or len(e["r"]) == 0:
                return np.nan
            return rc_sortino(e["r"] if months is None else e["r"].iloc[-months:], rf_monthly)
        t = tS if months is None else ws(plan["center"])
        n = avg([ws(s) for s in block])
        m = med([ws(s) for s in wide_have]) if has_wide else np.nan
        time_rows.append({"label": label, "tested": _rc_f2(t), "near": _rc_f2(n),
                          "typical": _rc_f2(m) if has_wide else "Full check",
                          "range": span(n, m, _rc_f1),
                          "pill": _rc_gap_pill(t - n, _rc_level_rel(t, n, mild, spike))})

    levels = []

    # trade timing
    strip_days = rc_strip_days(k0)
    day_vals = [val(s) for s in plan["strip"]]
    i = strip_days.index(k0) if k0 in strip_days else None
    timing = None
    if i is not None and len(finite(day_vals)) >= 2:
        day_vals[i] = tS if np.isfinite(tS) else day_vals[i]
        adj = [j for j in (i - 1, i + 1) if 0 <= j < len(day_vals) and np.isfinite(day_vals[j])]
        nbT = avg([day_vals[j] for j in adj])
        lvT, pillT = rc_verdict(tS, nbT, mild, spike)
        levels.append(lvT)

        def sm(j):
            return avg([day_vals[q] for q in (j - 1, j, j + 1) if 0 <= q < len(day_vals)])

        inner = [j for j in range(1, len(day_vals) - 1)
                 if all(np.isfinite(day_vals[q]) for q in (j - 1, j, j + 1))]
        best_j = max(inner, key=sm) if inner else i
        cur_sm, best_sm = sm(i), sm(best_j)
        better = best_j != i and np.isfinite(cur_sm) and cur_sm != 0 and (best_sm - cur_sm) / abs(cur_sm) >= mild
        many = len(adj) > 1
        name = rc_cap(rc_trade_phrase(k0))
        if lvT in ("bad", "warn"):
            text = (f"{name} scores {_rc_f2(tS)}. The {'days beside it average' if many else 'day beside it scores'} "
                    f"{_rc_f2(nbT)}, so about {_rc_f2(tS - nbT)} of the score comes from the exact day.")
        elif lvT == "dip":
            text = (f"{name} scores {_rc_f2(tS)}, below the {'days' if many else 'day'} beside it ({_rc_f2(nbT)}). "
                    f"A day either way would have done better.")
        else:
            text = (f"{name} scores {_rc_f2(tS)}, in line with the {'days' if many else 'day'} beside it "
                    f"({_rc_f2(nbT)}). A day either way changes little.")
        if better:
            suggest = (f"Steadier option: {rc_trade_phrase(strip_days[best_j])}. Its three-day stretch averages "
                       f"{_rc_f2(best_sm)}, against {_rc_f2(cur_sm)} here.")
            suggest_pill = rc_pill("good", "up", _rc_signed(best_sm - cur_sm))
        else:
            suggest = (f"No steadier day nearby. This three-day stretch already averages the highest, at {_rc_f2(cur_sm)}."
                       if best_j == i else
                       f"No steadier day nearby. This stretch of days averages {_rc_f2(cur_sm)}, and the best "
                       f"nearby stretch averages {_rc_f2(best_sm)}.")
            suggest_pill = rc_pill("neutral", "neutral", "No change suggested")
        vmax = max(finite(day_vals + [0.0])) or 1.0
        days = []
        for j, kk in enumerate(strip_days):
            e = runs.get(plan["strip"][j])
            v = day_vals[j]
            tip = rc_cap(rc_trade_phrase(kk)) + ": " + ("no result" if not np.isfinite(v) else (
                f"Sortino {_rc_f2(v)}" + (f" \u00b7 return {_rc_pct(e['C'])} a year \u00b7 worst drawdown {_rc_pct(e['D'])}"
                                          if e is not None else "")))
            days.append({"name": rc_trade_short(kk), "v": v, "label": _rc_f2(v) if (j == i or j in adj) else "",
                         "h": int(max(4, round(max(v, 0.0) / vmax * 118))) if np.isfinite(v) else 0,
                         "cur": j == i, "tip": tip})
        timing = {"pill": pillT, "text": text, "suggest": suggest, "suggest_pill": suggest_pill, "days": days,
                  "ref": int(min(144, round(max(mS, 0.0) / vmax * 118))) if has_wide and np.isfinite(mS) else None,
                  "typical": _rc_f2(mS) if has_wide else None, "pair": pair}

    # main lookback
    lookback = None
    cells = [s[0] for s in plan["cells"]]
    cell_vals = {c: val(s) for c, s in zip(cells, plan["cells"])}
    cell_vals[cur_lb] = tS if np.isfinite(tS) else cell_vals.get(cur_lb, np.nan)
    others = finite([v for c, v in cell_vals.items() if c != cur_lb])
    if others:
        nbL = avg(others)
        lvL, pillL = rc_verdict(tS, nbL, mild, spike)
        levels.append(lvL)
        allv = finite(list(cell_vals.values()))
        rank = 1 + sum(1 for v in allv if v > tS)
        tail = f" It ranks {rank} of {len(allv)} on this trade day."
        nb_words = f"{len(others)} lookback{'s' if len(others) != 1 else ''} beside it"
        if lvL in ("bad", "warn"):
            text = (f"{pair} scores {_rc_f2(tS)}. The {nb_words} average {_rc_f2(nbL)}, so about "
                    f"{_rc_f2(tS - nbL)} of the score comes from this exact lookback.{tail}")
        elif lvL == "dip":
            text = f"{pair} scores {_rc_f2(tS)}, below the {nb_words} ({_rc_f2(nbL)}).{tail}"
        else:
            text = f"{pair} scores {_rc_f2(tS)}, in line with the {nb_words} ({_rc_f2(nbL)}).{tail}"
        cmin, cmax = min(allv), max(allv)
        if len(cur_lb) == 2:
            row_keys = sorted({c[0] for c in cells})
            col_keys = sorted({c[1] for c in cells})
            get = lambda a, b: (a, b)
        else:
            row_keys, col_keys = [None], sorted({c[0] for c in cells})
            get = lambda a, b: (b,)
        rows = []
        for a in row_keys:
            row = []
            for b in col_keys:
                c = get(a, b)
                v = cell_vals.get(c, np.nan)
                if c not in cell_vals or not np.isfinite(v):
                    row.append(None)
                    continue
                e = runs.get(plan["cells"][cells.index(c)])
                t = (v - cmin) / ((cmax - cmin) or 1.0)
                mix = lambda c0, c1: int(round(c0 + (c1 - c0) * t))
                row.append({"v": _rc_f2(v), "bg": f"rgb({mix(243, 185)},{mix(240, 168)},{mix(251, 232)})",
                            "cur": c == cur_lb,
                            "tip": f"{rc_lb_name(c)} months: Sortino {_rc_f2(v)}"
                                   + (f" \u00b7 return {_rc_pct(e['C'])} a year \u00b7 worst drawdown {_rc_pct(e['D'])}"
                                      if e is not None else "")})
            rows.append({"head": (f"{a} month{'s' if a != 1 else ''}" if a is not None else "Lookback"), "cells": row})
        lookback = {"pill": pillL, "text": text, "rows": rows,
                    "cols": [f"{b} month{'s' if b != 1 else ''}" for b in col_keys],
                    "two": len(cur_lb) == 2, "day": rc_trade_phrase(k0)}

    def bar_rows(items, cur_key):
        vmax = max(finite([v for _k, _n, _s, v in items] + [0.0])) or 1.0
        return [{"name": n, "sub": s, "v": _rc_f2(v), "w": int(round(max(v, 0.0) / vmax * 100)) if np.isfinite(v) else 0,
                 "cur": k == cur_key} for k, n, s, v in items]

    axes = []

    # position sizing
    sizing0 = params.get("sizing", SIZING_IF_UNSET)
    for other, specs in plan["sizing"].items():
        pairs = [(cell_vals.get(c, np.nan), val(s)) for c, s in zip(cells, specs)]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        if not pairs:
            continue
        a_avg, b_avg = avg([a for a, _b in pairs]), avg([b for _a, b in pairs])
        wins_n, n = sum(1 for a, b in pairs if a > b), len(pairs)
        cur_name, other_name = rc_sizing_name(sizing0), rc_sizing_name(other)
        other_note = (", " + RC_SIZING_NOTES[other].replace("2026, previously", "2026 and previously")
                      if other in RC_SIZING_NOTES else "")
        cur_note = (" (" + RC_SIZING_NOTES[sizing0] + ")" if sizing0 in RC_SIZING_NOTES else "")
        where = f"the {n} nearby lookback{'s' if n != 1 else ''}"
        text = (f"{cur_name}{cur_note} averages {_rc_f2(a_avg)} across {where}, against {_rc_f2(b_avg)} for "
                f"{other_name[0].lower() + other_name[1:]}{other_note}. It is ahead in {wins_n} of {n}.")
        if a_avg >= b_avg and wins_n * 3 >= n * 2:
            lv, pill = "good", rc_pill("good", "good", "Robust")
        elif b_avg > a_avg * (1 + mild) and wins_n * 3 <= n:
            lv, pill = "dip", rc_pill("neutral", "down", "Dip")
            text += f" Steadier option: {other_name[0].lower() + other_name[1:]} ({_rc_signed(b_avg - a_avg)})."
        else:
            lv, pill = "warn", rc_pill("warn", "warn", "Close call")
        levels.append(lv)
        axes.append({"title": "Position sizing", "pill": pill, "text": text,
                     "caption": f"Average Sortino across {where}, by sizing method.",
                     "bars": bar_rows([(sizing0, cur_name, RC_SIZING_NOTES.get(sizing0, ""), a_avg),
                                       (other, other_name, RC_SIZING_NOTES.get(other, ""), b_avg)], sizing0)})

    # volatility window
    v0 = int(params["vol_months"])
    vol_vals = {v0: avg([cell_vals.get(c, np.nan) for c in cells])}
    for v, specs in plan["vol"].items():
        vol_vals[v] = avg([val(s) for s in specs])
    keys = sorted(k for k, x in vol_vals.items() if np.isfinite(x))
    if v0 in keys and len(keys) >= 2:
        j = keys.index(v0)
        nbV = avg([vol_vals[keys[q]] for q in (j - 1, j + 1) if 0 <= q < len(keys)])
        lvV, pillV = rc_verdict(vol_vals[v0], nbV, mild, spike)
        levels.append(lvV)
        rest = ", ".join(f"{k} months {_rc_f2(vol_vals[k])}" for k in keys if k != v0)
        text = (f"{v0} months averages {_rc_f2(vol_vals[v0])} across the nearby lookbacks. "
                f"Other windows: {rest}.")
        axes.append({"title": "Volatility window", "pill": pillV, "text": text,
                     "caption": "Average Sortino across the nearby lookbacks, by window length.",
                     "bars": bar_rows([(k, f"{k} months", "", vol_vals[k]) for k in keys], v0)})

    # holdings count
    n0 = int(params["final_top_n"])
    top_vals = {n0: tS}
    for n, s in plan["topn"].items():
        top_vals[n] = val(s)
    keys = sorted(k for k, x in top_vals.items() if np.isfinite(x))
    if n0 in keys and len(keys) >= 2:
        nbN = avg([top_vals[k] for k in keys if k != n0])
        lvN, pillN = rc_verdict(tS, nbN, mild, spike)
        levels.append(lvN)
        rest = " and ".join(f"{k} funds scores {_rc_f2(top_vals[k])}" for k in keys if k != n0)
        text = (f"{n0} funds scores {_rc_f2(tS)}. {rest[0].upper() + rest[1:]}. One fund more or fewer is a "
                f"bigger step than a month of lookback, so read this one loosely.")
        axes.append({"title": "Holdings count", "pill": pillN, "text": text,
                     "caption": "Sortino at this lookback and trade day, by number of funds held.",
                     "bars": bar_rows([(k, f"{k} funds", "", top_vals[k]) for k in keys], n0)})

    # equity lookback
    eq0 = tuple(int(x) for x in params.get("equity_lookbacks", []))
    eq_vals = {c: val(s) for c, s in plan["eq"].items()}
    eq_keys = sorted(c for c, x in eq_vals.items() if np.isfinite(x))
    if eq_keys:
        nbE = avg([eq_vals[c] for c in eq_keys])
        lvE, pillE = rc_verdict(tS, nbE, mild, spike)
        levels.append(lvE)
        rest = ", ".join(f"{rc_lb_name(c)} {_rc_f2(eq_vals[c])}" for c in eq_keys)
        text = (f"The {rc_lb_name(eq0)} month equity lookback scores {_rc_f2(tS)}. "
                f"One month away on either leg: {rest}.")
        items = [(eq0, f"{rc_lb_name(eq0)} months", "", tS)] + [(c, f"{rc_lb_name(c)} months", "", eq_vals[c]) for c in eq_keys]
        axes.append({"title": "Equity lookback", "pill": pillE, "text": text,
                     "caption": "Sortino at this lookback and trade day, by equity lookback.",
                     "bars": bar_rows(sorted(items, key=lambda it: it[0]), eq0)})

    def count(x):
        return sum(1 for y in levels if y == x)

    chips = []
    if count("bad"):
        chips.append(rc_pill("bad", "bad", f"{count('bad')} spike{'s' if count('bad') > 1 else ''}"))
    if count("warn"):
        chips.append(rc_pill("warn", "warn", f"{count('warn')} to watch"))
    if count("dip"):
        chips.append(rc_pill("neutral", "down", f"{count('dip')} dip{'s' if count('dip') > 1 else ''}"))
    if count("good"):
        chips.append(rc_pill("good", "good", f"{count('good')} robust"))

    return {"summary": summary, "chips": chips, "tiles": tiles, "time_rows": time_rows, "timing": timing,
            "lookback": lookback, "axes": axes, "has_wide": has_wide, "wide_n": len(wide_have),
            "block_n": len(finite([val(s) for s in block])), "tested_S": tS, "near_S": nS,
            "mild": mild, "spike": spike}


def rc_settings_line(params):
    """The settings the panel varies, as the run has them (for the header and the last-run comparison)."""
    return {
        "Lookback": rc_lb_name(params["main_lookbacks"]) + " months",
        "Trade day": rc_trade_noun(rc_current_offset(params)),
        "Sizing": rc_sizing_name(params.get("sizing", SIZING_IF_UNSET)).lower(),
        "Volatility window": f"{int(params['vol_months'])} months",
        "Holdings": f"{int(params['final_top_n'])} funds",
        "Equity lookback": rc_lb_name(params.get("equity_lookbacks", [])) + " months",
    }


def rc_compare(prev, now):
    """This run against the last one in the session, or None for the first."""
    if not prev:
        return None
    if prev["context"] != now["context"]:
        return {"changed": "Basket, period, costs or prices changed",
                "text": "A different basket, period or cost is a different set of backtests, so the two runs are not compared.",
                "comparable": False}
    parts = [f"{k} {prev['settings'][k]} \u2192 {v}" for k, v in now["settings"].items()
             if k in prev["settings"] and prev["settings"][k] != v]
    if not parts:
        return None
    dT, dN = now["tested"] - prev["tested"], now["near"] - prev["near"]

    def delta(d):
        if not np.isfinite(d) or abs(d) < 0.05:
            return rc_pill("neutral", "neutral", _rc_signed(d))
        return rc_pill("good", "up", _rc_signed(d)) if d > 0 else rc_pill("bad", "down", _rc_signed(d))

    if not (np.isfinite(dT) and np.isfinite(dN)):
        text = "One of the two runs has no neighbor average, so there is no comparison."
    elif abs(dT) < 0.05:
        text = "As tested, the result barely moved."
    elif abs(dN) < 0.4 * abs(dT):
        text = (f"Most of the {_rc_f2(abs(dT))} {'gain' if dT > 0 else 'drop'} is the luck of the exact setting. "
                f"The neighborhood moved only {_rc_f2(abs(dN))}.")
    else:
        text = (f"The neighborhood moved too ({_rc_signed(dN)}), so this looks like a real difference "
                f"and not only the exact setting.")
    return {"changed": " \u00b7 ".join(parts), "comparable": True, "text": text,
            "tested": f"{_rc_f2(prev['tested'])} \u2192 {_rc_f2(now['tested'])}", "tested_pill": delta(dT),
            "near": f"{_rc_f2(prev['near'])} \u2192 {_rc_f2(now['near'])}", "near_pill": delta(dN)}


# ---- a second basket (added November 17, 2026) ------------------------------
# Opt-in, in the sidebar's Reality Check section. The same lookbacks and trade
# days run on "Basket B", the run's basket with funds left out and/or added, and
# the panel shows how many of the paired runs each basket wins. A quick screen,
# not the 90% odds test: it counts runs and averages Sortino, and says nothing
# about how likely a gap is to be luck.

RC_CMP_WIN_SHARE = (3, 4)   # a basket must win at least 3 of every 4 paired runs ...
RC_CMP_FLOOR = 0.5          # ... and average RC_MILD higher than the other (its average, at least this)
RC_CMP_TIE = 0.005          # a Sortino gap smaller than this is a tie


def rc_make_basket_b(fixed, equity, equity_top_n, leave_out=(), add=(), add_to="fixed"):
    """Basket B: the run's fixed basket and equity list with funds left out, then funds added.

    add_to is "fixed" or "equity"; a fund added to one list leaves the other. The equity count is lowered to
    the equity funds left, since run_backtest picks no equity at all when fewer are eligible than it is asked
    for. Returns a dict: fixed, equity, equity_top_n, left_out, added, add_to, capped (the run's count, when it
    was lowered, else None) and same (True when Basket B trades the run's own funds).
    """
    out = list(dict.fromkeys(leave_out))
    fixed_b = [t for t in fixed if t not in out]
    equity_b = [t for t in equity if t not in out]
    added = list(dict.fromkeys(t for t in add if t))
    for t in added:
        mine, other = (equity_b, fixed_b) if add_to == "equity" else (fixed_b, equity_b)
        if t in other:
            other.remove(t)
        if t not in mine:
            mine.append(t)
    n0 = int(equity_top_n)
    top_b = min(n0, len(equity_b))
    same = set(fixed_b) == set(fixed) and (n0 == 0 or (set(equity_b) == set(equity) and top_b == n0))
    return {"fixed": fixed_b, "equity": equity_b, "equity_top_n": top_b, "left_out": out, "added": added,
            "add_to": add_to, "capped": n0 if (n0 > 0 and top_b < n0) else None, "same": same}


def rc_basket_b_funds(b):
    """The funds Basket B can trade: its fixed basket, and its equity list while equity picks are on."""
    return list(b["fixed"]) + (list(b["equity"]) if b["equity_top_n"] > 0 else [])


def rc_basket_b_text(b):
    """Basket B in words: "your basket without HGER, plus VNQ in the fixed basket"."""
    parts = []
    if b["left_out"]:
        parts.append("without " + ", ".join(b["left_out"]))
    if b["added"]:
        parts.append("plus " + ", ".join(b["added"])
                     + (" in the equity list" if b["add_to"] == "equity" else " in the fixed basket"))
    return "your basket" + ((" " + ", ".join(parts)) if parts else "")


def rc_basket_b_params(params, b):
    """The run's params with Basket B's funds in place; costs, leverage, G1 and the rest are the run's own."""
    p = dict(params)
    p.update(fixed_tickers=list(b["fixed"]), equity_tickers=list(b["equity"]), equity_top_n=int(b["equity_top_n"]))
    return p


def rc_basket_b_issues(daily_prices, params, b):
    """Basket B's funds that the run's basket does not already cover and that lack the history the run needs,
    as (fund, sentence) pairs. The cutoff is the main run's data check: the start month less the longest
    lookback and one more month. A fund that may join late (JOIN_WHEN_AVAILABLE) is allowed in the fixed basket,
    and one in JOIN_WHEN_AVAILABLE_EQUITY in the equity basket while Basket B's other equity funds can fill its
    equity picks (as in the main run's data check)."""
    try:
        y, m = (int(x) for x in str(params["backtest_start"]).split("-")[:2])
        longest = max(int(x) for x in params["main_lookbacks"])
        required_by = pd.Timestamp(year=y, month=m, day=1) - pd.DateOffset(months=longest + 1)
    except (ValueError, TypeError, KeyError):
        required_by = pd.Timestamp("2012-01-01")
    covered = set(params.get("fixed_tickers", []))
    if int(params.get("equity_top_n", 0)) > 0:
        covered |= set(params.get("equity_tickers", []))
    issues = []
    eq_on_time = 0
    for t in b["equity"]:
        fv = daily_prices[t].dropna().index.min() if t in daily_prices.columns else None
        if fv is not None and not pd.isna(fv) and fv <= required_by:
            eq_on_time += 1
    eq_late_ok = int(b["equity_top_n"]) > 0 and eq_on_time >= int(b["equity_top_n"])
    for t in rc_basket_b_funds(b):
        first = daily_prices[t].dropna().index.min() if t in daily_prices.columns else None
        # A fund the run's own basket covers passed the run's data check, unless it is a late equity
        # fund that Basket B, with a fund left out, can no longer cover for.
        stranded = (first is not None and not pd.isna(first) and first > required_by
                    and t in b["equity"] and t not in b["fixed"] and not eq_late_ok)
        if t in covered and not stranded:
            continue
        if first is None or pd.isna(first):
            issues.append((t, f"{t} has no prices in the data."))
        elif first > required_by and not (t in JOIN_WHEN_AVAILABLE and t in b["fixed"]) and not (
                t in JOIN_WHEN_AVAILABLE_EQUITY and t in b["equity"] and t not in b["fixed"] and eq_late_ok):
            issues.append((t, f"{t} has prices only from {first:%B %d, %Y}, and this run needs them by "
                              f"{required_by:%B %Y}."))
    return issues


def rc_basket_b_calendar_clashes(daily_prices, extra):
    """Basket B's own funds (extra) that would change the run's trading calendar if they sat in the price table.

    A fund that printed a close on a weekday none of the run's funds traded (a fund on another country's
    exchange, say) adds that day, and the main run's trade days would move with it. Returns those funds; the
    list is empty in the usual case, a US-listed fund.
    """
    extra = [c for c in dict.fromkeys(extra) if c in daily_prices.columns]
    if not extra:
        return []
    rest = daily_prices.drop(columns=extra)
    if rest.shape[1] == 0:
        return []
    base = market_trading_dates(rest)
    return [c for c in extra if not market_trading_dates(rest.join(daily_prices[[c]])).equals(base)]


def rc_compute_other(daily_prices, monthly_prices, params_b, specs, rf_monthly, cache=None, progress=None,
                     parallel=True):
    """Run (or reuse) the specs on another basket (params_b), as rc_compute does for the run's own.

    Returns (runs, stats): runs maps a spec to its entry; stats counts the runs needed, reused and run now.
    Basket B's saved runs sit in the same cache, under a key that holds its own funds, so the run's own saved
    runs are untouched and a Basket B backtest that has run once is not run again.
    """
    t0 = _rc_time.time()
    base = rc_base_key(params_b, rc_fingerprint(daily_prices, params_b, rc_price_cut(daily_prices, params_b)))
    specs = list(dict.fromkeys(specs))
    runs, todo = {}, []
    for spec in specs:
        key = rc_key(base, spec)
        if cache is not None and key in cache:
            runs[spec] = cache[key]
        else:
            todo.append((spec, key))
    pending, used_workers = list(todo), 1
    workers = min(rc_pool_workers(), len(pending)) if parallel else 1
    if workers >= 2 and len(pending) >= RC_POOL_MIN:
        try:
            got = rc_run_parallel(
                daily_prices, monthly_prices, rf_monthly,
                [(key, rc_spec_params(params_b, spec)) for spec, key in pending], workers,
                (lambda n: progress(n, len(todo))) if progress is not None else None,
            )
            used_workers = workers
        except Exception:
            got = {}
        for spec, key in pending:
            if key in got:
                runs[spec] = got[key]
                if cache is not None:
                    cache[key] = got[key]
        pending = [(spec, key) for spec, key in pending if key not in got]
    n_before = len(todo) - len(pending)
    for i, (spec, key) in enumerate(pending):
        if progress is not None:
            progress(n_before + i, len(todo))
        try:
            res, _trades, deq = run_backtest(daily_prices, monthly_prices, rc_spec_params(params_b, spec))
            runs[spec] = rc_entry(res, deq, rf_monthly)
        except Exception:
            runs[spec] = rc_entry(None, None, rf_monthly)
        if cache is not None:
            cache[key] = runs[spec]
    stats = {"need": len(specs), "fresh": len(todo), "reused": len(specs) - len(todo),
             "secs": _rc_time.time() - t0, "base": base, "workers": used_workers}
    return runs, stats


def rc_compare_baskets(specs, runs_a, runs_b, mild=RC_MILD):
    """Basket B against the run's basket over the runs both finished: the wins each way, the average Sortino
    of each, and a verdict ("b", "a" or "close"). None when no run finished on both.

    A basket is ahead only when it wins at least 3 of every 4 paired runs and its average Sortino is at least
    `mild` higher than the other's (the other's average, but at least RC_CMP_FLOOR, so a near-zero average
    does not make a tiny gap look large).
    """
    pairs = []
    for s in specs:
        ea, eb = runs_a.get(s), runs_b.get(s)
        if ea is None or eb is None:
            continue
        if np.isfinite(ea["S"]) and np.isfinite(eb["S"]):
            pairs.append((float(ea["S"]), float(eb["S"])))
    n = len(pairs)
    if n == 0:
        return None
    a_avg = float(np.mean([a for a, _b in pairs]))
    b_avg = float(np.mean([b for _a, b in pairs]))
    a_wins = sum(1 for a, b in pairs if a - b > RC_CMP_TIE)
    b_wins = sum(1 for a, b in pairs if b - a > RC_CMP_TIE)
    num, den = RC_CMP_WIN_SHARE
    gap = b_avg - a_avg
    b_ahead = b_wins * den >= n * num and gap >= mild * max(abs(a_avg), RC_CMP_FLOOR)
    a_ahead = a_wins * den >= n * num and -gap >= mild * max(abs(b_avg), RC_CMP_FLOOR)
    return {"n": n, "a_avg": a_avg, "b_avg": b_avg, "gap": gap, "a_wins": a_wins, "b_wins": b_wins,
            "ties": n - a_wins - b_wins, "verdict": "b" if b_ahead else ("a" if a_ahead else "close")}


def rc_basket_card(daily_prices, monthly_prices, params, plan, runs_a, rf_monthly, mode, cache, b,
                   progress=None, mild=RC_MILD, parallel=True):
    """The Basket comparison card (an axis card's keys), its chip for the Testing bar, and Basket B's run counts.

    Basket B runs the specs the run's own check reads for the lookback cells and trade days around the run's
    (the Quick check's runs, or the Full check's), through the same saved-run cache and worker pool. When it
    cannot run (the same funds as the run's, no funds, a fund without enough history, or a fund the price
    download could not supply) the card is a note that says why, and the counts are None.
    """
    desc = rc_basket_b_text(b)
    chip = f"Compared with Basket B: {desc}"

    def note(tone, pill_text, text, stats=None):
        card = {"title": "Basket comparison", "pill": rc_pill(tone, tone if tone != "bad" else "bad", pill_text),
                "text": text, "caption": "", "bars": []}
        return card, chip, stats

    if b.get("error"):
        return note("warn", "Not run", "Basket B was not run. " + str(b["error"]))
    if b["same"]:
        return note("neutral", "Same basket",
                    "Basket B has the same funds as your basket, so there is nothing to compare. Leave a fund "
                    "out or add one in the sidebar, under Reality Check.")
    if not rc_basket_b_funds(b):
        return note("warn", "Not run", "Basket B was not run, because it has no funds left.")
    issues = rc_basket_b_issues(daily_prices, params, b)
    if issues:
        return note("warn", "Not run",
                    "Basket B was not run. " + " ".join(s for _t, s in issues)
                    + " Choose a later start date or a fund with a longer history, or give the fund a "
                      "backfill under Backfill Rules in the sidebar.")
    specs = list(dict.fromkeys(plan["wide"] if mode == "full" else plan["block"]))
    runs_b, stats = rc_compute_other(daily_prices, monthly_prices, rc_basket_b_params(params, b), specs,
                                     rf_monthly, cache=cache, progress=progress, parallel=parallel)
    cmp = rc_compare_baskets(specs, runs_a, runs_b, mild)
    if cmp is None:
        return note("warn", "Not run", "Basket B's backtests did not finish, so there is no comparison.", stats)
    mp = int(round(mild * 100))
    n, a_avg, b_avg = cmp["n"], cmp["a_avg"], cmp["b_avg"]
    head = f"Basket B wins {cmp['b_wins']} of {n} paired runs and your basket wins {cmp['a_wins']}"
    head += (f", with {cmp['ties']} tied." if cmp["ties"] else ".")
    avgs = (f"Averaged over the {n} runs, Basket B scores {_rc_f2(b_avg)} and your basket {_rc_f2(a_avg)} "
            f"({_rc_signed(cmp['gap'])} for Basket B).")
    if cmp["verdict"] == "b":
        pill = rc_pill("good", "up", "Basket B is better")
        verdict = (f"Basket B is ahead on both counts: it wins at least 3 of every 4 runs and averages at "
                   f"least {mp}% higher.")
    elif cmp["verdict"] == "a":
        pill = rc_pill("bad", "down", "Basket B is worse")
        verdict = (f"Your basket is ahead on both counts: it wins at least 3 of every 4 runs and averages at "
                   f"least {mp}% higher.")
    else:
        pill = rc_pill("warn", "warn", "No clear difference")
        verdict = (f"Neither basket is ahead on both counts (winning at least 3 of every 4 runs and averaging "
                   f"at least {mp}% higher), so treat them as close.")
    text = f"{head} {avgs} {verdict} This is a quick screen, not the 90% odds test."
    if b["capped"]:
        if b["equity_top_n"] > 0:
            text += (f" Basket B picks {b['equity_top_n']} equity fund{'s' if b['equity_top_n'] != 1 else ''} "
                     f"instead of {b['capped']}, since that is all its equity list has left.")
        else:
            text += " Basket B picks no equity funds, since its equity list is empty."
    vmax = max(a_avg, b_avg, 0.0) or 1.0

    def width(v):
        return int(round(max(v, 0.0) / vmax * 100))

    card = {"title": "Basket comparison", "pill": pill, "text": text,
            "caption": (f"Average Sortino across the {n} paired runs. Basket B used {stats['need']} backtests: "
                        f"{stats['reused']} reused from saved results, {stats['fresh']} run just now."),
            "bars": [{"name": "Your basket", "sub": "", "v": _rc_f2(a_avg), "w": width(a_avg), "cur": True},
                     {"name": "Basket B", "sub": desc, "v": _rc_f2(b_avg), "w": width(b_avg), "cur": False}]}
    return card, chip, stats


# ---- drawing the panel ------------------------------------------------------

RC_CSS = "".join([
    "<style>",
    ".rc-wrap{font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;color:#55524D;font-size:14.5px;line-height:1.5;display:flex;flex-direction:column;gap:22px;margin:0.2rem 0 1rem 0}",
    ".rc-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:13px;font-weight:600;white-space:nowrap}",
    ".rc-flex{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}",
    ".rc-bar-top{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:12px 16px;background:#F5F4F0;border:1px solid #E2DED6;border-radius:10px}",
    ".rc-chip{padding:4px 10px;border-radius:6px;background:#ffffff;border:1px solid #D9D5CC;color:#1C1B19;font-weight:500}",
    ".rc-strong{font-weight:600;color:#1C1B19}",
    ".rc-muted{color:#55524D}",
    ".rc-small{font-size:13px;color:#55524D}",
    ".rc-cmp{display:flex;flex-direction:column;gap:10px;padding:16px 18px;border:1px solid #D6D1E8;background:#EFEDF6;border-radius:10px}",
    ".rc-lead{font-family:'IBM Plex Serif',Georgia,serif;font-size:20px;line-height:1.45;font-weight:500;color:#1C1B19;max-width:880px}",
    ".rc-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}",
    ".rc-tile{border:1px solid #E2DED6;border-radius:10px;padding:18px 20px;display:flex;flex-direction:column;gap:12px}",
    ".rc-nums{display:flex;flex-direction:column;gap:10px}",
    ".rc-num{display:flex;flex-direction:column;gap:2px}",
    ".rc-big{font-size:28px;line-height:1.1;font-weight:700;color:#1C1B19}",
    ".rc-big-p{font-size:28px;line-height:1.1;font-weight:700;color:#4B3F8F}",
    ".rc-h3{font-family:'IBM Plex Serif',Georgia,serif;font-size:21px;font-weight:600;color:#1C1B19}",
    ".rc-h4{font-family:'IBM Plex Serif',Georgia,serif;font-size:17px;font-weight:600;color:#1C1B19}",
    ".rc-sec{display:flex;flex-direction:column;gap:12px}",
    ".rc-table{width:100%;border-collapse:collapse;font-size:14px;color:#1C1B19}",
    ".rc-table thead th{text-align:right;padding:11px 14px;border-bottom:1px solid #D9D5CC;background-color:#F5F4F0;color:#55524D;font-weight:600;font-size:13px}",
    ".rc-table thead th:first-child,.rc-table thead th:last-child{text-align:left}",
    ".rc-table tbody td{padding:9px 14px;border-bottom:1px solid #E6E3DC;text-align:right;font-variant-numeric:tabular-nums}",
    ".rc-table tbody td:first-child{text-align:left;color:#55524D;font-weight:500}",
    ".rc-table tbody td:last-child{text-align:left}",
    ".rc-table tbody td.rc-t{font-weight:700;color:#1C1B19}",
    ".rc-table tbody td.rc-r{font-weight:600;color:#4B3F8F}",
    ".rc-card{border:1px solid #E2DED6;border-radius:10px;padding:20px 22px;display:flex;flex-wrap:wrap;gap:20px 36px}",
    ".rc-left{flex:1 1 320px;min-width:0;display:flex;flex-direction:column;gap:10px}",
    ".rc-right{flex:1.4 1 420px;min-width:0;display:flex;flex-direction:column;gap:8px}",
    ".rc-sug{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding-top:12px;border-top:1px solid #E6E3DC}",
    ".rc-bars{position:relative;display:flex;align-items:flex-end;gap:10px;height:150px;border-bottom:1px solid #c9c9d6}",
    ".rc-col{flex:1 1 0;min-width:0;height:150px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:4px;cursor:help}",
    ".rc-col-l{font-size:13px;font-weight:600;color:#1C1B19;min-height:18px;font-variant-numeric:tabular-nums}",
    ".rc-col-b{display:block;width:100%;max-width:60px;border-radius:4px 4px 0 0}",
    ".rc-axis{display:flex;gap:10px}",
    ".rc-axis span{flex:1 1 0;min-width:0;text-align:center;font-size:13px;color:#55524D}",
    ".rc-axis span.rc-on{color:#1C1B19;font-weight:700}",
    ".rc-ref{position:absolute;left:0;right:0;border-top:2px dashed #6b6b7b;pointer-events:none}",
    ".rc-dash{display:inline-block;width:24px;border-top:2px dashed #6b6b7b;vertical-align:middle;margin-right:8px}",
    ".rc-grid{display:grid;gap:6px;align-items:stretch}",
    ".rc-gh{font-size:13px;color:#55524D;text-align:center}",
    ".rc-gr{font-size:13px;color:#55524D;display:flex;align-items:center}",
    ".rc-cell{min-height:58px;border-radius:8px;border:2px solid transparent;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#1C1B19;cursor:help}",
    ".rc-cell-v{font-size:16px;font-weight:600;font-variant-numeric:tabular-nums}",
    ".rc-tag{font-size:11px;font-weight:700;color:#4B3F8F;text-transform:uppercase;letter-spacing:0.04em;min-height:14px}",
    ".rc-hb{display:grid;grid-template-columns:minmax(120px,240px) minmax(0,1fr) 44px;gap:12px;align-items:center}",
    ".rc-hb-n{color:#1C1B19}",
    ".rc-hb-s{display:block;font-size:12px;color:#55524D}",
    ".rc-hb-b{display:block;height:18px;border-radius:0 4px 4px 0}",
    ".rc-hb-v{text-align:right;font-weight:600;color:#1C1B19;font-variant-numeric:tabular-nums}",
    "</style>",
])


def _rc_pill_html(p):
    bg, fg = RC_TONES[p["tone"]]
    return (f'<span class="rc-pill" style="background:{bg};color:{fg}"><span aria-hidden="true">'
            f'{RC_GLYPHS[p["glyph"]]}</span><span>{_rc_html.escape(p["text"])}</span></span>')


def rc_render_html(A, settings, stats, cmp=None):
    """The whole panel as one HTML string (no blank lines, so Markdown leaves it alone)."""
    esc = _rc_html.escape

    def axis_label(name):
        """A bar label on two lines: all but its last word, then its last word."""
        head, _sp, tail = name.rpartition(" ")
        return esc(head) + "<br>" + esc(tail) if head else esc(name)

    h = [RC_CSS, '<div class="rc-wrap">']

    secs = f", {stats['secs']:.0f} s" if stats["fresh"] else ""
    runs_text = (f"{stats['need']} backtests behind this check \u00b7 {stats['reused']} reused from saved results "
                 f"\u00b7 {stats['fresh']} run just now{secs}")
    h.append('<div class="rc-bar-top"><span class="rc-strong">Testing</span>')
    h.append("".join(f'<span class="rc-chip">{esc(k)} {esc(v)}</span>' for k, v in settings.items()))
    if A.get("chip_b"):
        h.append(f'<span class="rc-chip">{esc(A["chip_b"])}</span>')
    h.append(f'<span class="rc-muted" style="margin-left:auto">{esc(runs_text)}</span></div>')

    if cmp:
        h.append('<div class="rc-cmp"><div class="rc-flex"><span class="rc-strong">Compared with your last run</span>'
                 f'<span class="rc-muted">{esc(cmp["changed"])}</span></div>')
        if cmp["comparable"]:
            h.append('<div class="rc-flex" style="gap:12px 32px">'
                     f'<span class="rc-flex"><span class="rc-muted">Sortino as tested</span><span class="rc-strong">{esc(cmp["tested"])}</span>{_rc_pill_html(cmp["tested_pill"])}</span>'
                     f'<span class="rc-flex"><span class="rc-muted">Neighbors\' average</span><span class="rc-strong">{esc(cmp["near"])}</span>{_rc_pill_html(cmp["near_pill"])}</span>'
                     '</div>')
        h.append(f'<div style="color:#1C1B19">{esc(cmp["text"])}</div></div>')

    h.append(f'<div class="rc-sec"><div class="rc-lead">{esc(A["summary"])}</div>')
    if A["chips"]:
        h.append('<div class="rc-flex">' + "".join(_rc_pill_html(p) for p in A["chips"]) + "</div>")
    h.append('<div class="rc-tiles">')
    for t in A["tiles"]:
        h.append(f'<div class="rc-tile"><span class="rc-small" style="font-weight:600">{esc(t["name"])}</span>'
                 f'<div class="rc-nums"><div class="rc-num"><span class="rc-big">{esc(t["tested"])}</span><span class="rc-small">as tested</span></div>'
                 f'<div class="rc-num"><span class="rc-big-p">{esc(t["range"])}</span><span class="rc-small">realistic range</span></div></div>'
                 f'<div>{_rc_pill_html(t["pill"])}</div><span class="rc-small">{esc(t["note"])}</span></div>')
    h.append("</div></div>")

    h.append('<div class="rc-sec"><div class="rc-h3">Does it hold across time</div><table class="rc-table"><thead><tr>'
             "<th>Sortino</th><th>As tested</th><th>Neighbors' average</th><th>Typical nearby run</th><th>Realistic</th><th>Read</th>"
             "</tr></thead><tbody>")
    for r in A["time_rows"]:
        h.append(f'<tr><td>{esc(r["label"])}</td><td class="rc-t">{esc(r["tested"])}</td><td>{esc(r["near"])}</td>'
                 f'<td>{esc(r["typical"])}</td><td class="rc-r">{esc(r["range"])}</td><td>{_rc_pill_html(r["pill"])}</td></tr>')
    h.append("</tbody></table></div>")

    mp, sp = int(round(A["mild"] * 100)), int(round(A["spike"] * 100))
    legend = [(rc_pill("good", "good", "Robust"), f"within {mp}% of its neighbors"),
              (rc_pill("warn", "warn", "Mild peak"), f"{mp} to {sp}% above them"),
              (rc_pill("bad", "bad", "Spike"), f"more than {sp}% above them"),
              (rc_pill("neutral", "down", "Dip"), f"more than {mp}% below them")]
    h.append('<div class="rc-sec"><div class="rc-h3">Which settings this result leans on</div><div class="rc-flex" style="gap:8px 20px">')
    h.append("".join(f'<span class="rc-flex rc-small">{_rc_pill_html(p)}<span>{esc(m)}</span></span>' for p, m in legend))
    h.append("</div>")

    T = A["timing"]
    if T:
        h.append('<div class="rc-card"><div class="rc-left">'
                 f'<div class="rc-flex"><span class="rc-h4">Trade timing</span>{_rc_pill_html(T["pill"])}</div>'
                 f'<div>{esc(T["text"])}</div>'
                 f'<div class="rc-sug">{_rc_pill_html(T["suggest_pill"])}<span style="flex:1 1 220px;min-width:0">{esc(T["suggest"])}</span></div>'
                 '</div><div class="rc-right">'
                 f'<span class="rc-small">Sortino by trade day, at the {esc(T["pair"])} lookback. Hover a bar for its figures.</span>'
                 '<div class="rc-bars">')
        if T["ref"] is not None:
            h.append(f'<div class="rc-ref" style="bottom:{T["ref"]}px"></div>')
        for d in T["days"]:
            fill = "#4B3F8F" if d["cur"] else "#948CC0"
            h.append(f'<div class="rc-col" title="{esc(d["tip"])}"><span class="rc-col-l">{esc(d["label"])}</span>'
                     f'<span class="rc-col-b" style="height:{d["h"]}px;background:{fill}"></span></div>')
        h.append('</div><div class="rc-axis">'
                 + "".join(f'<span class="{"rc-on" if d["cur"] else ""}">{axis_label(d["name"])}</span>' for d in T["days"])
                 + "</div>")
        if T["typical"] is not None:
            h.append(f'<span class="rc-small"><span class="rc-dash"></span>Typical nearby run: {esc(T["typical"])}</span>')
        h.append("</div></div>")

    L = A["lookback"]
    if L:
        ncol = len(L["cols"])
        tmpl = (f"84px repeat({ncol}, minmax(0, 1fr))" if L["two"] else f"repeat({ncol}, minmax(0, 1fr))")
        h.append('<div class="rc-card"><div class="rc-left">'
                 f'<div class="rc-flex"><span class="rc-h4">Main momentum lookback</span>{_rc_pill_html(L["pill"])}</div>'
                 f'<div>{esc(L["text"])}</div></div><div class="rc-right">'
                 f'<span class="rc-small">Sortino by lookback, trading on {esc(L["day"])}. Darker is higher. Hover a cell for its figures.</span>'
                 f'<div class="rc-grid" style="grid-template-columns:{tmpl}">')
        if L["two"]:
            h.append('<span class="rc-gh" style="text-align:left">Short / long</span>')
        h.append("".join(f'<span class="rc-gh">{esc(c)}</span>' for c in L["cols"]))
        for row in L["rows"]:
            if L["two"]:
                h.append(f'<span class="rc-gr">{esc(row["head"])}</span>')
            for c in row["cells"]:
                if c is None:
                    h.append("<span></span>")
                    continue
                border = "#4B3F8F" if c["cur"] else "transparent"
                h.append(f'<div class="rc-cell" title="{esc(c["tip"])}" style="background:{c["bg"]};border-color:{border}">'
                         f'<span class="rc-cell-v">{esc(c["v"])}</span><span class="rc-tag">{"current" if c["cur"] else ""}</span></div>')
        h.append("</div></div></div>")

    for ax in A["axes"]:
        h.append('<div class="rc-card"><div class="rc-left">'
                 f'<div class="rc-flex"><span class="rc-h4">{esc(ax["title"])}</span>{_rc_pill_html(ax["pill"])}</div>'
                 f'<div>{esc(ax["text"])}</div></div><div class="rc-right" style="gap:10px">'
                 f'<span class="rc-small">{esc(ax["caption"])}</span>')
        for b in ax["bars"]:
            fill = "#4B3F8F" if b["cur"] else "#948CC0"
            sub = f'<span class="rc-hb-s">{esc(b["sub"])}</span>' if b["sub"] else ""
            tag = ' <span class="rc-tag">current</span>' if b["cur"] else ""
            h.append(f'<div class="rc-hb"><span class="rc-hb-n">{esc(b["name"])}{tag}{sub}</span>'
                     f'<span><span class="rc-hb-b" style="width:{b["w"]}%;background:{fill}"></span></span>'
                     f'<span class="rc-hb-v">{esc(b["v"])}</span></div>')
        h.append("</div></div>")
    h.append("</div></div>")
    return "".join(h)


def rc_show_panel(daily_prices, monthly_prices, params, results_df, daily_equity, strat_metrics, mode, extras=(),
                  basket_b=None):
    """Run the check and draw the Reality Check section (called from the results page)."""
    ae_md(ae_head(
        "Reality Check", "How much of this result survives",
        "The settings moved a little in each direction. Every figure is a backtest: the same run repeated "
        "on the settings one step away."))
    rf_monthly = monthly_prices["BIL"].pct_change() if "BIL" in monthly_prices.columns else None
    cache = rc_cache_load()
    bar = st.progress(0.0, text="Reality Check: looking for saved runs")

    def progress(i, n):
        bar.progress(min(1.0, i / max(n, 1)), text=f"Reality Check: {i} of {n} neighbor backtests done")

    plan, runs, stats = rc_compute(daily_prices, monthly_prices, params, results_df, daily_equity,
                                   rf_monthly, mode, cache=cache, progress=progress, extras=extras)
    bar.empty()
    if stats["fresh"]:
        rc_cache_save(cache)

    def metric(key):
        v = strat_metrics.get(key, np.nan)
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else np.nan

    tested = (metric("Sortino Ratio"), metric("CAGR"), metric("Max Drawdown (Daily)"))
    A = rc_analyze(params, plan, runs, tested, rf_monthly)
    if basket_b is not None:
        # Basket B (opt-in; November 17, 2026): its card goes last, its chip into the Testing bar.
        bar_b = st.progress(0.0, text="Reality Check: Basket B, looking for saved runs")

        def progress_b(i, n):
            bar_b.progress(min(1.0, i / max(n, 1)), text=f"Reality Check: Basket B, {i} of {n} backtests done")

        try:
            card_b, chip_b, stats_b = rc_basket_card(daily_prices, monthly_prices, params, plan, runs, rf_monthly,
                                                     mode, cache, basket_b, progress=progress_b)
            A["axes"].append(card_b)
            A["chip_b"] = chip_b
            if stats_b and stats_b["fresh"]:
                rc_cache_save(cache)
        except Exception as _b_exc:
            st.warning(f"The Basket B comparison could not run: {_b_exc}")
        finally:
            bar_b.empty()
    settings = rc_settings_line(params)
    now = {"context": stats["base"], "settings": settings, "tested": A["tested_S"], "near": A["near_S"]}
    cmp = None
    try:
        cmp = rc_compare(st.session_state.get("rc_last_run"), now)
        st.session_state["rc_last_run"] = now
    except Exception:
        cmp = None
    st.markdown(rc_render_html(A, settings, stats, cmp), unsafe_allow_html=True)
    lines = [
        "Neighbors' average: the lookbacks one month away on each leg, on this trade day and the days either "
        f"side of it ({A['block_n']} runs). Typical nearby run: the median of the same lookbacks on 21 trade "
        "days, from the 19th-to-last trading day of the month to the 2nd trading day of next month. "
        "Realistic range: the span between the two."
    ]
    if not A["has_wide"]:
        lines.append(
            "The typical nearby run needs the Full check (Reality Check in the sidebar), which adds about "
            f"{max(0, len(plan['wide']) - A['wide_n'])} backtests the first time and is saved after that."
        )
    lines.append(
        "The signal is read at the close the day before the trade, and each lookback runs from earlier "
        "month-end closes to that close."
    )
    st.caption(" ".join(lines))
    return A, stats


# =============================================================================
# DEFAULT SETTINGS COMPARISON  (added November 20, 2026)
# =============================================================================
# Tyler's request of that day: hold the result of a run on all default
# settings and, when a setting changes, compare the new result with it
# automatically. Until then he ran the defaults over a period, took a
# screenshot, changed a setting and ran again.
#
# After a run whose settings differ from the app's starting ones, the same
# period is run on the starting settings ("the default run"), and a card at
# the top of the Overview tab shows the two side by side: what changed, each
# run's headline figures and the change in every one, flagged green, red or
# gray. The Performance Summary gains a Default settings column. When the
# Reality Check is on, the card also compares the two runs' nearby settings
# (the Reality Check's neighbors' average), because one run's gain can be the
# luck of the exact setting.
#
#   Default settings   the DS_* constants below, which the sidebar's widgets
#                      read as their starting values, with position sizing,
#                      trading cost and the Backfill Rules at their own
#                      defaults, leverage off and the G1 cash rule off
#   The default run    run_backtest on those settings, over the run's own
#                      period, on prices built by the default Backfill Rules
#
# The default run is saved in reality_check_cache\ under its settings, a
# fingerprint of its prices and a stamp of this file, so it runs once for a
# period and is reused until the prices behind it or this file change (on the
# day a month closes it is run fresh each time; see ds_month_closing). A run
# that is itself on the default settings is saved the same way. Nothing here
# changes the run's own results, trade log, results log or Reality Check,
# and unticking "Compare with the default settings" in the sidebar switches
# all of it off.
#
# Build and checks:
# C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System\Backtest Engine\Default Settings Compare 2026-11-20\

# The app's starting settings. The sidebar's widgets read these, so the
# default run and a run made without touching the sidebar cannot drift apart.
# To change a starting value, change it here.
DS_MAIN_LOOKBACK = "2/8-month blend (50/50)"
DS_FINAL_TOP_N = 4
DS_EXEC_OFFSET = 1                     # the last trading day of the month (see rc_exec_label)
DS_VOL_MONTHS = 12
DS_FIXED = ["TLT", "GLDM", "BIL", "BTAL", "HGER", "IBIT"]
DS_EQUITY = ["QQQ", "IWM", "DXJ", "EMXC"]
DS_EQUITY_TOP_N = 3
DS_EQUITY_LOOKBACK = "1/3-month blend (50/50)"

DS_TITLE = "Compared with the default settings"
DS_COLUMN = "Default settings"         # the default run's name on the page
DS_BAND = 0.05                         # a Sortino change smaller than this reads as about the same
DS_AGREE = 0.4                         # nearby settings agree when their average moved the same way, by at least
#                                        this share of the change as tested (and by at least DS_BAND)
DS_CACHE_FILE = os.path.join(RC_CACHE_DIR, "default_settings_runs.pkl")
DS_CACHE_MAX = 12                      # saved default runs kept, about 0.4 MB each (the oldest go first)
# The card's rows: label, compute_metrics key, kind ("ratio", "pct" or "dd"),
# the smallest change that gets a color (in the units the pill shows: ratio
# points, or percentage points), and whether a higher figure is better (None:
# the change is shown without a color). The first three are drawn in bold.
DS_ROWS = [
    ("Sortino ratio", "Sortino Ratio", "ratio", DS_BAND, True),
    ("Return a year (CAGR)", "CAGR", "pct", 0.25, True),
    ("Max drawdown, daily", "Max Drawdown (Daily)", "dd", 0.5, True),
    ("Max drawdown, month-end", "Max Drawdown", "dd", 0.5, True),
    ("Martin ratio", "Martin Ratio", "ratio", 0.10, True),
    ("Calmar ratio", "Calmar Ratio", "ratio", 0.05, True),
    ("Volatility a year", "Annualized Vol", "pct", None, None),
    ("Worst year", "Worst Year", "pct", 0.5, True),
]
DS_BOLD = ("Sortino Ratio", "CAGR", "Max Drawdown (Daily)")


# ---- the default settings ---------------------------------------------------

def ds_default_backfill_rules():
    """The Backfill Rules of a run with no row edited, in the order the app applies them: the sidebar's
    default rows (BACKFILL_DEFAULTS), then the balanced benchmarks' chains and the S&P 500 line's."""
    rules = {}
    for primary, chain in BACKFILL_DEFAULTS:
        primary = primary.strip().upper()
        chain = [t.strip().upper() for t in chain.split(",") if t.strip()]
        if primary and chain:
            rules[primary] = chain
    for cfg in BAL_BENCHMARKS:
        if cfg.get("backfill"):
            rules[cfg["ticker"]] = [t.strip().upper() for t in cfg["backfill"]]
    if BENCHMARK_BACKFILL and BENCHMARK_TICKER not in rules:
        rules[BENCHMARK_TICKER] = list(BENCHMARK_BACKFILL)
    return rules


def ds_default_tickers():
    """Every ticker a default run downloads: its funds, the benchmark lines, the calendar anchor and every
    Backfill Rules source."""
    need = list(DS_FIXED) + list(DS_EQUITY) + [BENCHMARK_TICKER, "BIL", CALENDAR_TICKER]
    need += [c["ticker"] for c in BAL_BENCHMARKS]
    for chain in ds_default_backfill_rules().values():
        need.extend(chain)
    return sorted(set(need))


def ds_default_params(backtest_start, backtest_end=""):
    """run_backtest's params for the default settings over a period, as the sidebar builds them at startup."""
    main_lbs, main_w = MAIN_LOOKBACK_OPTIONS[DS_MAIN_LOOKBACK]
    eq_lbs, eq_w = EQUITY_LOOKBACK_OPTIONS[DS_EQUITY_LOOKBACK]
    rules = ds_default_backfill_rules()
    return {
        "fixed_tickers": [t for t in DEFAULT_FIXED_ASSETS if t in DS_FIXED],
        "equity_tickers": [t for t in DEFAULT_EQUITY_ETFS if t in DS_EQUITY],
        "equity_top_n": DS_EQUITY_TOP_N,
        "equity_lookbacks": eq_lbs,
        "equity_lookback_weights": eq_w,
        "equity_lookback_label": DS_EQUITY_LOOKBACK,
        "main_lookbacks": main_lbs,
        "main_lookback_weights": main_w,
        "main_lookback_label": DS_MAIN_LOOKBACK,
        "final_top_n": DS_FINAL_TOP_N,
        "vol_months": DS_VOL_MONTHS,
        "sizing": SIZING_DEFAULT,
        "exec_timing": {1: "last_day", 0: "first_day"}.get(DS_EXEC_OFFSET, "offset"),
        "exec_timing_label": rc_exec_label(DS_EXEC_OFFSET),
        **({"signal_offset": DS_EXEC_OFFSET} if DS_EXEC_OFFSET not in (0, 1) else {}),
        # The sidebar's box holds the cost in percent and hands it back divided by 100.
        "cost_per_side": (DEFAULT_COST_PER_SIDE * 100) / 100.0,
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
        "backfill_summary": "; ".join(f"{k} <- {', '.join(v)}" for k, v in rules.items()) if rules else "None",
        "leverage_params": {"enabled": False},
        "g1": {"enabled": False},
    }


def ds_default_line():
    """The default settings in one line, for the card's footnote."""
    p = ds_default_params("")
    eq = ""
    if p["equity_top_n"] > 0:
        eq = (f" and the top {p['equity_top_n']} of {', '.join(p['equity_tickers'])} on the "
              f"{rc_lb_name(p['equity_lookbacks'])}-month lookback")
    return (f"the {rc_lb_name(p['main_lookbacks'])}-month lookback, {p['final_top_n']} holdings, "
            f"{rc_sizing_name(p['sizing']).lower()} on a {p['vol_months']}-month window, trades on the "
            f"{rc_trade_noun(rc_current_offset(p))}, {p['cost_per_side']:.2%} cost per side, "
            f"{', '.join(p['fixed_tickers'])}{eq}")


def ds_lev_text(lev):
    """A leverage setting in a word or two."""
    if not lev or not lev.get("enabled"):
        return "off"
    method = lev.get("method")
    if method == "constant":
        return f"constant {float(lev.get('mult_constant', 1.25)):.2f}x"
    if method == "holdings_gate":
        return f"holdings gate {float(lev.get('mult_leveraged', 1.25)):.2f}x"
    if method == "vt_trend":
        return f"{lev.get('ticker', 'VT')} trend"
    return "on"


def ds_signature(params, rules):
    """Everything that decides a run's results except its period, as text. Two runs with the same signature
    over the same period and prices are the same backtest."""
    lev = params.get("leverage_params") or {}
    g1 = params.get("g1") or {}
    n_eq = int(params.get("equity_top_n", 0))
    return json.dumps(_rc_plain({
        "fixed": list(params.get("fixed_tickers", [])),
        "equity": list(params.get("equity_tickers", [])) if n_eq > 0 else [],
        "equity_top_n": n_eq,
        "equity_lb": ([list(params.get("equity_lookbacks", [])), list(params.get("equity_lookback_weights", []))]
                      if n_eq > 0 else []),
        "main_lb": [list(params["main_lookbacks"]), list(params["main_lookback_weights"])],
        "top_n": int(params["final_top_n"]),
        "vol": int(params["vol_months"]),
        "sizing": params.get("sizing", SIZING_IF_UNSET),
        "day": rc_current_offset(params),
        "cost": params.get("cost_per_side", DEFAULT_COST_PER_SIDE),
        "leverage": lev if lev.get("enabled") else {},
        "g1": {k: v for k, v in g1.items() if k != "yield_series"} if g1.get("enabled") else {},
        "rules": [[k, list(v)] for k, v in rules.items()],
    }), sort_keys=True)


def ds_changes(params, rules, base_params, base_rules):
    """What a run changed from the default settings, as short phrases in sidebar order (none for a default run)."""
    out = []

    def moved(name, a, b, unit=""):
        if a != b:
            out.append(f"{name} {a} \u2192 {b}{unit}")

    def lookback(name, key):
        a = (list(base_params[key + "s"]), [round(float(x), 6) for x in base_params[key + "_weights"]])
        b = (list(params[key + "s"]), [round(float(x), 6) for x in params[key + "_weights"]])
        if a == b:
            return
        if a[0] != b[0]:
            out.append(f"{name} {rc_lb_name(a[0])} \u2192 {rc_lb_name(b[0])} months")
        else:
            out.append(f"{name} {base_params.get(key + '_label', '')} \u2192 {params.get(key + '_label', '')}")

    def funds(name, a, b):
        added = [t for t in b if t not in a]
        gone = [t for t in a if t not in b]
        if added or gone:
            out.append(name + " " + ", ".join(["+" + t for t in added] + ["\u2212" + t for t in gone]))
        elif list(a) != list(b):
            out.append(name + " in a different order")

    lookback("Main lookback", "main_lookback")
    moved("Holdings", int(base_params["final_top_n"]), int(params["final_top_n"]))
    moved("Trade day", rc_trade_noun(rc_current_offset(base_params)), rc_trade_noun(rc_current_offset(params)))
    moved("Position sizing", rc_sizing_name(base_params.get("sizing", SIZING_IF_UNSET)).lower(),
          rc_sizing_name(params.get("sizing", SIZING_IF_UNSET)).lower())
    moved("Volatility window", int(base_params["vol_months"]), int(params["vol_months"]), " months")
    moved("Trading cost", f"{float(base_params.get('cost_per_side', DEFAULT_COST_PER_SIDE)):.2%}",
          f"{float(params.get('cost_per_side', DEFAULT_COST_PER_SIDE)):.2%}", " per side")
    moved("Leverage", ds_lev_text(base_params.get("leverage_params")), ds_lev_text(params.get("leverage_params")))
    g1 = params.get("g1") or {}
    if g1.get("enabled"):
        out.append("G1 cash rule off \u2192 on")
    funds("Fixed basket", base_params["fixed_tickers"], params["fixed_tickers"])
    n_base, n_run = int(base_params["equity_top_n"]), int(params["equity_top_n"])
    moved("Equity picks", n_base, n_run)
    if n_run > 0:
        funds("Equity funds", base_params["equity_tickers"], params["equity_tickers"])
        lookback("Equity lookback", "equity_lookback")
    hist = []
    for k in dict.fromkeys(list(rules) + list(base_rules)):
        if rules.get(k) == base_rules.get(k) or (g1.get("enabled") and k == g1.get("fund") and k not in base_rules):
            continue
        hist.append(f"{k} history from {', '.join(rules[k])}" if k in rules else f"{k} with no backfill")
    if len(hist) > 3:
        hist = hist[:3] + [f"{len(hist) - 3} more fund histories"]
    out.extend(hist)
    if not hist and [k for k in rules if k in base_rules] != list(base_rules):
        out.append("Backfill Rules in a different order")
    return out


# ---- the default run's prices -----------------------------------------------

def ds_plan_prices(daily_prices, run_rules, extra=()):
    """How the default run gets its price table, decided before the run's own Backfill Rules are applied.

    daily_prices is the run's table as downloaded, with the default run's own tickers (extra: the ones the
    run itself did not need) still in it. Returns {"cols", "share", "raw"}. share: the run's finished table
    can serve, read through the default run's columns, because the run has every default ticker of its
    own, builds each of them by the default rules, and trades on the same calendar. Otherwise raw holds a
    copy of the default run's columns, taken now, for ds_build_prices. Raises when a default ticker has no
    prices.
    """
    want = ds_default_tickers()
    missing = [c for c in want if c not in daily_prices.columns or daily_prices[c].dropna().empty]
    if missing:
        raise MissingPriceDataError("no prices for " + ", ".join(missing))
    base_rules = ds_default_backfill_rules()
    share = False
    if not extra:
        mine = [(k, list(v)) for k, v in run_rules.items() if k in base_rules]
        others = [k for k in run_rules if k not in base_rules]
        share = (mine == [(k, list(v)) for k, v in base_rules.items()]
                 and not any(k in want for k in others)
                 and market_trading_dates(daily_prices).equals(market_trading_dates(daily_prices[want])))
    return {"cols": want, "share": share, "raw": None if share else daily_prices[want].copy()}


def ds_build_prices(raw_prices, rules):
    """The default Backfill Rules applied to raw prices, as the app applies a run's own."""
    daily = raw_prices.copy()
    for primary, chain in rules.items():
        if primary in daily.columns or any(bf in daily.columns for bf in chain):
            spliced, _segments = build_backfilled_series(
                daily, primary, chain,
                fee_annual=BACKFILL_FEE_ANNUAL.get(primary, 0.0),
                primary_from=PRIMARY_HISTORY_FROM.get(primary),
                source_from=BACKFILL_SOURCE_FROM.get(primary),
            )
            daily[primary] = spliced
    return daily


def ds_finish_prices(plan, run_daily):
    """The default run's daily and month-end prices, once the run's own table is finished (see ds_plan_prices)."""
    if plan["share"]:
        daily = run_daily[plan["cols"]]
    else:
        daily = ds_build_prices(plan["raw"], ds_default_backfill_rules())
    return daily, get_month_end_prices(daily)


# ---- the saved default runs -------------------------------------------------

def ds_cache_load():
    try:
        with open(DS_CACHE_FILE, "rb") as fh:
            cache = _rc_pickle.load(fh)
        return cache if isinstance(cache, dict) else {}
    except Exception:
        return {}


def ds_cache_save(cache):
    """Write the saved default runs (oldest dropped past DS_CACHE_MAX). Never raises."""
    try:
        if len(cache) > DS_CACHE_MAX:
            keep = sorted(cache, key=lambda k: cache[k].get("t", 0.0))[-DS_CACHE_MAX:]
            for k in set(cache) - set(keep):
                del cache[k]
        os.makedirs(RC_CACHE_DIR, exist_ok=True)
        tmp = DS_CACHE_FILE + ".tmp"
        with open(tmp, "wb") as fh:
            _rc_pickle.dump(cache, fh, protocol=4)
        os.replace(tmp, DS_CACHE_FILE)
        return True
    except Exception:
        return False


def ds_engine_stamp():
    """A short hash of this file, so a saved default run is not reused after the engine's code changes."""
    try:
        with open(RC_ENGINE_FILE, "rb") as fh:
            return _rc_hashlib.sha1(fh.read()).hexdigest()[:12]
    except Exception:
        return "unknown"


def ds_keys(base_daily, base_params):
    """Two keys for the default run on these prices and this period: the one the Reality Check saves it
    under (as the center of a default run's own check), and this section's own, which adds the file stamp."""
    base = rc_base_key(base_params, rc_fingerprint(base_daily, base_params, rc_price_cut(base_daily, base_params)))
    rc_k = rc_key(base, rc_spec(base_params))
    return rc_k, _rc_hashlib.sha1((rc_k + "|" + ds_engine_stamp()).encode("utf-8")).hexdigest()


def ds_month_closing(daily_prices):
    """True when the prices end on the last calendar day of a month.

    On that day a run that trades on the last trading day (the default settings' trade day) already counts
    the month, on a close that may still change, while rc_fingerprint, which the saved runs are keyed by,
    moves only once the prices reach the next month. So a default run made on such a day is neither read
    from the saved runs nor saved: it is run again each time, on the prices the run itself reads.
    """
    dates = market_trading_dates(daily_prices)
    if len(dates) == 0:
        return False
    last = dates[-1]
    return (last + pd.Timedelta(days=1)).month != last.month


def ds_held(trade_df):
    """Each hold month's funds, read from a trade log: {"YYYY-MM": frozenset of tickers}."""
    out = {}
    if trade_df is None or len(trade_df) == 0 or "final_holdings" not in trade_df.columns:
        return out
    for month, text in zip(trade_df["hold_month"], trade_df["final_holdings"]):
        out[str(month)] = frozenset(part.split(":")[0].strip() for part in str(text).split(" | ") if ":" in part)
    return out


def ds_entry(monthly_returns, trade_df, daily_equity, rf_monthly):
    """What is kept of a default run: the Reality Check's entry (monthly returns and three headline
    figures), plus its daily equity curve and each month's funds."""
    r = monthly_returns.dropna().astype(float)
    m = compute_metrics(r, rf_monthly=rf_monthly, daily_equity=daily_equity) if len(r) else {}

    def num(key):
        v = m.get(key, np.nan)
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else np.nan

    return {"r": r, "S": num("Sortino Ratio"), "C": num("CAGR"), "D": num("Max Drawdown (Daily)"),
            "t": _rc_time.time(), "deq": daily_equity, "held": ds_held(trade_df)}


def ds_get_run(base_daily, base_monthly, base_params, rf_monthly):
    """The default run for these prices and this period: read from the saved runs, or run now and saved.
    Returns (entry, run just now, the Reality Check's key for it)."""
    rc_k, key = ds_keys(base_daily, base_params)
    saved = not ds_month_closing(base_daily)
    cache = ds_cache_load() if saved else {}
    entry = cache.get(key)
    if isinstance(entry, dict) and "deq" in entry and "r" in entry:
        return entry, False, rc_k
    p = dict(base_params)
    p["fast_daily"] = True    # the same daily curve as the day-by-day loop, bit for bit, in half the time
    res, trades, deq = run_backtest(base_daily, base_monthly, p)
    r = res["portfolio_return"] if len(res) else pd.Series(dtype=float)
    entry = ds_entry(r, trades, deq, rf_monthly)
    if saved and len(entry["r"]):
        cache[key] = entry
        ds_cache_save(cache)
    return entry, True, rc_k


# ---- comparing the two runs -------------------------------------------------

def ds_compare(run_r, run_deq, run_metrics, run_held, entry, rf_run=None, rf_base=None):
    """This run against the default run over the months both cover. None when they share no month.

    When the two cover the same months, the run's figures are the Performance Summary's own (run_metrics)
    and the default run's are compute_metrics on all of it. When they do not (a start the default settings
    cannot reach, or a trade day whose last month is not finished), both are worked out again on the
    shared months, with each daily curve cut to them the way run_backtest cuts one to a period.
    """
    a = run_r.dropna().astype(float)
    b = entry["r"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    same = len(common) == len(a) == len(b)

    def cut(deq):
        if deq is None or len(deq) == 0:
            return deq
        lo = pd.Timestamp(year=common[0].year, month=common[0].month, day=1)
        return deq[(deq.index >= lo) & (deq.index <= common[-1])]

    if same:
        m_a = run_metrics
        m_b = compute_metrics(b, rf_monthly=rf_base, daily_equity=entry.get("deq"))
    else:
        m_a = compute_metrics(a.loc[common], rf_monthly=rf_run, daily_equity=cut(run_deq))
        m_b = compute_metrics(b.loc[common], rf_monthly=rf_base, daily_equity=cut(entry.get("deq")))
    months = [d.strftime("%Y-%m") for d in common]
    held_b = entry.get("held") or {}
    both = [m for m in months if m in run_held and m in held_b]
    return {
        "same": same, "n": len(common), "first": common[0], "last": common[-1],
        "run": m_a, "base": m_b,
        "held_n": len(both), "held_diff": sum(1 for m in both if run_held[m] != held_b[m]),
        # With a different number of funds held, every month differs, and the count says nothing.
        "held_same_count": bool(both) and (float(np.median([len(run_held[m]) for m in both]))
                                           == float(np.median([len(held_b[m]) for m in both]))),
        "identical": bool(same and np.allclose(a.values, b.values, rtol=0.0, atol=1e-12)),
    }


def ds_prepare(params, run_rules, plan, run_daily, strat_returns, strat_metrics, daily_equity, trade_df,
               rf_monthly, error=None):
    """Everything the Overview's comparison card needs for one finished run.

    Returns a dict. default: the run is itself on the default settings (it is then saved as the default
    run for its period). error: why there is no comparison. changes: what the run changed (ds_changes).
    cmp: ds_compare's result. base: the default run's prices, params and saved entry, for the Reality
    Check's nearby settings (ds_nearby). fresh: the default run was run just now, not read from the saved runs.
    """
    state = {"on": True, "default": False, "error": error, "changes": [], "cmp": None, "base": None,
             "fresh": False}
    base_rules = ds_default_backfill_rules()
    end = params.get("backtest_end", "")
    own = ds_default_params(params["backtest_start"], end)
    state["default"] = ds_signature(params, run_rules) == ds_signature(own, base_rules)
    if state["default"]:
        # A default run is the default run for its period: save it, so a changed setting finds it.
        try:
            if (plan is not None and plan["share"] and len(strat_returns.dropna()) > 0
                    and not ds_month_closing(run_daily[plan["cols"]])):
                _rc_k, key = ds_keys(run_daily[plan["cols"]], own)
                cache = ds_cache_load()
                if key not in cache:
                    cache[key] = ds_entry(strat_returns, trade_df, daily_equity, rf_monthly)
                    ds_cache_save(cache)
        except Exception:
            pass
        return state
    if error or plan is None:
        state["error"] = error or "Its prices were not prepared."
        return state
    # The default settings cannot start before their own lookback and volatility window allow, or before
    # FIRST_HOLD_FLOOR (earliest_backtest_start applies both).
    start = max(str(params["backtest_start"]),
                earliest_backtest_start(MAIN_LOOKBACK_OPTIONS[DS_MAIN_LOOKBACK][0], DS_VOL_MONTHS))
    base_params = ds_default_params(start, end)
    state["changes"] = (ds_changes(params, run_rules, base_params, base_rules)
                        or ["A setting this card does not name"])
    base_daily, base_monthly = ds_finish_prices(plan, run_daily)
    rf_base = base_monthly["BIL"].pct_change() if "BIL" in base_monthly.columns else None
    entry, fresh, rc_k = ds_get_run(base_daily, base_monthly, base_params, rf_base)
    state["fresh"] = fresh
    state["base"] = {"daily": base_daily, "monthly": base_monthly, "params": base_params, "rf": rf_base,
                     "entry": entry, "rc_key": rc_k}
    state["cmp"] = ds_compare(strat_returns, daily_equity, strat_metrics, ds_held(trade_df), entry,
                              rf_monthly, rf_base)
    if state["cmp"] is None:
        state["error"] = "The default settings have no results in the months this run covers."
    return state


def ds_nearby(state, A, progress=None):
    """The Reality Check's nearby settings for both runs: the average Sortino over the main lookback a month
    either way on each leg, traded on the same day and the days either side (rc_plan's "block", up to 27
    runs each). This run's average is the panel's own (A["near_S"]). The default run's block goes through
    the Reality Check's saved runs and worker pool, under the keys a default run's own check uses, so
    whichever comes first saves the other the work. Returns {"run", "base", "n_run", "n_base", "fresh"}.
    """
    base = state["base"]
    plan = rc_plan(base["params"])
    specs = list(dict.fromkeys(plan["block"]))
    cache = rc_cache_load()
    if base["rc_key"] not in cache:
        cache[base["rc_key"]] = {k: base["entry"][k] for k in ("r", "S", "C", "D", "t")}
    runs, stats = rc_compute_other(base["daily"], base["monthly"], base["params"], specs, base["rf"],
                                   cache=cache, progress=progress)
    if stats["fresh"]:
        rc_cache_save(cache)
    vals = [float(runs[s]["S"]) for s in specs if s in runs and np.isfinite(runs[s]["S"])]
    return {"run": float(A["near_S"]), "base": float(np.mean(vals)) if vals else np.nan,
            "n_run": int(A["block_n"]), "n_base": len(vals), "fresh": int(stats["fresh"])}


# ---- drawing the card -------------------------------------------------------

def ds_change_pill(a, b, kind, band, better):
    """The change from the default run's figure (b) to this run's (a) as a pill: green when it is better by
    at least the row's band, red when worse by as much, gray inside the band or for a row with no better way."""
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return ""
    # The change is worked out from the two figures as the card shows them (two decimals), so the
    # arithmetic on the card always adds up.
    if kind == "dd":                      # drawdowns are negative; shallower is better
        d = round(round(abs(b) * 100, 2) - round(abs(a) * 100, 2), 2)
        text = "same" if d == 0 else f"{abs(d):.2f} pts {'shallower' if d > 0 else 'deeper'}"
    elif kind == "pct":
        d = round(round(a * 100, 2) - round(b * 100, 2), 2)
        text = "same" if d == 0 else ae_neg(f"{d:+.2f} pts")
    else:
        d = round(round(a, 2) - round(b, 2), 2)
        text = "same" if d == 0 else ae_neg(f"{d:+.2f}")
    if better is None or band is None or abs(d) < band:
        return ae_pill(text, "neutral")
    return ae_pill(text, "good" if (d > 0) == bool(better) else "bad")


def ds_verdict(a, b, near=None, band=DS_BAND):
    """The card's headline read of this run's Sortino (a) against the default run's (b): (tone, pill text,
    sentence). near is (this run's nearby average, the default run's) once the Reality Check has them.

    As tested alone, a change of at least `band` reads higher or lower. With the nearby averages, it reads
    as a real difference only when they moved the same way by at least DS_AGREE of the change as tested
    (and by at least `band`); otherwise the change is put down to the exact setting. Changes are worked
    out from the figures as the card shows them (two decimals), like the pills.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return "neutral", "No Sortino to compare", "One of the two runs has no Sortino ratio, so there is no read."
    d = round(round(a, 2) - round(b, 2), 2)
    flat = abs(d) < band
    same_text = f"Sortino is about the same as the default settings as tested ({_rc_f2(b)} against {_rc_f2(a)})"
    moved_text = (f"Sortino is {_rc_f2(abs(d))} {'higher' if d > 0 else 'lower'} than the default settings as "
                  f"tested ({_rc_f2(b)} to {_rc_f2(a)})")
    if near is None or not (np.isfinite(near[0]) and np.isfinite(near[1])):
        if flat:
            return "neutral", "About the same", same_text + "."
        if d > 0:
            return "good", "Higher Sortino as tested", moved_text + "."
        return "bad", "Lower Sortino as tested", moved_text + "."
    dn = round(round(near[0], 2) - round(near[1], 2), 2)
    near_text = f"the average of nearby settings moved {_rc_signed(dn)} ({_rc_f2(near[1])} to {_rc_f2(near[0])})"
    if flat:
        if abs(dn) < band:
            return "neutral", "About the same", f"{same_text}, and {near_text}."
        return ("neutral", "Same as tested, nearby settings " + ("better" if dn > 0 else "worse"),
                f"{same_text}, but {near_text}.")
    if (dn > 0) == (d > 0) and abs(dn) >= max(band, DS_AGREE * abs(d)):
        return (("good", "Better, and nearby settings agree") if d > 0 else ("bad", "Worse, and nearby settings agree")) + (
            f"{moved_text}, and {near_text}, so this looks like a real difference and not only the exact setting.",)
    return ("warn", ("Better" if d > 0 else "Worse") + " as tested only",
            f"{moved_text}, but {near_text}, so most of the {'gain' if d > 0 else 'drop'} is the luck of the "
            f"exact setting.")


def ds_card_html(state, near=None, rc="off"):
    """The Overview's comparison card as one HTML string (no blank lines, so Markdown leaves it alone).

    near: ds_nearby's result, once the Reality Check has finished. rc: "off" (the Reality Check is off for
    this run), "wait" (it is still running), "done" or "failed".
    """
    esc = ae_esc
    head = f'<span class="ae-card-l">{esc(DS_TITLE)}</span>'
    if state.get("default"):
        return ('<div id="ov-defaults" class="ae-card ae-anchor" style="margin:0.5rem 0 0.2rem 0;padding:12px 16px">'
                f'<div class="ae-row">{ae_pill("This is the default run", "neutral")}'
                '<span class="ae-small" style="flex:1 1 320px">Every setting is at its default, so this is the run '
                'that other runs are compared with. Change a setting and run again, and the two appear side by '
                'side here.</span></div></div>')
    cmp = state.get("cmp")
    if not cmp:
        why = state.get("error") or "The default settings could not be run for this period."
        return ('<div id="ov-defaults" class="ae-card ae-anchor" style="margin:0.5rem 0 0.2rem 0">'
                f'{head}<div class="ae-small">No comparison with the default settings for this run. {esc(why)}</div></div>')
    m_a, m_b = cmp["run"], cmp["base"]

    def num(m, key):
        v = m.get(key, np.nan)
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else np.nan

    def show(v, kind):
        if not np.isfinite(v):
            return "N/A"
        return ae_neg(f"{v:.2f}") if kind == "ratio" else ae_neg(f"{v:.2%}")

    pair = None
    if near and np.isfinite(near.get("run", np.nan)) and np.isfinite(near.get("base", np.nan)):
        pair = (near["run"], near["base"])
    tone, label, text = ds_verdict(num(m_a, "Sortino Ratio"), num(m_b, "Sortino Ratio"), pair)
    if cmp.get("identical"):
        tone, label = "neutral", "Identical"
        text = ("Every month's return is the same as on the default settings, so the change made no difference "
                "over this period.")
    else:
        d_dd = (round(abs(num(m_b, "Max Drawdown (Daily)")) * 100, 2)
                - round(abs(num(m_a, "Max Drawdown (Daily)")) * 100, 2))
        if np.isfinite(d_dd) and abs(d_dd) >= 0.5:
            text += f" The worst daily drawdown is {abs(d_dd):.1f} points {'shallower' if d_dd > 0 else 'deeper'}."
        if pair is None and rc == "off":
            text += (" One run's gap can be the luck of the exact setting. Turn on the Reality Check (sidebar) "
                     "and this card also compares the two runs' nearby settings.")
    rows = []
    for name, key, kind, band, better in DS_ROWS:
        a, b = num(m_a, key), num(m_b, key)
        rows.append((name, show(b, kind), show(a, kind), ds_change_pill(a, b, kind, band, better), key in DS_BOLD))
        if key == "Sortino Ratio" and pair:
            rows.append(("Sortino, average of nearby settings", _rc_f2(pair[1]), _rc_f2(pair[0]),
                         ds_change_pill(pair[0], pair[1], "ratio", DS_BAND, True), False))
    table = ['<div class="ae-tbl" style="margin:0"><table><thead><tr><th>Figure</th>'
             f'<th>{esc(DS_COLUMN)}</th><th>This run</th><th>Change</th></tr></thead><tbody>']
    for name, b_txt, a_txt, pill, bold in rows:
        table.append(('<tr class="ae-bold">' if bold else "<tr>")
                     + f'<td class="ae-k">{esc(name)}</td><td>{esc(b_txt)}</td>'
                       f'<td class="ae-strat">{esc(a_txt)}</td><td>{pill}</td></tr>')
    table.append("</tbody></table></div>")
    span = f"{cmp['first']:%B %Y} to {cmp['last']:%B %Y}"
    notes = []
    if cmp["same"]:
        notes.append(f"Both runs cover {span} ({cmp['n']} months) on the same prices.")
    else:
        notes.append(f"The two runs cover different months, so both columns here are worked out on the "
                     f"{cmp['n']} months they share, {span}. The figures elsewhere on this page are for this "
                     f"run's full period.")
    if cmp["held_n"] and cmp.get("held_same_count") and not cmp.get("identical"):
        if cmp["held_diff"] == 0:
            notes.append("They held the same funds in every month.")
        else:
            notes.append(f"They held different funds in {cmp['held_diff']} of {cmp['held_n']} months.")
    notes.append(f"Default settings: {ds_default_line()}.")
    if pair:
        notes.append(f"Nearby settings: the main lookback a month either way on each leg, with the month-end "
                     f"trade on the same day and the days either side ({near['n_run']} backtests for this run, "
                     f"{near['n_base']} for the default settings), from the Reality Check. A quick read, not "
                     f"the 90% odds test.")
    elif rc == "wait" and cmp["same"]:
        notes.append("The Reality Check is still running. When it finishes, this card adds the two runs' "
                     "nearby settings.")
    notes.append("The default run is saved and reused until its prices change.")
    chips = "".join(ae_tag(c) for c in state.get("changes", []))
    return ('<div id="ov-defaults" class="ae-card ae-anchor" style="margin:0.5rem 0 0.2rem 0">'
            f'<div class="ae-row" style="justify-content:space-between">{head}'
            f'{ae_pill(label, tone, RC_GLYPHS.get(tone))}</div>'
            f'<div class="ae-row"><span class="ae-small" style="font-weight:600;color:#1C1B19">What changed</span>'
            f'{chips}</div>'
            f'<div style="font-size:15px;line-height:1.55;color:#1C1B19">{esc(text)}</div>'
            + "".join(table)
            + '<div class="ae-small" style="display:flex;flex-direction:column;gap:3px">'
            + "".join(f"<div>{esc(n)}</div>" for n in notes) + "</div></div>")


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

st.sidebar.title("Backtest Parameters")

# ---- RUN BACKTEST (first position, top of the sidebar) ----
# Moved here from below the G1 Cash Rule section November 13, 2026, so it is
# in view without scrolling. The script still reads every control below on
# the same run, so the button's place does not change what a run uses.
run_button = st.sidebar.button("Run Backtest", type="primary", width="stretch")
# Default settings comparison (added November 20, 2026; see the section above
# the sidebar). Ticked at startup. Unticked, none of it runs or shows.
ds_on = st.sidebar.checkbox(
    "Compare with the default settings",
    value=True,
    key="ds_on",
    help=(
        "After a run with a changed setting, the same period is also run on "
        "the default settings, and the Overview tab shows the two side by "
        "side with the change in each figure. The default run takes a few "
        "seconds the first time for a period and is saved after that. With "
        "the Reality Check on, the two runs' nearby settings are compared "
        "too (about 27 more backtests the first time, saved after that)."
    ),
)
# Extra comparison ticker (added November 21, 2026). Blank at startup: nothing
# is downloaded or drawn for it. See "Extra comparison ticker" above
# ae_series_set.
cmp_ticker = cmp_clean(st.sidebar.text_input(
    "Compare with another ticker (optional)",
    value="",
    key="cmp_ticker",
    max_chars=20,
    placeholder="Blank for none; for example QQQ",
    help=(
        "Type any ticker Yahoo Finance carries (QQQ, VTI, BRK-B, BTC-USD), "
        "then press Run Backtest. It is added as one more line on the Growth, "
        "Drawdowns and Annual returns charts and one more column in the "
        "Performance Summary, beside the S&P 500, 60/40 and 80/20. It is never "
        "part of the backtest: the strategy's results are the same with or "
        "without it. A ticker with a shorter history than the backtest starts "
        "where its prices begin, and with leverage on it carries the "
        "benchmarks' leverage. Leave the box blank for none."
    ),
))
# Its color: a preset or a typed hex code. Drawn only while the box holds a ticker, so the
# sidebar at startup is the one box. A valid hex code wins over the preset.
cmp_color = CMP_COLOR
if cmp_ticker:
    _cmp_c1, _cmp_c2 = st.sidebar.columns([1.2, 1])
    _cmp_pick = _cmp_c1.selectbox(
        "Its color", list(CMP_PALETTE), index=0, key="cmp_color_pick",
        help="The color of the added ticker wherever it is drawn. Pick a preset here, or type a hex "
             "code in the box beside it.",
    )
    _cmp_typed = _cmp_c2.text_input(
        "Or a hex code", value="", key="cmp_color_hex", max_chars=7, placeholder="#2B2F36",
        help="Any hex color code, such as #1F9D6B or #2a7. When it is filled in it is used in place "
             "of the preset. Leave it blank to use the preset.",
    )
    cmp_color = CMP_PALETTE.get(_cmp_pick, CMP_COLOR)
    _cmp_says = f"{_cmp_pick} ({cmp_color})"
    if _cmp_typed.strip():
        if cmp_hex(_cmp_typed):
            cmp_color = cmp_hex(_cmp_typed)
            _cmp_says = f"Hex code {cmp_color}"
        else:
            _cmp_says = f"Not a hex color code, so {_cmp_pick} ({cmp_color}) is used"
    st.sidebar.markdown(
        '<div class="ae-cmp-swatch" style="display:flex;align-items:center;gap:8px;font-size:12.5px;'
        'color:#6B6862;margin:-4px 0 6px 0;">'
        f'<span style="flex:0 0 28px;height:11px;border-radius:3px;background:{cmp_color};"></span>'
        f"<span>{ae_esc(_cmp_says)}</span></div>",
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")

# ---- MAIN STRATEGY (second position) ----
st.sidebar.header("Main Strategy")

main_lookback_label = st.sidebar.selectbox(
    "Main momentum lookback (for final ranking)",
    options=list(MAIN_LOOKBACK_OPTIONS.keys()),
    index=list(MAIN_LOOKBACK_OPTIONS.keys()).index(DS_MAIN_LOOKBACK),
)
main_lookbacks, main_lookback_weights = MAIN_LOOKBACK_OPTIONS[main_lookback_label]

final_top_n = st.sidebar.radio(
    "How many assets to hold in the final portfolio?",
    options=[2, 3, 4, 5, 6],
    index=[2, 3, 4, 5, 6].index(DS_FINAL_TOP_N),
    horizontal=True,
)

# Any of the 21 trade days the Reality Check's Full check charts (rc_exec_label
# names each), earliest first (November 17, 2026). The value is the signal day k
# that run_backtest's params["signal_offset"] reads: 1 is the last trading day,
# 0 the first trading day of next month. Those two keep their original labels
# and run through the original "last_day" and "first_day" paths with no
# signal_offset in the params, so their results and saved runs are what they
# were. Any other day sets exec_timing "offset" and params["signal_offset"].
EXEC_TIMING_OPTIONS = {rc_exec_label(_k): _k for _k in RC_WIDE_DAYS}
exec_timing_label = st.sidebar.selectbox(
    "Trade execution timing",
    options=list(EXEC_TIMING_OPTIONS.keys()),
    index=list(EXEC_TIMING_OPTIONS.values()).index(DS_EXEC_OFFSET),
    help=(
        "The day the month-end trade is made, at that day's close. The "
        "allocation is computed from the close of the trading day before "
        "the trade, with each lookback running from earlier month-end "
        "closes to that close, so there is no look-ahead. "
        "Last trading day of month: the default, and the one that matches "
        "real-world execution (allocation from the penultimate close, trade "
        "near the final close). "
        "First trading day of next month: the allocation uses the month-end "
        "close and the trade is made at the next trading day's close "
        "(1-day lag). "
        "The other days are the ones the Reality Check charts, from the "
        "19th-to-last trading day of the month to the 2nd trading day of "
        "next month."
    ),
)
exec_offset = EXEC_TIMING_OPTIONS[exec_timing_label]
exec_timing = {1: "last_day", 0: "first_day"}.get(exec_offset, "offset")

sizing_label = st.sidebar.selectbox(
    "Position sizing",
    options=list(SIZING_OPTIONS.keys()),
    index=list(SIZING_OPTIONS.values()).index(SIZING_DEFAULT),
    key="position_sizing",
    help=(
        "How the final holdings are weighted. Equal risk contribution (ERC), the "
        "default since November 13, 2026: BIL keeps its inverse-volatility weight, "
        "and the other holdings split the rest so each carries the same share of "
        "that group's risk, counting how they move together. A month with too "
        "little data uses inverse volatility, and the results say how often. "
        "Inverse volatility (simple risk parity), the method until then: each "
        "holding's weight is in proportion to 1 over its volatility, so a fund "
        "half as volatile gets twice the weight, and correlations are ignored."
    ),
)
sizing = SIZING_OPTIONS[sizing_label]

vol_months = st.sidebar.slider(
    "Volatility window (months)",
    min_value=3,
    max_value=24,
    value=DS_VOL_MONTHS,
    step=1,
    help=(
        "The months of daily returns, up to the signal date, that the position "
        "weights read: each holding's volatility for inverse volatility, and the "
        "holdings' volatilities and correlations (their covariance) for ERC. "
        "12 is the standing setting."
    ),
)

cost_per_side_pct = st.sidebar.number_input(
    "Trading cost per side (%)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_COST_PER_SIDE * 100,
    step=0.01,
    format="%.2f",
    help=(
        "Charged at every rebalance on one-way turnover, measured against the "
        "drifted weights, so reweighting held funds counts as well as swaps. "
        "0.05% is the standing assumption (adopted November 1, 2026). Set 0 "
        "to reproduce gross results."
    ),
)
cost_per_side = cost_per_side_pct / 100.0

# ---- REALITY CHECK (added November 16, 2026) ----
with st.sidebar.expander("Reality Check", expanded=True):
    rc_mode_label = st.radio(
        "Neighbor backtests",
        options=list(RC_MODE_OPTIONS.keys()),
        index=list(RC_MODE_OPTIONS.values()).index(RC_MODE_DEFAULT),
        key="rc_mode",
        help=(
            "The Reality Check section repeats the backtest on the settings one "
            "step away (lookback, trade day, position sizing, volatility "
            "window) and shows how much of the result "
            "survives. Quick check: about 70 extra backtests the first time "
            "(about half a minute on this PC). Full check: adds the same "
            "lookbacks on 21 trade days (about 160 more) for the typical "
            "nearby run. Finished backtests are saved, so a repeat run reuses "
            "them. Off: the section is not shown."
        ),
    )
    rc_check_topn = st.checkbox(
        "Also check the holdings count", value=False, key="rc_topn",
        help="One fund more and one fewer than the count above. Adds 2 backtests.",
    )
    rc_check_eq = st.checkbox(
        "Also check the equity lookback", value=False, key="rc_eq",
        help="The equity lookbacks one month away on either leg. Adds about 3 backtests.",
    )
    rc_cmp_on = st.checkbox(
        "Compare with a second basket", value=False, key="rc_cmp",
        help=(
            "Runs the same lookbacks and trade days on a second basket, Basket B "
            "(your basket with funds left out or added), and shows how many of "
            "the paired runs each basket wins. Adds about 27 backtests to a "
            "Quick check and about 189 to a Full check, the first time; they "
            "are saved after that. A quick screen, not the 90% odds test."
        ),
    )
    # Basket B's controls are drawn into this slot further down, below the
    # Equity Sub-Selection section, once the run's own basket is known.
    rc_cmp_slot = st.container()
    st.caption("Runs after each backtest. Saved results are reused until the prices behind them change.")
rc_mode = RC_MODE_OPTIONS[rc_mode_label]
rc_extras = tuple(x for x, on in (("topn", rc_check_topn), ("eq", rc_check_eq)) if on)

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

# ---- G1 CASH RULE ----
# Off by default (added November 7, 2026). The defaults are the tested rule
# (G1_DEFAULTS); every setting can be changed here to explore it.
g1_ui = {"enabled": False}
with st.sidebar.expander("G1 Cash Rule", expanded=False):
    st.caption(
        "In months when the strategy holds BIL (cash), move part of BIL's weight "
        "to an intermediate Treasury fund, but only while Treasury yields are "
        "falling and the fund's main score is above BIL's. The defaults are the "
        "rule tested on November 7, 2026: half of BIL's weight to IEI when the "
        "5-year yield is below its level 63 trading days earlier. That test "
        "rated it Suggestive (a small, consistent gain, earned mostly in the "
        "1998, 2000-01 and 2008 sell-offs), and it is not part of the strategy."
    )

    g1_on = st.checkbox(
        "Enable G1 cash rule",
        value=False,
        key="g1_on",
    )

    if g1_on:
        g1_fund_ui = st.text_input(
            "Fund that takes part of BIL's weight",
            value=G1_DEFAULTS["fund"],
            key="g1_fund",
            help="IEI is the iShares 3-7 Year Treasury Bond ETF. Any fund with "
                 "prices works; it is held only in months the rule acts, and "
                 "does not compete for a basket slot.",
        ).strip().upper() or G1_DEFAULTS["fund"]

        g1_backfill_ui = st.text_input(
            "Its backfill chain (before the fund's launch)",
            value=", ".join(G1_DEFAULTS["backfill"]),
            key="g1_backfill",
            help="IEI_X (a 3 to 7-year Treasury ladder, from January 1969) "
                 "carries IEI before its January 2007 launch. VFITX (Vanguard "
                 "Intermediate-Term Treasury, from October 1991) did until "
                 "November 18, 2026; type it to reproduce the November 7 test. "
                 "Leave blank to use the fund's own prices only. A row for the "
                 "same fund under Backfill Rules takes precedence.",
        )

        g1_fraction_pct = st.slider(
            "Share of BIL's weight moved",
            min_value=5,
            max_value=100,
            value=int(round(G1_DEFAULTS["fraction"] * 100)),
            step=5,
            format="%d%%",
            key="g1_fraction",
            help="The tested rule moves 50%. The November 7 test also ran 25% "
                 "and 100%.",
        )

        g1_yield_gate = st.checkbox(
            "Yield gate: only while the Treasury yield is falling",
            value=G1_DEFAULTS["yield_gate"],
            key="g1_yield_gate",
        )
        g1_yield_ticker = G1_DEFAULTS["yield_ticker"]
        g1_yield_lookback = G1_DEFAULTS["yield_lookback"]
        if g1_yield_gate:
            g1_yield_ticker = st.selectbox(
                "Yield",
                options=list(G1_YIELD_OPTIONS),
                index=list(G1_YIELD_OPTIONS).index(G1_DEFAULTS["yield_ticker"]),
                format_func=lambda k: G1_YIELD_OPTIONS[k],
                key="g1_yield_ticker",
            )
            g1_yield_lookback = st.number_input(
                "Falling over the last N trading days",
                min_value=5,
                max_value=252,
                value=G1_DEFAULTS["yield_lookback"],
                step=1,
                key="g1_yield_lookback",
                help="The yield at the signal date must be below its level N "
                     "trading days earlier. 63 is about three months.",
            )

        g1_trend_gate = st.checkbox(
            "Trend gate: only while the fund's main score is above BIL's",
            value=G1_DEFAULTS["trend_gate"],
            key="g1_trend_gate",
            help="The fund's score on the main lookback blend, computed as for "
                 "the basket. In 2022-2023 this gate kept the rule out of "
                 "intermediate Treasuries while they were still falling.",
        )

        g1_ui = {
            "enabled": True,
            "fund": g1_fund_ui,
            "backfill": [t.strip().upper() for t in g1_backfill_ui.split(",") if t.strip()],
            "fraction": g1_fraction_pct / 100.0,
            "yield_gate": bool(g1_yield_gate),
            "yield_ticker": g1_yield_ticker,
            "yield_lookback": int(g1_yield_lookback),
            "trend_gate": bool(g1_trend_gate),
        }
        st.caption("Rule: " + g1_label(g1_ui) + ".")

# ---- PERIOD (third position) ----
st.sidebar.markdown("---")
st.sidebar.header("Backtest Period")

# The start box is blank at startup since November 18, 2026 (Tyler's decision
# that day; it was pre-filled with 1995-06 before). Blank means the earliest
# cleared month for the lookback and volatility window chosen above
# (earliest_backtest_start, never before FIRST_HOLD_FLOOR), shown under the
# box. A month typed in is used as it is, so 1995-06 runs as it always did; a
# month before FIRST_HOLD_FLOOR is labelled a sensitivity run (November 25,
# 2026).
earliest_start = earliest_backtest_start(main_lookbacks, vol_months)
backtest_start_typed = st.sidebar.text_input(
    "Backtest start (YYYY-MM)",
    value="",
    help=(
        "Leave blank to start at the earliest cleared month for the lookback and volatility window chosen "
        "above (February 1973 at the standing settings). Enter YYYY-MM to start later; results are filtered "
        "to start from that month. A month before February 1973 runs as a sensitivity run."
    ),
    placeholder="e.g. 1995-06",
)
backtest_start = backtest_start_typed.strip() or earliest_start
_sensitivity_note = ("Before February 1973 is a sensitivity run: TLT's stand-in is cleared from February 1973, "
                     "QQQ's rows before February 8, 1971 are a fit, and gold was unusually calm in 1970 and "
                     "1971, while governments still held its official price fixed.")
_sensitivity_run = bool(backtest_start_typed.strip()) and backtest_start < FIRST_HOLD_FLOOR
if not backtest_start_typed.strip():
    st.sidebar.caption(f"Blank starts at {earliest_start}, the earliest cleared month for these settings.")
elif _sensitivity_run:
    st.sidebar.caption(_sensitivity_note)

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
    # Never ask Yahoo for exactly 1970-01-01 (a start typed in 1972 would). Its
    # client turns that date into timestamp 0 for a ticker on UTC time (BTC-USD),
    # reads 0 as no start and returns one month, so IBIT silently lost its
    # Bitcoin history and the short series was cached. 1970-01-01 was a market
    # holiday, so asking from the next day loses nothing. Found by the fresh
    # download test of November 25, 2026.
    if download_start == "1970-01-01":
        download_start = "1970-01-02"
except (ValueError, IndexError):
    download_start = "2011-01-01"

# ---- FIXED BASKET (fourth, open by default since November 13, 2026) ----
st.sidebar.markdown("---")

# IBIT is pre-selected since November 3, 2026. It joins the basket once it has
# enough history (JOIN_WHEN_AVAILABLE), so a 1995 start still runs.
default_fixed = list(DS_FIXED)
selected_fixed = []
with st.sidebar.expander("Fixed Basket Assets", expanded=True):
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

# ---- EQUITY (fifth, open by default since November 13, 2026) ----
default_equity = list(DS_EQUITY)
selected_equity = []
with st.sidebar.expander("Equity Sub-Selection", expanded=True):
    st.caption("Choose which equity ETFs are eligible for monthly ranking.")
    for ticker, label in DEFAULT_EQUITY_ETFS.items():
        checked = st.checkbox(
            f"{ticker} - {label}",
            value=(ticker in default_equity),
            key=f"eq_{ticker}"
        )
        if checked:
            selected_equity.append(ticker)

    if "MNK_SYNTH" in selected_equity:
        try:
            _mnk = load_local_index_data()["MNK_SYNTH"].dropna()
            _mnk_first, _mnk_last = _mnk.index.min(), _mnk.index.max()
            # Earliest backtest start at which MNK can be picked: the engine
            # needs more month-ends of history than the longest lookback
            # (main or equity) before a ticker is eligible.
            _need = max(max(main_lookbacks),
                        max(max(v[0]) for v in EQUITY_LOOKBACK_OPTIONS.values()))
            _mnk_earliest = (_mnk_first.to_period("M") + _need + 1).strftime("%Y-%m")
            # Must match PRICE_SWITCH in "Build Synthetic MNK.py" (first NKD=F print).
            _mnk_switch = pd.Timestamp("2004-02-17")
            st.caption(
                f"**MNK data starts {_mnk_first:%B %d, %Y}** and runs through "
                f"{_mnk_last:%B %d, %Y}.  \n"
                f"**Earliest backtest start with MNK: {_mnk_earliest}** "
                f"(with the {main_lookback_label} main lookback, MNK needs "
                f"{_need} months of history before it can be picked).  \n"
                f"**Most faithful from {(_mnk_switch.to_period('M') + 1).strftime('%Y-%m')}.** "
                f"Before {_mnk_switch:%B %d, %Y} the price moves come from the "
                "Nikkei 225 index at the Tokyo close. From then on they come "
                "from the real dollar Nikkei futures at the Chicago close, the "
                "same as MNK.  \n"
                "Other assets you tick may start later. The data check after "
                "Run Backtest flags any that do.  \n"
                "The data only refreshes when you click Update MNK data below. "
                "Tick MNK in place of DXJ to test the account version of the "
                "Japan slot."
            )
        except Exception:
            st.warning("MNK SYNTH Daily Prices.csv not found. Click "
                       "Update MNK data below to build it.")

        # One-click updater: runs "Build Synthetic MNK.py" with the same
        # Python that runs this dashboard, then reloads so the date above
        # shows the new data. The result message survives the reload via
        # session_state.
        _mnk_msg = st.session_state.pop("mnk_update_msg", None)
        if _mnk_msg:
            _kind, _text, _tail = _mnk_msg
            (st.success if _kind == "ok" else st.error)(_text)
            if _tail:
                st.code(_tail, language="text")

        if st.button("Update MNK data", key="mnk_update_btn",
                     help="Downloads fresh prices and rebuilds the MNK series. "
                          "Takes about 1 to 2 minutes and needs internet."):
            import subprocess
            import sys
            _script = os.path.join(_base_dir, "Build Synthetic MNK.py")
            with st.spinner("Updating MNK data. This takes about 1 to 2 minutes..."):
                try:
                    _proc = subprocess.run(
                        [sys.executable, "-u", _script], cwd=_base_dir,
                        capture_output=True, text=True, timeout=600,
                        encoding="utf-8", errors="replace",
                    )
                    _out = (_proc.stdout or "") + (_proc.stderr or "")
                    _ok = _proc.returncode == 0 and "Wrote " in (_proc.stdout or "")
                except subprocess.TimeoutExpired:
                    _out, _ok = "Stopped after 10 minutes with no finish.", False
                except Exception as _e:  # noqa: BLE001
                    _out, _ok = f"Could not start the updater: {_e}", False
            _lines = [ln for ln in _out.splitlines() if ln.strip()]
            if _ok:
                try:
                    _new_last = load_local_index_data()["MNK_SYNTH"].dropna().index.max()
                    _txt = f"MNK data updated. It now runs through {_new_last:%B %d, %Y}."
                except Exception:
                    _txt = "MNK data updated."
                _fb = [ln for ln in _lines if "FALLBACK" in ln]
                if _fb:
                    _txt += (" Note: the Japan interest-rate site was unreachable, "
                             "so a built-in rate table was used. Still usable; "
                             "update again later for the official series.")
                st.session_state["mnk_update_msg"] = ("ok", _txt, "")
            else:
                st.session_state["mnk_update_msg"] = (
                    "err",
                    "The MNK update did not finish. If the message below "
                    "mentions Yahoo, a timeout, or no data, wait a few "
                    "minutes and click Update MNK data again. Otherwise "
                    "use the manual steps.",
                    "\n".join(_lines[-15:]),
                )
            st.rerun()

        if st.checkbox("Show manual update steps (use if the button fails)",
                       value=False, key="mnk_update_help"):
            st.markdown(
                "**Update the MNK data** (about 1 to 2 minutes, needs internet)\n\n"
                "**1. Open Windows PowerShell.** Right-click the **Start** "
                "button on the taskbar and choose **Terminal** or "
                "**Windows PowerShell**, whichever the menu shows. "
                "No Windows key needed.\n\n"
                "If neither appears: press **Ctrl + Shift + Esc** to open "
                "Task Manager, click **Run new task**, type `powershell`, "
                "and click **OK**.\n\n"
                "**2. Copy this line** with the copy icon at its right edge, "
                "paste it into PowerShell with a right-click or **Ctrl + V**, "
                "and press **Enter**. It works from any folder."
            )
            st.code(
                r'python -u "C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System'
                r'\Backtest Engine\Build Synthetic MNK.py"',
                language="powershell",
            )
            st.markdown(
                "**3. Wait. There are no prompts to answer.** Progress lines "
                "scroll by. It is finished when the last line reads:"
            )
            st.code(
                r"Wrote C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System"
                r"\Backtest Engine\Synthetic MNK Validation Stats.txt",
                language="text",
            )
            st.markdown(
                "**4. Close PowerShell and come back here.** Click "
                "**Run Backtest**. The \"MNK data runs through\" date above "
                "shows the new last date. You do not need **Clear Cache**; "
                "that button only affects the Yahoo price cache.\n\n"
                "**If something goes wrong**\n\n"
                "- *\"python is not recognized\"*: paste this line instead."
            )
            st.code(
                r'& "C:\Users\tyler\AppData\Local\Programs\Python\Python311\python.exe" -u '
                r'"C:\Users\tyler\OneDrive\PROJECTS\Strategy Backtesting System'
                r'\Backtest Engine\Build Synthetic MNK.py"',
                language="powershell",
            )
            st.markdown(
                "- *An error mentioning Yahoo, a timeout, or \"no data\"*: "
                "Yahoo Finance is having a moment. Wait a few minutes and run "
                "step 2 again. Nothing is overwritten until the run succeeds.\n"
                "- *A line saying \"FALLBACK Bank of Japan table\"*: the Japan "
                "interest-rate site was unreachable, so a built-in rate table "
                "was used. The update is still usable. Run it again later for "
                "the official series.\n"
                "- *To see the data-quality checks*: open "
                "`C:\\Users\\tyler\\OneDrive\\PROJECTS\\Strategy Backtesting "
                "System\\Backtest Engine\\Synthetic MNK Validation Stats.txt`."
            )

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
        index=[0, 1, 2, 3, 4].index(DS_EQUITY_TOP_N),
        horizontal=True,
        help="0 = no equity sub-selection (only fixed assets in basket)"
    )

    equity_lookback_label = st.selectbox(
        "Equity momentum lookback",
        options=list(EQUITY_LOOKBACK_OPTIONS.keys()),
        index=list(EQUITY_LOOKBACK_OPTIONS.keys()).index(DS_EQUITY_LOOKBACK),
    )
    equity_lookbacks, equity_lookback_weights = EQUITY_LOOKBACK_OPTIONS[equity_lookback_label]

# ---- BASKET B (Reality Check; added November 17, 2026) ----
# Its controls live in the Reality Check expander (rc_cmp_slot), which comes
# above the Fixed Basket and Equity sections, so they are drawn here, where
# the run's own basket is known. Off unless the checkbox is ticked.
basket_b = None
if rc_cmp_on:
    with rc_cmp_slot:
        if rc_mode == "off":
            st.caption("The Reality Check is off, so there is nothing to compare. Choose Quick check or Full check above.")
        else:
            _b_run_funds = list(dict.fromkeys(
                list(selected_fixed) + (list(selected_equity) if equity_top_n > 0 else [])))
            if "rc_b_out" in st.session_state:
                st.session_state["rc_b_out"] = [t for t in st.session_state["rc_b_out"] if t in _b_run_funds]
            _b_out = st.multiselect(
                "Funds to leave out of Basket B", options=_b_run_funds, key="rc_b_out",
                help="Basket B is your basket without these funds.",
            )
            _b_add_text = st.text_input(
                "Funds to add to Basket B (comma-separated)", value="", placeholder="e.g. SHY, UUP", key="rc_b_add",
            )
            _b_where_opts = ["Fixed basket"] + (["Equity list"] if equity_top_n > 0 else [])
            if "rc_b_to" in st.session_state and st.session_state["rc_b_to"] not in _b_where_opts:
                st.session_state["rc_b_to"] = _b_where_opts[0]
            _b_where = st.radio("Add them to", options=_b_where_opts, key="rc_b_to", horizontal=True)
            basket_b = rc_make_basket_b(
                selected_fixed, selected_equity, equity_top_n, _b_out,
                [t.strip().upper() for t in _b_add_text.split(",") if t.strip()],
                "equity" if _b_where == "Equity list" else "fixed",
            )

# ---- BACKFILL ----
st.sidebar.markdown("---")

# Pre-filled defaults based on Portfolio Visualizer backfill mappings
BACKFILL_DEFAULTS = [
    # The last source in most rows is the fund's older segment, added November
    # 18, 2026 to carry the history back to January 2, 1980 (emerging markets
    # to October 16, 1991): "<NAME> X Daily Prices.csv" and "HGER TR Daily
    # Prices.csv", spliced at the 1981 research panel's dates. Each already
    # carries its fund's fee (see BACKFILL_FEE_ANNUAL), and
    # BACKFILL_SOURCE_FROM keeps NAESX, ^NDX and GOLD out of the years before
    # 1990. Sources, checks and limits: "Data Adjustments.md". QQQ's segment
    # is the exception since November 25, 2026: it is QQQ's whole chain, from
    # 1970-01-02 in the file to the fund's first day. HGER's older segment is
    # HGER_FUT since the same day (below). Since November 25, 2026 most of these
    # files reach back to about 1969 or 1970, and the default start is February
    # 1973 (FIRST_HOLD_FLOOR).
    # HGER's index history carries its 0.68% fee (BACKFILL_FEE_ANNUAL) since
    # November 3, 2026. HGER_FUT before 1991, since November 25, 2026: a daily
    # index built from individual futures contracts in 16 markets on fixed
    # sector weights, with T-bill interest added and HGER's 0.68% a year
    # taken out inside the file, from "HGER FUT Daily Prices.csv" (1969-01-02
    # to 1991-01-02). It holds no energy before November 1979. A run uses its
    # rows only on days the engine's calendar has. Built in History Extension
    # 2026-11-18\HGER 1970 Extension 2026-11-25. HGER_TR, the stand-in it
    # replaced (T-bills + 0.66 x the S&P GSCI Total Return's excess over
    # T-bills, less 0.68%, scaled to QCI_SYNTH's daily volatility), stays
    # beside the engine: type "QCI, QCI_SYNTH, HGER_TR" in this row to use it.
    ("HGER",    "QCI, QCI_SYNTH, HGER_FUT"),
    # BTAL_NAV: BTAL's daily NAV with distributions reinvested, from "BTAL NAV
    # Daily Prices.csv" (AGF's NAV file), for September 12, 2011 to 2018, when
    # BTAL's own closes are replaced (PRIMARY_HISTORY_FROM). Added November 4,
    # 2026. BTAL_SYNTH before launch: T-bills + 0.733 x BAB - 0.600 x the S&P
    # 500's return over T-bills, fitted to BTAL 2011-2025, from "BTAL SYNTH
    # Daily Prices.csv". BTAL_SYNTH replaced the raw BAB factor, which is beta
    # neutral and so lacked BTAL's short-market tilt, November 3, 2026.
    # BTAL_X before 1990: BTAL_SYNTH's formula carried back to 1980.
    # Since November 18, 2026 (Tyler's decision that night) BTAL_JKP carries
    # every day before launch, in place of BTAL_SYNTH and BTAL_X: T-bills +
    # 0.873 x the Jensen-Kelly-Pedersen daily US low-beta factor (Frazzini-
    # Pedersen beta, capped value weights), less 1.4% a year for fund costs,
    # from "BTAL JKP Daily Prices.csv" (1968-12-31 to its splice date,
    # 2011-09-12). Out of sample it tracks real BTAL at 0.92 monthly (BTAL_SYNTH
    # 0.78), and BTAL's own index back-test (2001-2011) at 0.97 (0.87). The
    # 1.4% is in the file, so no fee is charged. JKP's data is CC BY-NC 4.0:
    # Jensen, Kelly and Pedersen, "Is There a Replication Crisis in Finance?",
    # Journal of Finance 2023, jkpfactors.com. BTAL_SYNTH and BTAL_X stay
    # beside the engine as research brackets: type "BTAL_NAV, BTAL_SYNTH,
    # BTAL_X" in this row to get the history used until then. Evidence:
    # History Extension 2026-11-18\Step 9 BTAL Deep Dive\BTAL Deep Dive.md
    ("BTAL",    "BTAL_NAV, BTAL_JKP"),
    # TBILL_SYNTH: 13-week T-bill total return less BIL's fee, from
    # "TBILL SYNTH Daily Prices.csv". Replaced VFISX, a 1-3 year Treasury
    # fund, November 3, 2026. BIL is also the risk-free rate. TBILL_X before
    # 1990: the same build from 1969.
    ("BIL",     "TBILL_SYNTH, TBILL_X"),
    # TLT_SYNTH: 20-30 year par Treasuries priced off the Federal Reserve's
    # zero-coupon curve, from "TLT SYNTH Daily Prices.csv". Replaced VUSTX
    # (10+ year bonds, about 15% less duration) November 3, 2026. TLT_X before
    # January 31, 1990: a long Treasury priced off Yahoo's 30-year yield
    # (the Federal Reserve curve has no 30-year point before November 1985),
    # and before February 16, 1977 mainly off the 20-year yield (TLT 1970
    # Extension 2026-11-25, in History Extension 2026-11-18).
    ("TLT",     "TLT_SYNTH, TLT_X"),
    # GLDM: GLD's history before GLDM's June 26, 2018 launch, then GOLD (the
    # daily gold price; dates corrected November 7, 2026) before GLD's
    # November 18, 2004 launch, both fee-adjusted to GLDM's 0.10%
    # (BACKFILL_FEE_ANNUAL). The default since November 3, 2026.
    # Keep this row above GLD's: rules run in order, and GLDM's chain must
    # read GLD's own prices before GLD's rule extends them with GOLD.
    # GOLD_X before 1990: the same gold price with a 1984 bad print repaired;
    # before the GOLD file begins in June 1980, spot gold worked out from COMEX
    # futures (1974-12-31 to 1980-06-24) and Investing.com's spot before that.
    ("GLDM",    "GLD, GOLD, GOLD_X"),
    # GLD is no longer a ticked option; this rule serves a GLD typed into the
    # custom tickers box.
    ("GLD",     "GOLD"),
    # QQQ: QQQ_X alone since November 25, 2026 ("QQQ X Daily Prices.csv", to
    # March 10, 1999). In the file: the Nasdaq-100 price index with no credit
    # from February 14, 1994; the index plus 0.56% a year from its first day
    # (January 31, 1985); and 1.3477 x the Nasdaq Composite's daily move plus
    # 0.56% before it, back to 1970 (a stand-in before the Composite began in
    # February 1971). It replaced "RYOCX, ^NDX, QQQ_X" (RYOCX: Rydex OTC, a
    # Nasdaq-100 index fund, from February 1994), which still works if typed.
    # FSPTX, an actively managed tech fund, was replaced November 3, 2026.
    ("QQQ",     "QQQ_X"),
    # NAESX: Vanguard Small-Cap Index, which tracked the Russell 2000 until 2003.
    # Replaced OTCFX, an actively managed fund that ran well ahead of the index,
    # November 3, 2026 (see NAESX Impact Check.py in Basket Research\ETF Scouting).
    # IWM_X before 1990: a small-company series fitted to the Russell 2000
    # (NAESX's own 1980s prices miss distributions).
    ("IWM",     "NAESX, IWM_X"),
    # EMXC_SYNTH: synthetic MSCI EM ex China, 2004-10 to EMXC's July 2017
    # launch, from "EMXC SYNTH Daily Prices.csv" (Build Synthetic EMXC.py).
    # Replaced the EEM-only backfill October 3, 2026, because EEM holds China.
    # EMXC_X before May 4, 1994: Templeton Developing Markets (TEDMX), from
    # October 16, 1991. Nothing earlier, so EMXC joins late from any start
    # before mid-1992 (JOIN_WHEN_AVAILABLE_EQUITY); before 1993 DEVXJ_X takes
    # its place among
    # the equity candidates (EQUITY_SEAT_BEFORE).
    ("EMXC",    "EMXC_SYNTH, EEM, VEIEX, EMXC_X"),
    # DXJ_HEDGED: DXJ's own unhedged returns before its April 1, 2010 change
    # to a yen-hedged fund and EWJ's from November 1996, converted to
    # yen-hedged, with MNK_SYNTH before that; from "DXJ HEDGED Daily
    # Prices.csv". DXJ's own prices before 2010-04-01 are replaced
    # (PRIMARY_HISTORY_FROM). Replaced FJPNX, an actively managed unhedged
    # fund, November 3, 2026. DXJ_X before February 1993: the Nikkei 225 with
    # dividends, yen-hedged at the Japan and US short rates.
    ("DXJ",     "DXJ_HEDGED, DXJ_X"),
    # BTC-USD: Bitcoin's own price before IBIT's January 11, 2024 launch, less
    # IBIT's 0.25% fee (BACKFILL_FEE_ANNUAL). For stress tests, the "Bitcoin
    # history" switch in Backfill Rules (BTC_HISTORY_OPTIONS) puts BTC_TAMED in
    # its place: Bitcoin from January 2012 (Bitstamp, then BTC-USD) with
    # 2012-2017 given 2021-2026's volatility and growth per unit of risk and
    # BTC-USD unchanged from 2018, from "BTC TAMED Daily Prices.csv" (built by
    # Basket Research\ETF Scouting\Tamed Bitcoin History 2026-11-04\Build Tamed
    # Bitcoin.py). IBIT then joins in October 2012 from a 1995 start. Not the
    # default. Added November 4, 2026; the switch the same night.
    ("IBIT",    "BTC-USD"),
    ("",        ""),
    ("",        ""),
]

# The "Bitcoin history" switch at the top of Backfill Rules: each choice and
# the chain it puts in the IBIT row. Real (None) leaves the row as typed,
# BTC-USD by default. Tamed shows BTC_TAMED in the row, greyed out, and uses
# it; switching back to Real resets the row to BTC-USD. Added November 4, 2026.
BTC_HISTORY_OPTIONS = {
    "Real (BTC-USD, from September 2014)": None,
    "Tamed (BTC_TAMED, from January 2012)": "BTC_TAMED",
}

backfill_rules = {}
with st.sidebar.expander("Backfill Rules", expanded=False):
    btc_history = st.radio(
        "Bitcoin history for IBIT, before its January 2024 launch",
        list(BTC_HISTORY_OPTIONS),
        index=0,
        key="btc_history",
        help=(
            "Real: Bitcoin's own price (BTC-USD), from September 2014. Tamed: Bitcoin from "
            "January 2012, with 2012-2017's swings and growth scaled down to 2021-2026's, "
            "for stress tests. Both are charged IBIT's 0.25% fee."
        ),
    )
    btc_history_chain = BTC_HISTORY_OPTIONS[btc_history]
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
            if bf_primary == "IBIT" and btc_history_chain:
                # A key of its own, so the row shows the switch's chain.
                bf_chain = st.text_input(
                    f"Backfill chain {slot}",
                    value=btc_history_chain,
                    key=f"bf_chain_{slot}_btc_history",
                    disabled=True,
                    help="Set by the Bitcoin history switch above.",
                )
            else:
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
# AOR/AOA history extended back to 1969 (via VSMGX/VASGX, then AOR_X/AOA_X) on every
# run so the benchmark lines match the strategy and SPY across the full window.
for _bcfg in BAL_BENCHMARKS:
    _bf_chain = _bcfg.get("backfill")
    if _bf_chain:
        backfill_rules[_bcfg["ticker"]] = [t.strip().upper() for t in _bf_chain]
# The S&P 500 line's history before SPY's first close (BENCHMARK_BACKFILL),
# unless a row above already gives SPY a chain. Added November 18, 2026.
if BENCHMARK_BACKFILL and BENCHMARK_TICKER not in backfill_rules:
    backfill_rules[BENCHMARK_TICKER] = list(BENCHMARK_BACKFILL)

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

# Brand bar (page facelift, November 17, 2026): the Chart Dashboard's wordmark and tag.
_ae_loaded = ae_local_now()
ae_md(
    '<div class="ae-brand-header"><span class="ae-brand-name">Asymmetric Edge</span>'
    '<span class="ae-brand-tag">Backtest Engine</span></div>'
    '<div class="ae-brand-sub">Relative strength momentum across an uncorrelated asset basket &middot; '
    f'loaded {_ae_loaded:%b} {_ae_loaded.day}, {_ae_loaded:%Y}, {_ae_loaded.hour % 12 or 12}:{_ae_loaded:%M} '
    f'{"a.m." if _ae_loaded.hour < 12 else "p.m."}</div>'
)


if run_button:
    if len(selected_fixed) == 0 and equity_top_n == 0:
        st.error("You need at least some assets in the basket. Select fixed assets or enable equity sub-selection.")
    else:
        # Gather ALL tickers needed (including backfills)
        all_tickers_needed = (
            list(selected_fixed) + list(selected_equity)
            + [BENCHMARK_TICKER, "BIL", CALENDAR_TICKER]
            + [c["ticker"] for c in BAL_BENCHMARKS]
        )
        if leverage_enabled and leverage_params_ui.get("method") == "vt_trend" and leverage_params_ui.get("ticker"):
            all_tickers_needed.append(leverage_params_ui["ticker"])
        # G1 cash rule: its fund, and the fund's backfill chain unless a
        # Backfill Rules row already covers that fund.
        if g1_ui.get("enabled"):
            all_tickers_needed.append(g1_ui["fund"])
            _g1_in_basket = g1_ui["fund"] in selected_fixed or g1_ui["fund"] in selected_equity
            if g1_ui["fund"] not in backfill_rules and g1_ui.get("backfill"):
                if _g1_in_basket:
                    # Never change a basket asset's own history from the G1 section.
                    st.info(
                        f"{g1_ui['fund']} is also in the basket, so the G1 cash rule uses its basket "
                        f"history; set its backfill under Backfill Rules."
                    )
                else:
                    backfill_rules[g1_ui["fund"]] = list(g1_ui["backfill"])
        for bf_chain in backfill_rules.values():
            all_tickers_needed.extend(bf_chain)
        # Pass as tuple for cache hashability
        tickers_tuple = tuple(sorted(set(all_tickers_needed)))
        # Basket B (Reality Check; November 17, 2026): its funds download with the
        # run's. A fund Yahoo cannot supply leaves Basket B out, and the run goes on.
        _b_extra = (tuple(sorted(set(rc_basket_b_funds(basket_b)) - set(tickers_tuple)))
                    if basket_b is not None else ())

        # Default settings comparison (November 20, 2026): a default run's own tickers download with
        # the run's when the run's basket or Backfill Rules leave some of them out (a run that has
        # them all downloads what it always did). The download is tried with everything, then without
        # Basket B's funds, then without the default run's, then with the run's tickers alone, as
        # before; whichever is left out says so on its card, and the run goes on.
        _ds_error = None
        _ds_extra = tuple(sorted(set(ds_default_tickers()) - set(tickers_tuple))) if ds_on else ()
        _dl_sets = []
        for _dl_b, _dl_d in ((_b_extra, _ds_extra), ((), _ds_extra), (_b_extra, ()), ((), ())):
            _dl_set = tuple(sorted(set(tickers_tuple) | set(_dl_b) | set(_dl_d)))
            if _dl_set not in [_s for _s, _sb, _sd in _dl_sets]:
                _dl_sets.append((_dl_set, _dl_b, _dl_d))
        daily_prices, download_error, _dl_first_error = None, None, None
        for _dl_set, _dl_b, _dl_d in _dl_sets:
            try:
                daily_prices = download_data(_dl_set, download_start, force_refresh=force_refresh)
            except MissingPriceDataError as exc:
                daily_prices, download_error = None, str(exc)
                _dl_first_error = _dl_first_error or str(exc)
                continue
            download_error = None
            if _b_extra and not _dl_b:
                basket_b["error"] = _dl_first_error
            if _ds_extra and not _dl_d:
                _ds_error, _ds_extra = f"Its prices could not be loaded. {_dl_first_error}", ()
            break

        if daily_prices is None or daily_prices.empty:
            st.error(download_error or "Failed to download price data. Check your internet connection.")
        else:
            # Make a mutable copy (cached data is read-only)
            daily_prices = daily_prices.copy()

            # Default settings comparison (November 20, 2026): the default run's price table is
            # planned here, before the run's own Backfill Rules are applied (ds_plan_prices). The
            # default run's own tickers then leave the table again, so the run reads what it always did.
            _ds_plan = None
            if ds_on and _ds_error is None:
                try:
                    _ds_plan = ds_plan_prices(daily_prices, backfill_rules, _ds_extra)
                except Exception as _ds_exc:
                    _ds_error = f"Its prices could not be prepared ({_ds_exc})."
            if _ds_extra:
                daily_prices = daily_prices.drop(columns=[c for c in _ds_extra if c in daily_prices.columns])

            # Basket B's own funds must not move the main run's trading calendar (a fund on another
            # country's exchange would add days none of the run's funds traded). If one would, Basket B
            # is left out, named on its card, and the price table is the run's alone, as it was before.
            if _b_extra:
                _clash = rc_basket_b_calendar_clashes(daily_prices, _b_extra)
                if _clash:
                    daily_prices = daily_prices.drop(columns=[c for c in _b_extra if c in daily_prices.columns])
                    basket_b["error"] = (", ".join(_clash) + (" prints" if len(_clash) == 1 else " print")
                                         + " closes on days none of your other funds trade, which would "
                                           "change the trade days of your own run.")

            # Apply backfill rules
            backfill_info = []
            for primary, chain in backfill_rules.items():
                if primary in daily_prices.columns or any(bf in daily_prices.columns for bf in chain):
                    spliced, segments = build_backfilled_series(
                        daily_prices, primary, chain,
                        fee_annual=BACKFILL_FEE_ANNUAL.get(primary, 0.0),
                        primary_from=PRIMARY_HISTORY_FROM.get(primary),
                        source_from=BACKFILL_SOURCE_FROM.get(primary),
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

            # An equity fund in JOIN_WHEN_AVAILABLE_EQUITY may start late, as long
            # as enough other equity funds have history to fill the equity picks.
            _eq_on_time = 0
            if equity_top_n > 0:
                for _t in selected_equity:
                    if _t in daily_prices.columns:
                        _fv = daily_prices[_t].dropna().index.min()
                        if _fv is not None and not pd.isna(_fv) and _fv <= required_by:
                            _eq_on_time += 1

            data_issues = []
            late_joiners = []
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
                        if ticker in JOIN_WHEN_AVAILABLE and ticker in selected_fixed:
                            late_joiners.append((ticker, first_valid))
                        elif (ticker in JOIN_WHEN_AVAILABLE_EQUITY and ticker in selected_equity
                              and ticker not in selected_fixed and _eq_on_time >= equity_top_n):
                            late_joiners.append((ticker, first_valid))
                        else:
                            data_issues.append({
                                "Ticker": ticker,
                                "Issue": f"Data starts too late (need {max_lb}mo lookback before {backtest_start})",
                                "Earliest Data": first_valid.strftime("%Y-%m-%d"),
                                "Data Required By": required_by.strftime("%Y-%m"),
                            })

            if data_issues:
                st.error("**Backtest blocked.** The following assets do not have enough data for your selected start date:")
                st.dataframe(pd.DataFrame(data_issues), width="stretch", hide_index=True)

                latest_inception = pd.Timestamp("1900-01-01")
                late_tickers = {lt for lt, _ in late_joiners}
                for ticker in all_selected:
                    if ticker in late_tickers:
                        continue
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
                for _lt, _lt_first in late_joiners:
                    _join = (_lt_first.to_period("M") + max_lb + 1).strftime("%Y-%m")
                    if _lt in selected_fixed:
                        st.info(
                            f"{_lt} has prices from {_lt_first:%B %d, %Y}, so it joins the "
                            f"basket from {_join}, once it has {max_lb} months of history."
                        )
                    elif (_lt in EQUITY_SEAT_BEFORE and EQUITY_SEAT_BEFORE[_lt][0] in daily_prices.columns
                          and str(backtest_start) < EQUITY_SEAT_BEFORE[_lt][1]):
                        _seat_name, _seat_until = EQUITY_SEAT_BEFORE[_lt]
                        _seat_month = pd.Period(_seat_until, "M").strftime("%B %Y")
                        st.info(
                            f"{_lt} has prices from {_lt_first:%B %d, %Y}. Until {_seat_month} its place "
                            f"among the equity candidates is taken by {_seat_name} "
                            f"({ae_asset_name(_seat_name)}); see Data adjustments."
                        )
                    else:
                        st.info(
                            f"{_lt} has prices from {_lt_first:%B %d, %Y}, so the equity step picks "
                            f"from the other equity funds until {_join}, once it has {max_lb} months "
                            f"of history."
                        )
                backfill_summary = "; ".join(
                    [f"{k} <- {', '.join(v)}" for k, v in backfill_rules.items()]
                ) if backfill_rules else "None"

                # ---- G1 CASH RULE (load the yield series once) ----
                g1_config = {"enabled": False}
                if g1_ui.get("enabled"):
                    g1_fund_run = g1_ui["fund"]
                    if g1_fund_run == "BIL":
                        st.warning("The G1 cash rule needs a fund other than BIL; G1 is off for this run.")
                    elif g1_fund_run not in daily_prices.columns or daily_prices[g1_fund_run].dropna().empty:
                        st.warning(
                            f"The G1 cash rule is on, but there are no prices for {g1_fund_run}; "
                            f"G1 is off for this run."
                        )
                    else:
                        g1_yield_series = None
                        if g1_ui["yield_gate"]:
                            with st.spinner(f"Loading {g1_ui['yield_ticker']} for the G1 yield gate..."):
                                g1_yield_ohlc = download_gate_ohlc(
                                    g1_ui["yield_ticker"], download_start, force_refresh=force_refresh
                                )
                            if g1_yield_ohlc is not None and not g1_yield_ohlc.empty:
                                g1_yield_series = g1_yield_ohlc["Close"].dropna()
                        if g1_ui["yield_gate"] and (g1_yield_series is None or g1_yield_series.empty):
                            st.warning(
                                f"Could not load {g1_ui['yield_ticker']} for the G1 yield gate; "
                                f"G1 is off for this run."
                            )
                        else:
                            g1_config = {**g1_ui, "yield_series": g1_yield_series}
                            _g1_first = daily_prices[g1_fund_run].dropna().index.min()
                            if _g1_first > required_by:
                                st.info(
                                    f"{g1_fund_run} has prices from {_g1_first:%B %d, %Y}, so the G1 "
                                    f"cash rule can act only from then."
                                )

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
                    "sizing": sizing,
                    "exec_timing": exec_timing,
                    "exec_timing_label": exec_timing_label,
                    **({"signal_offset": exec_offset} if exec_timing == "offset" else {}),
                    "cost_per_side": cost_per_side,
                    "backtest_start": backtest_start,
                    "backtest_end": backtest_end,
                    "backfill_summary": backfill_summary,
                    "leverage_params": leverage_params_ui,
                    "g1": g1_config,
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

                    # Extra comparison ticker (November 21, 2026): one more entry, marked
                    # "custom", when the sidebar's box holds a ticker. Its prices load apart
                    # from the run's price table (cmp_load), and it gets the benchmarks'
                    # leverage treatment. Left out, with a warning, when it has no prices.
                    cmp_note, cmp_error = "", ""
                    if cmp_ticker:
                        if cmp_ticker == BENCHMARK_TICKER or cmp_ticker in [c["ticker"] for c in BAL_BENCHMARKS]:
                            cmp_error = f"{cmp_ticker} is already one of the benchmarks on every chart"
                        else:
                            _cx_px, _cx_why = cmp_load(cmp_ticker, daily_prices, download_start)
                            if _cx_px is None:
                                cmp_error = _cx_why
                            else:
                                _cx_r = (get_month_end_prices(_cx_px.to_frame(cmp_ticker))[cmp_ticker]
                                         .pct_change().reindex(common_dates))
                                if _cx_r.notna().sum() < 2:
                                    cmp_error = f"{cmp_ticker} has no prices in the months of this backtest"
                                else:
                                    _cx_first, _cx_last = _cx_r.first_valid_index(), _cx_r.last_valid_index()
                                    _cx_late, _cx_early = _cx_first > common_dates[0], _cx_last < common_dates[-1]
                                    _cx_label = cmp_ticker
                                    if _cx_late and _cx_early:
                                        _cx_label += f" ({_cx_first:%b %Y} to {_cx_last:%b %Y})"
                                        cmp_note = (f"{cmp_ticker}'s prices cover {_cx_first:%B %Y} to {_cx_last:%B %Y} only, "
                                                    "so its lines and figures cover a shorter period than the others.")
                                    elif _cx_late:
                                        _cx_label += f" (from {_cx_first:%b %Y})"
                                        cmp_note = (f"{cmp_ticker} joins in {_cx_first:%B %Y}, the first full month it has prices, "
                                                    "so its lines and figures cover a shorter period than the others.")
                                    elif _cx_early:
                                        _cx_label += f" (to {_cx_last:%b %Y})"
                                        cmp_note = (f"{cmp_ticker}'s prices end in {_cx_last:%B %Y}, so its lines and figures "
                                                    "stop there.")
                                    if leverage_enabled:
                                        _cx_r = _cx_r * leverage_mults - borrow_costs
                                    balanced_data.append({
                                        "label": _cx_label, "ticker": cmp_ticker, "color": cmp_color, "linestyle": ":",
                                        "custom": True, "daily_px": _cx_px,
                                        "returns": _cx_r,
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
                            # The extra comparison ticker carries its own prices (cmp_load).
                            _bal_px = entry.get("daily_px")
                            if _bal_px is None:
                                if t not in daily_prices.columns:
                                    continue
                                _bal_px = daily_prices[t]
                            bal_daily = _bal_px.loc[de_start:de_end].dropna()
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

                    # Which position sizing made this run, and how often ERC fell
                    # back to inverse volatility (read from the trade log).
                    sizing_info = describe_sizing(params, trade_df)
                    params["erc_fallback_summary"] = sizing_info["fallback"]

                    try:
                        append_to_excel(params, strat_metrics, bench_metrics, RESULTS_FILE)
                        _ae_logged = True
                    except Exception as e:
                        _ae_logged = False
                        st.warning(f"Could not write to Excel (file may be open): {e}")

                    # =====================================================================
                    # DEFAULT SETTINGS COMPARISON  (November 20, 2026)
                    # =====================================================================
                    # The same period on the default settings, from the saved runs when the prices
                    # behind them have not changed (ds_prepare). It reads this run's results and
                    # changes none of them. A failure here leaves a note on the card, nothing more.
                    ds_state = {"on": False}
                    if ds_on:
                        try:
                            with st.spinner("Comparing with the default settings..."):
                                ds_state = ds_prepare(params, backfill_rules, _ds_plan, daily_prices, strat_returns,
                                                      strat_metrics, daily_equity, trade_df, rf_monthly, _ds_error)
                        except Exception as _ds_exc:
                            ds_state = {"on": True, "default": False, "cmp": None,
                                        "error": f"The comparison could not run ({_ds_exc})."}

                    # =====================================================================
                    # RESULTS PAGE  (page facelift, November 17, 2026)
                    # =====================================================================
                    # Six tabs. Everything below only draws what the backtest above produced;
                    # no figure is computed differently than before. The Reality Check runs
                    # last, so the other tabs appear while it works, and Overview's tiles are
                    # drawn again with its realistic ranges once it finishes.
                    lev_tag = " (lev)" if leverage_enabled else ""
                    _now = ae_local_now()
                    _ran = (f"{_now:%a}, {_now:%b} {_now.day}, {_now:%Y}, {_now.hour % 12 or 12}:{_now:%M} "
                            f"{'a.m.' if _now.hour < 12 else 'p.m.'}")
                    _px_col = daily_prices[BENCHMARK_TICKER].dropna() if BENCHMARK_TICKER in daily_prices.columns else daily_prices
                    _last_px = _px_col.index.max()
                    ae_md('<div class="ae-status"><span><span class="ae-dot"></span>Backtest ran ' + ae_esc(_ran) + "</span>"
                          f"<span>Prices through the {_last_px:%b} {_last_px.day}, {_last_px:%Y} close</span>"
                          + (f"<span>Results logged to {ae_esc(os.path.basename(RESULTS_FILE))}</span>" if _ae_logged else "")
                          + "</div>")

                    # Extra comparison ticker (November 21, 2026): say so when it was left out, and
                    # keep its short-history note for the headings of the places it appears.
                    if cmp_error:
                        st.warning(f"The comparison ticker was left out: {cmp_error}. The backtest itself is not affected.")
                    _cmp_sub = (" " + ae_esc(cmp_note)) if cmp_note else ""

                    tab_ov, tab_rc, tab_sig, tab_rd, tab_hl, tab_td = st.tabs(
                        ["Overview", "Reality Check", "Signals", "Returns and Drawdowns", "Strategy Health", "Trades and Data"])


                    def _ae_num(mm, k):
                        v = mm.get(k, np.nan)
                        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else np.nan


                    def _ae_pct(v, d=1):
                        return ae_neg(f"{v:.{d}%}") if np.isfinite(v) else "N/A"


                    # ---- run notes (formerly the captions under the table and the page's last caption) ----
                    if leverage_params_ui.get("enabled", False):
                        lp = leverage_params_ui
                        if lp.get("method") == "constant":
                            cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                            cost_str = f" Borrowing cost: {cost_pct:.1f}% annual on leveraged portion." if cost_pct > 0 else ""
                            lev_note = (f" Constant {lp.get('mult_constant', 1.25):.2f}x leverage applied "
                                        f"to strategy and benchmarks.{cost_str}")
                        elif lp.get("method") == "holdings_gate":
                            defensives = ", ".join(lp.get("defensive_tickers", []))
                            cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                            cost_str = f" Borrowing cost: {cost_pct:.1f}% annual on leveraged portion." if cost_pct > 0 else ""
                            lev_note = (f" Dynamic leverage: {lp.get('mult_leveraged', 1.25):.2f}x when no "
                                        f"defensive assets ({defensives}) are in the final portfolio, 1.0x otherwise.{cost_str}")
                        elif lp.get("method") == "vt_trend":
                            lev_note = (" Dynamic leverage scales monthly returns by a multiplier "
                                        f"({lp['mult_risk_off']:.2f}x to {lp['mult_risk_on']:.2f}x) "
                                        f"based on {lp['ticker']} trend regime.")
                        else:
                            lev_note = ""
                    else:
                        lev_note = ""
                    g1_note = ""
                    if g1_config.get("enabled"):
                        g1_note = (f" G1 cash rule: {g1_label(g1_config)}, read at the same signal "
                                   f"date as the rest of the allocation (not part of the strategy; "
                                   f"the trade log's g1 column gives each month's outcome).")
                    if exec_timing == "last_day":
                        exec_note = ("Allocations computed from the close of the trading day before "
                                     "the last trading day of each month; trades execute at the last "
                                     "trading day's close (no look-ahead). ")
                    elif exec_timing == "offset":
                        exec_note = ("Allocations computed from the close of the trading day before "
                                     f"the {rc_trade_each(exec_offset)}; trades execute at that day's "
                                     "close (no look-ahead). ")
                    else:
                        exec_note = ("Signals computed at month-end close; trades execute at next "
                                     "trading day's close (1-day lag). ")
                    _avg_turnover = float(results_df["turnover"].loc[common_dates].mean())
                    if cost_per_side > 0:
                        _cost_note = (f"Strategy returns are net of trading costs: {cost_per_side:.2%} per side "
                                      f"on one-way turnover, which averaged {_avg_turnover:.1%} of the "
                                      f"portfolio a month. Benchmarks carry no trading costs.")
                    else:
                        _cost_note = (f"Strategy returns are gross, with no trading costs. One-way turnover "
                                      f"averaged {_avg_turnover:.1%} of the portfolio a month.")
                    _dd_note = ""
                    peak_dt = strat_metrics.get("DD Peak Date")
                    trough_dt = strat_metrics.get("DD Trough Date")
                    if peak_dt is not None and trough_dt is not None:
                        _dd_note = f"Daily: peak on {peak_dt:%B} {peak_dt.day}, {peak_dt:%Y}, trough on {trough_dt:%B} {trough_dt.day}, {trough_dt:%Y}. "
                    worst_day_dt = strat_metrics.get("Worst Day Date")
                    if worst_day_dt is not None:
                        _wd = strat_metrics.get("Worst Day", 0)
                        _wd_txt = ae_neg(f"{_wd:.2%}")
                        _dd_note += f"Worst single day: {_wd_txt} on {worst_day_dt:%B} {worst_day_dt.day}, {worst_day_dt:%Y}."
                    _method_note = ("All prices are dividend- and split-adjusted (total return). " + exec_note
                                    + sizing_info["note"] + " Sortino ratio uses BIL as the risk-free rate." + lev_note + g1_note)

                    # ---- headline figures ----
                    s_sort, s_cagr = _ae_num(strat_metrics, "Sortino Ratio"), _ae_num(strat_metrics, "CAGR")
                    s_dd = _ae_num(strat_metrics, "Max Drawdown (Daily)")
                    if not np.isfinite(s_dd):
                        s_dd = _ae_num(strat_metrics, "Max Drawdown")
                    s_mar = _ae_num(strat_metrics, "Martin Ratio")
                    _b8020 = next((e for e in balanced_data if e["label"] == "80/20" and e.get("metrics")), None)
                    _cmp_label = ("80/20" + lev_tag) if _b8020 else ("S&P 500" + lev_tag)
                    _cmp_m = _b8020["metrics"] if _b8020 else bench_metrics


                    def _ae_dd(mm):
                        v = _ae_num(mm, "Max Drawdown (Daily)")
                        return v if np.isfinite(v) else _ae_num(mm, "Max Drawdown")


                    _n_m = len(strat_returns)
                    _span = f"{_n_m / 12:.0f} years" if _n_m >= 24 else f"{_n_m} months"
                    _p0, _p1 = strat_returns.index[0], strat_returns.index[-1]
                    _eyebrow = (f"Backtest · {_p0:%B %Y} to {_p1:%B %Y} · "
                                + ("net of trading costs" if cost_per_side > 0 else "gross, no trading costs")
                                + (" · leveraged" if leverage_enabled else ""))
                    _spy_cagr, _spy_dd = _ae_num(bench_metrics, "CAGR"), _ae_dd(bench_metrics)
                    if _b8020:
                        _desc = (f"{_ae_pct(s_cagr)} a year against {_ae_pct(_ae_num(_b8020['metrics'], 'CAGR'))} for the 80/20 "
                                 f"portfolio and {_ae_pct(_spy_cagr)} for the S&amp;P 500, while the deepest daily drawdown was "
                                 f"{abs(s_dd):.1%} against their {abs(_ae_dd(_b8020['metrics'])):.1%} and {abs(_spy_dd):.1%}.")
                    else:
                        _desc = (f"{_ae_pct(s_cagr)} a year against {_ae_pct(_spy_cagr)} for the S&amp;P 500, while the deepest "
                                 f"daily drawdown was {abs(s_dd):.1%} against its {abs(_spy_dd):.1%}.")
                    if leverage_enabled:
                        _desc += " The benchmarks carry the same leverage."
                    if rc_mode != "off":
                        _desc += " The realistic ranges come from the Reality Check."


                    def _ae_tiles_html(rc_tiles=None, rc_note=""):
                        """The four headline tiles; the bottom line carries the Reality Check's range once it has run."""
                        by = {}
                        for t in rc_tiles or []:
                            nm = str(t.get("name", "")).lower()
                            key = "sortino" if nm.startswith("sortino") else "cagr" if nm.startswith("return") else \
                                  "dd" if ("drawdown" in nm or nm.startswith("worst")) else nm
                            by[key] = t

                        def bottom(key, fallback):
                            t = by.get(key)
                            if t:
                                return (f'<div class="ae-row"><b>Realistic range {ae_esc(t["range"])}</b>{_rc_pill_html(t["pill"])}</div>'
                                        f'<div>{ae_esc(t["note"])}</div>')
                            return ae_esc(fallback)

                        spec = [
                            ("Sortino ratio", f"{s_sort:.2f}", _ae_num(_cmp_m, "Sortino Ratio"), "ratio", "sortino", True),
                            ("Annual growth rate", _ae_pct(s_cagr), _ae_num(_cmp_m, "CAGR"), "pct", "cagr", True),
                            ("Max drawdown, daily", _ae_pct(s_dd), _ae_dd(_cmp_m), "dd", "dd", True),
                            ("Martin ratio", f"{s_mar:.2f}", _ae_num(_cmp_m, "Martin Ratio"), "ratio", "martin", True),
                        ]
                        tiles = []
                        for label, val, bval, kind, key, hb in spec:
                            b_txt = (f"{bval:.2f}" if kind == "ratio" else _ae_pct(bval)) if np.isfinite(bval) else "N/A"
                            s_raw = {"sortino": s_sort, "cagr": s_cagr, "dd": s_dd, "martin": s_mar}[key]
                            pill = ae_gap_pill(s_raw, bval, kind, hb)
                            fb = ("Return over the depth and length of every drawdown." if key == "martin" else rc_note)
                            tiles.append(f'<div class="ae-tile"><div class="ae-tile-l">{ae_esc(label)}</div>'
                                         f'<div class="ae-tile-v">{ae_esc(val)}</div>'
                                         f'<div class="ae-tile-c"><span>{ae_esc(_cmp_label)} {ae_esc(b_txt)}</span>{pill}</div>'
                                         f'<div class="ae-tile-r">{bottom(key, fb)}</div></div>')
                        return '<div class="ae-tiles">' + "".join(tiles) + "</div>"


                    _rc_wait = ("Reality Check running; its range appears here when it finishes." if rc_mode != "off"
                                else "Reality Check is off for this run (sidebar).")

                    # =====================================================================
                    # OVERVIEW
                    # =====================================================================
                    with tab_ov:
                        _ov_links = [("ov-growth", "Growth of $10,000"), ("ov-summary", "Performance Summary"), ("ov-notes", "Run notes")]
                        if ds_state.get("cmp"):
                            _ov_links.insert(0, ("ov-defaults", "Against the default settings"))
                        ae_md(ae_head(_eyebrow, f"Sortino {s_sort:.2f} over {_span}, with a worst drop of {abs(s_dd):.1%}", _desc,
                                      _ov_links))
                        _ov_tiles = st.empty()
                        _ov_tiles.markdown(_ae_tiles_html(None, _rc_wait), unsafe_allow_html=True)

                        # Against the default settings (November 20, 2026). Drawn again with the two
                        # runs' nearby settings once the Reality Check has finished.
                        _ov_ds = None
                        if ds_state.get("on"):
                            _ov_ds = st.empty()
                            _ov_ds.markdown(ds_card_html(ds_state, None, "wait" if rc_mode != "off" else "off"),
                                            unsafe_allow_html=True)

                        ae_md(ae_h3(f"Growth of ${INITIAL_CAPITAL:,}",
                                    "Log scale, so equal heights are equal percentage gains. Hover for each month's values; "
                                    "the buttons change the period and a legend click hides a line." + _cmp_sub,
                                    anchor="ov-growth"))
                        ae_plot(ae_fig_growth(strat_returns, bench_returns, balanced_data, lev_tag), f"Growth of ${INITIAL_CAPITAL:,}",
                                "ae_growth")
                        ae_md('<p style="text-align:center;font-size:12px;color:#8A8680;margin:-0.75rem 0 1rem;">'
                              "Ending value &middot; yearly return, for the period selected</p>")

                        # ---- PERFORMANCE SUMMARY ----
                        def fmt_pct(v):
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                return ae_neg(f"{v:.2%}")
                            elif isinstance(v, float) and np.isnan(v):
                                return "N/A"
                            return v

                        def fmt_ratio(v):
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                return ae_neg(f"{v:.2f}")
                            elif isinstance(v, float) and np.isnan(v):
                                return "N/A"
                            return v

                        def fmt_dollar(v):
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                return f"${v:,.0f}"
                            elif isinstance(v, float) and np.isnan(v):
                                return "N/A"
                            return v

                        # Columns: Strategy, SPY, then one per balanced benchmark (as before)
                        metric_cols = [("Strategy", strat_metrics, BRAND_PURPLE), (BENCHMARK_TICKER, bench_metrics, BRAND_INDIGO)]
                        for entry in balanced_data:
                            metric_cols.append((entry["label"], entry["metrics"], entry["color"]))
                        # The default run's column, beside the strategy's, when this run changed a
                        # setting and the two cover the same months (November 20, 2026).
                        if ds_state.get("cmp") and ds_state["cmp"]["same"]:
                            metric_cols.insert(1, (DS_COLUMN, ds_state["cmp"]["base"], BRAND_PURPLE_FADE))

                        def _fmt_occurrence(ts, kind):
                            # (inline text, hover text) for the date beneath a value,
                            # matching the Chart Dashboard's labels.
                            ts = pd.Timestamp(ts)
                            if kind == "month":
                                return ts.strftime("%b %Y"), ts.strftime("%B %Y")
                            d = f"{ts.strftime('%b')} {ts.day}, {ts.year}"
                            return d, (f"Week ending {d}" if kind == "week" else d)

                        # Plain-language hover explanations for the three ratios (added November 8, 2026).
                        RATIO_TIPS = {
                            "Sortino Ratio": (
                                "Return per unit of bad volatility. It takes the yearly return above "
                                "cash (BIL) and divides it by how large the losing months typically "
                                "are. Gains don't count against it, only losses. Higher is better."
                            ),
                            "Calmar Ratio": (
                                "Return per unit of worst loss. It divides the yearly growth rate by "
                                "the deepest peak-to-trough drop (on month-end values). A Calmar of 1 "
                                "means a year's return about equals the worst drawdown. It rests on "
                                "that one episode, so it can jump when a single bad stretch changes. "
                                "Higher is better."
                            ),
                            "Martin Ratio": (
                                "Return per unit of drawdown pain. It divides the yearly return above "
                                "cash (BIL) by the Ulcer Index, which measures how far below its last "
                                "high the portfolio sat in every month. So it counts how deep and how "
                                "long every drawdown was, not just the worst one, and is steadier "
                                "than Calmar. Higher is better."
                            ),
                        }

                        def metric_row(name, key, fmt, bold, date_key=None, date_kind=None):
                            label = ae_esc(name)
                            if name in RATIO_TIPS:
                                label = (f'<span class="ae-tip">{ae_esc(name)}<span class="ae-tip-i">i</span>'
                                         f'<span class="ae-tip-box">{ae_esc(RATIO_TIPS[name])}</span></span>')
                            vals = []
                            for _, m, _c in metric_cols:
                                v = ae_esc(fmt(m.get(key, 0)))
                                dt = m.get(date_key) if date_key else None
                                if dt is not None and not (isinstance(dt, float) and np.isnan(dt)):
                                    inline, tip = _fmt_occurrence(dt, date_kind)
                                    v = f'<span title="{ae_esc(tip)}">{v}<span class="ae-date">{ae_esc(inline)}</span></span>'
                                vals.append(v)
                            return (label, vals, bold)

                        # Ordered with key metrics first; Sharpe and Total Months removed
                        growth_key = f"Growth of ${INITIAL_CAPITAL:,}"
                        metrics_rows = [
                            metric_row("Sortino Ratio",     "Sortino Ratio",       fmt_ratio,  True),
                            metric_row("CAGR",              "CAGR",                fmt_pct,    True),
                            metric_row("Max Drawdown",      "Max Drawdown",        fmt_pct,    True),
                            metric_row("Max DD (Daily)",    "Max Drawdown (Daily)", fmt_pct,   True),
                            metric_row("Worst Single Day",  "Worst Day",           fmt_pct,    False),
                            metric_row("Worst Week",        "Worst Week",          fmt_pct,    False, "Worst Week Date",  "week"),
                            metric_row("Best Week",         "Best Week",           fmt_pct,    False, "Best Week Date",   "week"),
                            metric_row("Worst Month",       "Worst Month",         fmt_pct,    False, "Worst Month Date", "month"),
                            metric_row("Best Month",        "Best Month",          fmt_pct,    False, "Best Month Date",  "month"),
                            metric_row("Calmar Ratio",      "Calmar Ratio",        fmt_ratio,  False),
                            metric_row("Martin Ratio",      "Martin Ratio",        fmt_ratio,  False),
                            metric_row("Annualized Vol",    "Annualized Vol",      fmt_pct,    False),
                            metric_row("Best Year",         "Best Year",           fmt_pct,    False),
                            metric_row("Worst Year",        "Worst Year",          fmt_pct,    False),
                            metric_row("Win Rate",          "Win Rate (monthly)",  fmt_pct,    False),
                            metric_row("Total Return",      "Total Return",        fmt_pct,    False),
                            metric_row(growth_key,          growth_key,            fmt_dollar, False),
                            ("Period", [ae_esc(m.get("Period", "")) for _, m, _c in metric_cols], False),
                        ]
                        header_cells = "".join(
                            f'<th><span class="ae-sw" style="background:{c}"></span>{ae_esc(label)}{lev_tag if label not in ("Strategy", DS_COLUMN) else ""}</th>'
                            for label, _m, c in metric_cols)
                        html_parts = ['<div class="ae-tbl"><table>', f"<thead><tr><th>Metric</th>{header_cells}</tr></thead><tbody>"]
                        for metric, vals, is_bold in metrics_rows:
                            cells = "".join(('<td class="ae-strat">' if i == 0 else "<td>") + v + "</td>" for i, v in enumerate(vals))
                            html_parts.append(('<tr class="ae-bold">' if is_bold else "<tr>") + f'<td class="ae-k">{metric}</td>{cells}</tr>')
                        html_parts.append("</tbody></table></div>")
                        ae_md(ae_h3("Performance Summary", "Key figures first, in bold. Hover the i beside a ratio for what it measures."
                                    + (" The Default settings column is the same period on the default settings."
                                       if any(_lbl == DS_COLUMN for _lbl, _m, _c in metric_cols) else "") + _cmp_sub,
                                    anchor="ov-summary"))
                        ae_md("".join(html_parts))

                        ae_md(ae_h3("Run notes", anchor="ov-notes"))
                        _notes = [("Trading costs", _cost_note), ("Position sizing", sizing_info["line"])]
                        if _dd_note:
                            _notes.append(("Worst drawdown", _dd_note))
                        _notes.append(("How it is calculated", _method_note))
                        ae_md('<div class="ae-notes">' + "".join(
                            f'<div class="ae-note"><div class="ae-note-k">{ae_esc(k)}</div><div class="ae-note-v">{ae_esc(v)}</div></div>'
                            for k, v in _notes) + "</div>")

                    # =====================================================================
                    # SIGNALS
                    # =====================================================================
                    with tab_sig:
                        signals = compute_current_signals(daily_prices, monthly_prices, params)
                        if signals is None:
                            ae_md(ae_head("Current Model Signals", "No current signal",
                                          "The model could not compute a signal for the latest close."))
                        else:
                            sig_date = signals["signal_date"]
                            eq_details = signals.get("equity_details", [])
                            _sig_links = [("sg-holdings", "Holdings and changes"), ("sg-rank", "Asset ranking")]
                            if eq_details:
                                _sig_links.append(("sg-equity", "Equity sub-selection"))
                            ae_md(ae_head(f"As of the {sig_date:%A, %B} {sig_date.day}, {sig_date:%Y} close", "Current Model Signals",
                                          "What the model would hold if it rebalanced today.", _sig_links))

                            # ---- holdings card ----
                            _w = signals["weights"]
                            _held = sorted(signals["final_holdings"], key=lambda t: -_w.get(t, 0))
                            _wmax = max([_w.get(t, 0) for t in _held] + [1e-9])
                            _rows = []
                            for t in _held:
                                sub = ae_asset_name(t)
                                if t in signals["selected_equities"]:
                                    sub = (sub + " · " if sub else "") + "equity pick"
                                _rows.append(f'<div class="ae-hold"><div><span class="ae-hold-t">{ae_esc(t)}</span>'
                                             f'<span class="ae-hold-n">{ae_esc(sub)}</span></div>'
                                             f'<div class="ae-hold-b"><span style="width:{_w.get(t, 0) / _wmax * 100:.1f}%"></span></div>'
                                             f'<div class="ae-hold-w">{_w.get(t, 0):.1%}</div></div>')
                            _sz_live = signals.get("sizing_note") or ""
                            _extra = []
                            if _sz_live.startswith("Inverse vol (ERC fallback"):
                                _extra.append(f'<div class="ae-small">This signal uses inverse volatility ({ae_esc(_sz_live[len("Inverse vol ("):])}</div>')
                            # Current leverage signal
                            if leverage_params_ui.get("enabled", False):
                                method = leverage_params_ui.get("method", "holdings_gate")
                                if method == "constant":
                                    _extra.append('<div class="ae-row">' + ae_pill(f"Leverage {leverage_params_ui.get('mult_constant', 1.25):.2f}x", "neutral")
                                                  + '<span class="ae-small">constant</span></div>')
                                else:
                                    lev_m, lev_r, lev_d = compute_leverage_signal(daily_prices, sig_date, leverage_params_ui,
                                                                                  final_holdings=signals["final_holdings"])
                                    _tone = {"leveraged": "good", "risk-on": "good", "defensive": "warn", "neutral": "warn",
                                             "risk-off": "bad"}.get(lev_r, "neutral")
                                    if method == "holdings_gate":
                                        _det = f"holding {', '.join(lev_d['defensives_held'])}" if lev_d.get("defensives_held") else ""
                                    else:
                                        _lt = leverage_params_ui["ticker"]
                                        _det = (f"{_lt} ${lev_d['price']:.2f}, {'above' if lev_d['price_above_long'] else 'below'} its "
                                                f"{leverage_params_ui['long_sma']}-day SMA ${lev_d['long_sma']:.2f}; "
                                                f"{leverage_params_ui['short_sma']}-day SMA {'rising' if lev_d['short_sma_rising'] else 'falling'}"
                                                if lev_d else "")
                                    _extra.append('<div class="ae-row">' + ae_pill(f"Leverage {lev_r} ({lev_m:.2f}x)", _tone)
                                                  + (f'<span class="ae-small">{ae_esc(_det)}</span>' if _det else "") + "</div>")
                            # Current G1 cash rule signal
                            if g1_config.get("enabled"):
                                _g1i = signals.get("g1_info") or {}
                                _g1s = _g1i.get("status", "no data")
                                _g1f = g1_config["fund"]
                                _g1_words = {
                                    "in": f"in: {_g1i.get('moved', 0.0):.2%} of the portfolio moves from BIL to {_g1f}",
                                    "no cash": "not a cash month (BIL not held)",
                                    "yield not down": "out: the yield is not falling",
                                    "fund below BIL": f"out: {_g1f}'s score is not above BIL's",
                                    "no data": "no data at this date",
                                }
                                _g1_tone = {"in": "good", "yield not down": "bad", "fund below BIL": "bad"}.get(_g1s, "neutral")
                                _g1_bits = []
                                if "yield_now" in _g1i:
                                    _g1_bits.append(f"{g1_config['yield_ticker']} {_g1i['yield_now']:.2f} vs "
                                                    f"{_g1i['yield_then']:.2f} {g1_config['yield_lookback']} trading days earlier")
                                if "fund_score" in _g1i:
                                    _g1_bits.append(f"{_g1f} score {_g1i['fund_score']:.2%} vs BIL {_g1i['bil_score']:.2%}")
                                _extra.append('<div class="ae-row">' + ae_pill("G1 cash rule " + _g1_words.get(_g1s, _g1s), _g1_tone)
                                              + (f'<span class="ae-small">{ae_esc(" · ".join(_g1_bits))}</span>' if _g1_bits else "")
                                              + "</div>")
                            _hold_card = ('<div class="ae-card"><div class="ae-row" style="justify-content:space-between">'
                                          '<span class="ae-card-l">Recommended holdings</span>'
                                          f'<span class="ae-row">{ae_tag(sizing_info["label"])}</span></div>'
                                          + "".join(_rows) + "".join(_extra) + "</div>")

                            # ---- changes against the previous month-end signal ----
                            prev_date = signals.get("prev_month_end_date")
                            prev_holdings = signals.get("prev_holdings", [])
                            prev_weights = signals.get("prev_weights", {})
                            if prev_date is not None and prev_holdings:
                                _p_data = signals.get("prev_data_date")
                                _p_exec = signals.get("prev_exec_date")
                                if _p_data is not None and _p_exec is not None:
                                    _p_title = f"Against the {_p_exec:%A, %B} {_p_exec.day} rebalance"
                                    _p_label = (f"That rebalance used the {_p_data:%A, %B} {_p_data.day} close, the trading day before "
                                                f"the trade.")
                                else:
                                    _p_title = f"Against the {prev_date:%B} {prev_date.day} month-end signal"
                                    _p_label = ""
                                _p_sz = signals.get("prev_sizing_note") or ""
                                if _p_sz.startswith("Inverse vol (ERC fallback"):
                                    _p_label += " It used inverse volatility (" + _p_sz[len("Inverse vol ("):]
                                _chg = []
                                _n_in = _n_out = 0
                                for t in sorted(set(prev_weights) | set(_w), key=lambda t: -(_w.get(t, 0) + prev_weights.get(t, 0))):
                                    a, b = prev_weights.get(t, 0.0), _w.get(t, 0.0)
                                    if a <= 0 and b <= 0:
                                        continue
                                    if a <= 0:
                                        _n_in += 1
                                        p, txt = ae_pill("New", "good", "+"), f"{b:.1%}"
                                    elif b <= 0:
                                        _n_out += 1
                                        p, txt = ae_pill("Out", "bad", "−"), f"was {a:.1%}"
                                    else:
                                        d = (b - a) * 100
                                        p, txt = ae_pill(ae_neg(f"{d:+.1f} pts"), "neutral"), f"{a:.1%} to {b:.1%}"
                                    nm = ae_asset_name(t)
                                    _chg.append(f'<div class="ae-chg"><span class="ae-hold-t">{ae_esc(t)}</span>'
                                                f'<span class="ae-small" style="font-size:13.5px">{ae_esc((nm + " · ") if nm else "")}{ae_esc(txt)}</span>{p}</div>')
                                if _n_in or _n_out:
                                    _sum = (f"{_n_in} fund{'s' if _n_in != 1 else ''} would rotate in and {_n_out} out "
                                            "if the month ended today.")
                                else:
                                    _sum = "The same funds as the last rebalance; only the weights move."
                                _chg_card = (f'<div class="ae-card" style="gap:8px"><span class="ae-card-l">{ae_esc(_p_title)}</span>'
                                             f'<div>{"".join(_chg)}</div><div class="ae-small">{ae_esc(_sum)} {ae_esc(_p_label)}</div></div>')
                            else:
                                _chg_card = ('<div class="ae-card"><span class="ae-card-l">Against the last rebalance</span>'
                                             '<div class="ae-small">No previous month-end signal in this run.</div></div>')
                            ae_md(f'<div id="sg-holdings" class="ae-anchor ae-grid2">{_hold_card}{_chg_card}</div>')

                            # ---- asset ranking ----
                            details = signals["asset_details"]
                            vol_months_label = f"{params['vol_months']}mo Vol (ann)"
                            lb_cols = sorted(set(params["main_lookbacks"]))
                            lb_headers = [f"{lb}mo Return" for lb in lb_cols]
                            _maxabs = max([abs(r.get("Blended Score", 0) or 0) for r in details
                                           if isinstance(r.get("Blended Score"), (int, float)) and np.isfinite(r.get("Blended Score"))] + [1e-9])

                            def _pc(v):
                                return ae_neg(f"{v:.2%}") if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else "-"

                            _t = ['<div class="ae-tbl ae-tbl-fit"><table><thead><tr><th>Asset</th>'
                                  + "".join(f"<th>{lb}-month return</th>" for lb in lb_cols)
                                  + f'<th>Blended score</th><th>{params["vol_months"]}-month volatility</th><th>Weight</th>'
                                  '<th class="l">Status</th></tr></thead><tbody>']
                            for row in details:
                                name = row["Asset"]
                                sub = ae_asset_name(name)
                                tags = ""
                                if row.get("Equity Pick"):
                                    tags = " " + ae_tag("equity pick")
                                elif row.get("G1 Fund"):
                                    tags = " " + ae_tag("G1 fund")
                                cells = [f'<td><span class="ae-asset">{ae_esc(name)}</span>'
                                         f'<span class="ae-asset-n">{ae_esc(sub)}</span>{tags}</td>']
                                cells += [f"<td>{_pc(row.get(h, np.nan))}</td>" for h in lb_headers]
                                cells.append(f"<td>{ae_score_bar(row.get('Blended Score', np.nan), _maxabs)}</td>")
                                cells.append(f'<td style="color:{AE_BODY}">{_pc(row.get(vol_months_label, np.nan))}</td>')
                                w = row.get("Weight", 0)
                                cells.append(f'<td style="font-weight:700">{w:.2%}</td>' if w > 0 else "<td>-</td>")
                                if row.get("Selected"):
                                    st_html = ae_pill("Held", "good", "✓")
                                elif row.get("In Basket"):
                                    st_html = ae_pill("In basket", "neutral")
                                else:
                                    st_html = '<span class="ae-small">not in basket</span>'
                                cells.append(f'<td class="l">{st_html}</td>')
                                _t.append(('<tr class="ae-held">' if row.get("Selected") else "<tr>") + "".join(cells) + "</tr>")
                            _t.append("</tbody></table></div>")
                            ae_md(ae_h3("Asset ranking",
                                        f"Every eligible fund on the {ae_esc(params.get('main_lookback_label', ''))}. The top "
                                        f"{params['final_top_n']} by blended score are held, sized by {ae_esc(sizing_info['label'][:1].lower() + sizing_info['label'][1:])}.",
                                        anchor="sg-rank"))
                            ae_md("".join(_t))

                            # ---- equity sub-selection ----
                            if eq_details:
                                eq_lbs = params["equity_lookbacks"]
                                eq_desc = f"{eq_lbs[0]}-month" if len(eq_lbs) == 1 else "/".join(str(l) for l in eq_lbs) + "-month blend"
                                eq_top_n = params["equity_top_n"]
                                eq_lb_cols = sorted(set(eq_lbs))
                                _e = ['<div class="ae-tbl ae-tbl-fit"><table><thead><tr><th>Asset</th>'
                                      + "".join(f"<th>{lb}-month return</th>" for lb in eq_lb_cols)
                                      + f'<th>Blended score</th><th>{params["vol_months"]}-month volatility</th><th class="l">Status</th>'
                                      "</tr></thead><tbody>"]
                                _emax = max([abs(r.get("Blended Score", 0) or 0) for r in eq_details
                                             if isinstance(r.get("Blended Score"), (int, float)) and np.isfinite(r.get("Blended Score"))] + [1e-9])
                                for row in eq_details:
                                    is_pick = row.get("Equity Pick")
                                    cells = [f'<td><span class="ae-asset">{ae_esc(row["Asset"])}</span>'
                                             f'<span class="ae-asset-n">{ae_esc(ae_asset_name(row["Asset"]))}</span></td>']
                                    cells += [f"<td>{_pc(row.get(f'{lb}mo Return', np.nan))}</td>" for lb in eq_lb_cols]
                                    cells.append(f"<td>{ae_score_bar(row.get('Blended Score', np.nan), _emax)}</td>")
                                    cells.append(f'<td style="color:{AE_BODY}">{_pc(row.get(vol_months_label, np.nan))}</td>')
                                    cells.append('<td class="l">' + (ae_pill("Pick", "good", "✓") if is_pick
                                                                     else ae_pill("Dropped", "bad", "✕")) + "</td>")
                                    _e.append(('<tr class="ae-held">' if is_pick else "<tr>") + "".join(cells) + "</tr>")
                                _e.append("</tbody></table></div>")
                                ae_md(ae_h3("Equity sub-selection",
                                            f"Equity ETFs ranked by the equity momentum lookback ({eq_desc}). The top {eq_top_n} "
                                            "advance into the main basket; the rest are dropped before the main-lookback ranking "
                                            "above. This is why an ETF can rank high above yet still not be an equity pick.",
                                            anchor="sg-equity"))
                                ae_md("".join(_e))

                    # =====================================================================
                    # RETURNS AND DRAWDOWNS
                    # =====================================================================
                    with tab_rd:
                        ae_md(ae_head("Returns and drawdowns", "How deep the losses went, year by year",
                                      "Hover for values. The buttons switch the view, and a legend click hides or shows a line.",
                                      [("rd-dd", "Drawdowns"), ("rd-context", "Drawdowns in context"),
                                       ("rd-annual", "Annual returns")]))
                        ae_md(ae_h3("Drawdowns", "How far each portfolio sat below its last high, from daily closes or month-end values."
                                    + _cmp_sub, anchor="rd-dd"))
                        ae_plot(ae_fig_drawdowns(strat_returns, bench_returns, balanced_data, daily_equity, spy_daily_equity, lev_tag),
                                "Drawdowns", "ae_drawdowns")
                        # Drawdowns in context (moved here from the Chart Dashboard, November 21, 2026): this run's
                        # drawdowns of 10% or more, with the fixed study's shaded range beside them.
                        try:
                            if daily_equity is None or not len(daily_equity):
                                raise ValueError("this run has no daily equity curve")
                            _ddc_b, _ddc_note = ddc_band(params.get("sizing", SIZING_IF_UNSET), leverage_enabled)
                            _ddc_cal = market_trading_dates(daily_prices)
                            _ddc_fig, _ddc_sub = ae_fig_drawdown_context(
                                daily_equity, balanced_data, spy_daily_equity, _ddc_cal,
                                pd.Timestamp(_p0.year, _p0.month, 1), lev_tag, _ddc_b, _ddc_note, ddc_live())
                            ae_md(ae_h3("Drawdowns in context", _ddc_sub, anchor="rd-context"))
                            ae_plot(_ddc_fig, "Drawdowns in Context", "ae_drawdown_context")
                        except Exception as _ddc_exc:  # noqa: BLE001 - the section must never take the tab down
                            ae_md(ae_h3("Drawdowns in context", ae_esc(f"Not drawn for this run ({type(_ddc_exc).__name__}: "
                                                                       f"{_ddc_exc})."), anchor="rd-context"))
                        _ys, _yb = ae_yearly(strat_returns), ae_yearly(bench_returns).reindex(ae_yearly(strat_returns).index)
                        _partial = []
                        if _p0.month != 1:
                            _partial.append(f"{_p0.year} starts in {_p0:%B}")
                        if _p1.month != 12:
                            _partial.append(f"{_p1.year} runs through {_p1:%B}")
                        _spy = "S&P 500" + lev_tag
                        _pills = (ae_pill(f"Down years: {int((_ys < 0).sum())} against {int((_yb < 0).sum())} for the {_spy}",
                                          "good" if (_ys < 0).sum() <= (_yb < 0).sum() else "bad")
                                  + ae_pill(f"Ahead of the {_spy} in {int((_ys > _yb).sum())} of {len(_ys)} years", "neutral")
                                  + ae_pill(ae_neg(f"Worst year {_ys.min():.1%} against {_yb.min():.1%}"),
                                            "good" if _ys.min() >= _yb.min() else "bad"))
                        ae_md(ae_h3("Annual returns",
                                    "Calendar-year returns. The 60/40 is one legend click away."
                                    + (" " + ae_esc("; ".join(_partial)) + "." if _partial else ""), _pills, anchor="rd-annual"))
                        ae_plot(ae_fig_annual(strat_returns, bench_returns, balanced_data, lev_tag), "Annual Returns", "ae_annual")

                    # =====================================================================
                    # STRATEGY HEALTH
                    # =====================================================================
                    with tab_hl:
                        ae_md(ae_head("Strategy Health", "Is the edge holding up?",
                                      "Rolling diagnostics show whether performance is holding up or fading. Windows are 36 months "
                                      "unless a chart says otherwise. Hover any chart for its values.",
                                      [("hl-sortino", "Sortino"), ("hl-calmar", "Calmar"), ("hl-cagr", "Return"),
                                       ("hl-hit", "Hit rate"), ("hl-uw", "Time underwater"), ("hl-excess", "Excess return")]))
                        _spy_n = "S&P 500" + lev_tag

                        def _last(s):
                            s = s.dropna()
                            return float(s.iloc[-1]) if len(s) else np.nan

                        _rs36 = compute_rolling_sortino(strat_returns, 36, rf_monthly)
                        _rs24 = compute_rolling_sortino(strat_returns, 24, rf_monthly)
                        _rb36 = compute_rolling_sortino(bench_returns, 36, rf_monthly)
                        _de_s = daily_equity if len(daily_equity) > 0 else None
                        _de_b = spy_daily_equity if len(spy_daily_equity) > 0 else None
                        _rc60 = compute_rolling_calmar(strat_returns, _de_s, 60)
                        _rc36 = compute_rolling_calmar(strat_returns, _de_s, 36)
                        _rcb60 = compute_rolling_calmar(bench_returns, _de_b, 60)
                        _rg_s = compute_rolling_cagr(strat_returns, 36)
                        _rg_b = compute_rolling_cagr(bench_returns, 36)
                        _hr_s = compute_rolling_hit_rate(strat_returns, 36)
                        _hr_b = compute_rolling_hit_rate(bench_returns, 36)
                        _dd_s, _uw_s = compute_underwater(strat_returns)
                        _dd_b, _uw_b = compute_underwater(bench_returns)
                        _ex = (_rg_s - _rg_b).dropna()

                        def _r2(v):
                            return ae_neg(f"{v:.2f}") if np.isfinite(v) else "N/A"

                        _cards = [
                            ("hl_sortino", "hl-sortino", "Rolling Sortino ratio",
                             "Above the dotted line at 1, the return has paid well for the losing months.",
                             ae_now(f"Now {_r2(_last(_rs36))}") + ae_tag(f"{_spy_n} {_r2(_last(_rb36))}"),
                             ae_fig_lines([("Strategy, 36 mo", _rs36, BRAND_PURPLE, 2.2, None),
                                           ("Strategy, 24 mo", _rs24, BRAND_PURPLE_FADE, 1.1, None),
                                           (f"{_spy_n}, 36 mo", _rb36, BRAND_INDIGO, 1.7, "dash")], yfmt=".2f", ref=1.0, tickfmt="~g")),
                            ("hl_calmar", "hl-calmar", "Rolling Calmar ratio",
                             "Return over the worst daily drawdown in the window; 60 months sees a real drawdown more often. "
                             "The axis stops at 5; hover shows the full value.",
                             ae_now(f"Now {_r2(_last(_rc60))}") + ae_tag(f"{_spy_n} {_r2(_last(_rcb60))}"),
                             ae_fig_lines([("Strategy, 60 mo", _rc60, BRAND_PURPLE, 2.2, None),
                                           ("Strategy, 36 mo", _rc36, BRAND_PURPLE_FADE, 1.1, None),
                                           (f"{_spy_n}, 60 mo", _rcb60, BRAND_INDIGO, 1.7, "dash")], yfmt=".2f", ref=1.0, yrange=[0, 5],
                                          tickfmt="~g")),
                            ("hl_cagr", "hl-cagr", "Rolling 36-month return",
                             "The annual growth rate over each trailing three years.",
                             ae_now(f"Now {_ae_pct(_last(_rg_s))}") + ae_tag(f"{_spy_n} {_ae_pct(_last(_rg_b))}"),
                             ae_fig_lines([("Strategy", _rg_s, BRAND_PURPLE, 2.2, None),
                                           (_spy_n, _rg_b, BRAND_INDIGO, 1.7, "dash")], yfmt=".0%")),
                            ("hl_hit", "hl-hit", "Rolling hit rate",
                             "The share of positive months in each trailing three years. The dotted line is 50%.",
                             ae_now(f"Now {_last(_hr_s):.0%}") + ae_tag(f"{_spy_n} {_last(_hr_b):.0%}"),
                             ae_fig_lines([("Strategy", _hr_s, BRAND_PURPLE, 2.2, None),
                                           (_spy_n, _hr_b, BRAND_INDIGO, 1.7, "dash")], yfmt=".0%", ref=0.5, yrange=[0.3, 1.0])),
                            ("hl_uw", "hl-uw", "Time underwater",
                             "Months since the last month-end high. The depth of each drawdown is on Returns and Drawdowns.",
                             ae_now(f"Longest {int(_uw_s.max())} months") + ae_tag(f"{_spy_n} longest {int(_uw_b.max())} months"),
                             ae_fig_underwater(_uw_s, _uw_b, lev_tag)),
                            ("hl_excess", "hl-excess", "Rolling 36-month excess return",
                             f"The strategy's 36-month return minus the {_spy_n}'s. Green when ahead.",
                             ae_now(ae_neg(f"Now {_last(_ex) * 100:+.1f} pts")) + ae_tag(f"Ahead in {(_ex > 0).mean():.0%} of windows")
                             if len(_ex) else "",
                             ae_fig_lines([("Strategy minus " + _spy_n, _ex, AE_INK, 1.3, None)], yfmt=".0%",
                                          fills=[(_ex.clip(lower=0), "rgba(31,138,76,0.22)"), (_ex.clip(upper=0), "rgba(184,67,47,0.22)")])),
                        ]
                        for _i in range(0, len(_cards), 2):
                            _cols = st.columns(2, gap="medium")
                            for _col, (_key, _anchor, _title, _how, _chips, _fig) in zip(_cols, _cards[_i:_i + 2]):
                                with _col:
                                    with st.container(border=True, key=f"aecard_{_key}"):
                                        ae_md(f'<div class="ae-chart-h" id="{_anchor}"><div class="ae-chart-t">{ae_esc(_title)}</div>'
                                              f'<div class="ae-small">{ae_esc(_how)}</div><div class="ae-row">{_chips}</div></div>')
                                        ae_plot(_fig, _title, f"ae_{_key}")
                        # Leverage timeline (only shown when dynamic leverage is on)
                        _lev_fig, _lev_title = ae_fig_leverage(trade_df, leverage_params_ui)
                        if _lev_fig is not None:
                            ae_md(ae_h3(_lev_title, "The leverage multiplier held each month."))
                            ae_plot(_lev_fig, "Dynamic Leverage", "ae_leverage")

                    # =====================================================================
                    # TRADES AND DATA
                    # =====================================================================
                    with tab_td:
                        ae_md(ae_head("Trades and data", "Every rebalance, and where the data comes from",
                                      "The trade log, downloads, this run's settings, the notes on each fund's history and the last few runs.",
                                      [("td-log", "Trade log"), ("td-dl", "Downloads"), ("td-settings", "Settings"),
                                       ("td-data", "Data notes"), ("td-prev", "Previous results")]))

                        # ---- TRADE LOG ----
                        trade_df_display = trade_df.iloc[::-1].reset_index(drop=True)
                        ae_md(ae_h3("Trade log",
                                    "Most recent first. Each chip is a holding, its weight and its return that month. "
                                    f"The sizing column says how each month was weighted ({ae_esc(sizing_info['label'])} in this run).",
                                    anchor="td-log"))
                        _has_lev = leverage_params_ui.get("enabled", False) and "leverage" in trade_df_display.columns
                        _th = ('<div class="ae-tbl"><table><thead><tr><th>Month held</th><th class="l">Holdings, weight and return</th>'
                               '<th>Return</th><th>Turnover</th><th>Cost</th>' + ("<th>Leverage</th>" if _has_lev else "")
                               + '<th class="l">Sizing</th></tr></thead><tbody>')
                        _tr = []
                        for _, _r in trade_df_display.head(6).iterrows():
                            _ret = str(_r.get("portfolio_return", ""))
                            try:
                                _hm = pd.Timestamp(str(_r.get("hold_month")) + "-01")
                                _hm_txt = f"{_hm:%b %Y}"
                            except Exception:
                                _hm_txt = str(_r.get("hold_month", ""))
                            try:
                                _ent = pd.Timestamp(_r.get("exec_entry"))
                                _ent_txt = f"traded {_ent:%b} {_ent.day}"
                            except Exception:
                                _ent_txt = ""
                            _tr.append(f'<tr><td class="l ae-nw"><div style="font-weight:600">{ae_esc(_hm_txt)}</div>'
                                       f'<div class="ae-small">{ae_esc(_ent_txt)}</div></td>'
                                       f'<td class="l">{ae_holding_chips(_r.get("final_holdings", ""))}</td>'
                                       f'<td>{ae_pill(ae_neg(_ret), "bad" if _ret.startswith("-") else "good")}</td>'
                                       f'<td style="color:{AE_BODY}">{ae_esc(_r.get("turnover", ""))}</td>'
                                       f'<td style="color:{AE_BODY}">{ae_esc(_r.get("trading_cost", ""))}</td>'
                                       + (f'<td>{ae_esc(_r.get("leverage", ""))}</td>' if _has_lev else "")
                                       + f'<td class="l ae-small">{ae_esc(_r.get("sizing", ""))}</td></tr>')
                        ae_md(_th + "".join(_tr) + "</tbody></table></div>")
                        with st.expander(f"Full trade log, all {len(trade_df_display)} months"):
                            st.dataframe(trade_df_display, width="stretch", hide_index=True)

                        # ---- DOWNLOADS ----
                        ae_md(ae_h3("Downloads", anchor="td-dl"))
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "Download Trade Log (CSV)",
                                data=trade_df.to_csv(index=False),
                                file_name=f"Trade Log ({sizing_info['file_tag']}).csv",
                                mime="text/csv",
                                width="stretch",
                                on_click="ignore",
                            )
                        with col_dl2:
                            export_df = pd.DataFrame({
                                "date": strat_returns.index,
                                "strategy_return": strat_returns.values,
                                "benchmark_return": bench_returns.values,
                            })
                            export_df["strategy_cumulative"] = INITIAL_CAPITAL * (1 + export_df["strategy_return"]).cumprod()
                            export_df["benchmark_cumulative"] = INITIAL_CAPITAL * (1 + export_df["benchmark_return"]).cumprod()
                            export_df["position_sizing"] = sizing_info["label"]
                            st.download_button(
                                "Download Monthly Returns (CSV)",
                                data=export_df.to_csv(index=False),
                                file_name=f"Monthly Returns ({sizing_info['file_tag']}).csv",
                                mime="text/csv",
                                width="stretch",
                                on_click="ignore",
                            )

                        # ---- SETTINGS FOR THIS RUN ----
                        _set = [("Fixed basket", ", ".join(selected_fixed) if selected_fixed else "None"),
                                ("Equity selection", f"Top {equity_top_n} from {', '.join(selected_equity)}" if equity_top_n > 0 else "None"),
                                ("Final", f"Top {final_top_n} by {main_lookback_label}, {sizing_info['final']}"),
                                ("Trade day", exec_timing_label),
                                ("Trading cost", f"{cost_per_side:.2%} per side" if cost_per_side > 0 else "None (gross)"),
                                ("Period", f"{_p0:%Y-%m} to {_p1:%Y-%m}")]
                        if _sensitivity_run:
                            _set.append(("Sensitivity run", _sensitivity_note))
                        if leverage_params_ui.get("enabled", False):
                            lp = leverage_params_ui
                            if lp.get("method") == "constant":
                                cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                                _set.append(("Leverage", f"Constant {lp.get('mult_constant', 1.25):.2f}x"
                                             + (f", cost {cost_pct:.1f}%" if cost_pct > 0 else "")))
                            elif lp.get("method") == "holdings_gate":
                                cost_pct = lp.get("leverage_cost_annual", 0.0) * 100
                                _set.append(("Leverage", f"Holdings gate ({', '.join(lp.get('defensive_tickers', []))}), "
                                             f"{lp.get('mult_leveraged', 1.25):.2f}x when clear"
                                             + (f", cost {cost_pct:.1f}%" if cost_pct > 0 else "")))
                            elif lp.get("method") == "vt_trend":
                                _set.append(("Leverage", f"{lp['ticker']} {lp['long_sma']}/{lp['short_sma']}d SMA, "
                                             f"on={lp['mult_risk_on']:.2f}x / off={lp['mult_risk_off']:.2f}x"))
                        if g1_config.get("enabled"):
                            _set.append(("G1 cash rule", g1_label(g1_config)))
                        if backfill_rules:
                            _set.append(("Backfills", "; ".join(f"{k} <- {', '.join(v)}" for k, v in backfill_rules.items())))
                        ae_md(ae_h3("Settings for this run", anchor="td-settings"))
                        ae_md('<div class="ae-set">' + "".join(
                            ('<div class="ae-wide">' if k == "Backfills" else "<div>")
                            + f'<span class="ae-set-k">{ae_esc(k)}</span><span class="ae-set-v">{ae_esc(v)}</span></div>'
                            for k, v in _set) + "</div>")

                        # ---- BACKFILL DETAILS & DATA ADJUSTMENTS ----
                        ae_md(ae_h3("Data notes", "Which source covers which dates, and why some histories are stand-ins or repaired.",
                                    anchor="td-data"))
                        if backfill_info:
                            with st.expander("Backfill details (which ticker covers which dates)"):
                                st.dataframe(pd.DataFrame(backfill_info), width="stretch", hide_index=True)
                        _run_assets = list(dict.fromkeys(
                            list(selected_fixed)
                            + (list(selected_equity) if equity_top_n > 0 else [])
                            + [BENCHMARK_TICKER]
                            + [c["ticker"] for c in BAL_BENCHMARKS]
                            + ([g1_config["fund"]] if g1_config.get("enabled") else [])
                        ))
                        _adj_assets = [t for t in _run_assets if t in DATA_ADJUSTMENT_NOTES]
                        if _adj_assets:
                            _default_chains = {
                                p.strip().upper(): [x.strip().upper() for x in c.split(",") if x.strip()]
                                for p, c in BACKFILL_DEFAULTS if p.strip()
                            }
                            # The G1 fund's default chain lives in G1_DEFAULTS.
                            _default_chains.setdefault(G1_DEFAULTS["fund"], list(G1_DEFAULTS["backfill"]))
                            with st.expander("Data adjustments (why some histories are stand-ins or repaired)"):
                                st.caption(
                                    "The table above shows which source covers which dates; these notes say why. "
                                    "They describe the default rules."
                                )
                                for _t in _adj_assets:
                                    _note = DATA_ADJUSTMENT_NOTES[_t]
                                    if _t in _default_chains and backfill_rules.get(_t) != _default_chains[_t]:
                                        _used = ", ".join(backfill_rules.get(_t, [])) or "none"
                                        _note += (
                                            f" *This run's backfill rule for {_t} is not the default "
                                            f"({_used}), so the table above, not this note, describes it.*"
                                        )
                                    st.markdown(f"**{_t}.** {_note}")
                                st.caption(
                                    "The full record, with how each stand-in was checked against its real fund and "
                                    "when each change was made: Data Adjustments.md in the Backtest Engine project folder."
                                )

                        # ---- PREVIOUS RESULTS (the log this run was just added to) ----
                        ae_md(ae_h3("Previous backtest results", f"The latest runs in {ae_esc(os.path.basename(RESULTS_FILE))}, "
                                    "newest first, this run included.", anchor="td-prev"))
                        try:
                            _log = pd.read_excel(RESULTS_FILE, engine="openpyxl")
                            _lg = _log.iloc[::-1].head(6)
                            _lc = [c for c in ("Timestamp", "Period", "Main Lookback", "Final Top N",
                                               "Sortino", "CAGR", "Max DD (Daily)", "Martin") if c in _lg.columns]

                            def _lv(c, v):
                                if c == "Timestamp":
                                    try:
                                        _ts = pd.Timestamp(v)
                                        return (f"{_ts:%b} {_ts.day}, {_ts:%Y}, {_ts.hour % 12 or 12}:{_ts:%M} "
                                                f"{'a.m.' if _ts.hour < 12 else 'p.m.'}")
                                    except Exception:
                                        return str(v)
                                if c in ("CAGR", "Max DD (Daily)") and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                                    return ae_neg(f"{v:.1%}")
                                if c in ("Sortino", "Martin") and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                                    return f"{v:.2f}"
                                return "" if (isinstance(v, float) and np.isnan(v)) else str(v)

                            _num = {"Sortino", "CAGR", "Max DD (Daily)", "Martin"}
                            _hn = {"Final Top N": "Holdings", "Max DD (Daily)": "Max DD, daily"}
                            _h = "".join(("<th>" if c in _num else '<th class="l">') + ae_esc(_hn.get(c, c)) + "</th>" for c in _lc)
                            _sort_style = f' style="font-weight:700;color:{BRAND_PURPLE}"'

                            def _lcell(c, v):
                                cls = ' class="ae-nw"' if c in _num else ' class="l ae-nw"'
                                sty = _sort_style if c == "Sortino" else ""
                                return f"<td{cls}{sty}>{ae_esc(_lv(c, v))}</td>"

                            _b = "".join("<tr>" + "".join(_lcell(c, r[c]) for c in _lc) + "</tr>" for _, r in _lg.iterrows())
                            ae_md(f'<div class="ae-tbl"><table><thead><tr>{_h}</tr></thead><tbody>{_b}</tbody></table></div>')
                            with st.expander(f"All {len(_log)} logged runs"):
                                st.dataframe(_log, width="stretch", hide_index=True)
                        except Exception as _log_exc:
                            st.caption(f"Could not read the results log: {_log_exc}")

                    # =====================================================================
                    # REALITY CHECK (runs last; then Overview's tiles get its ranges)
                    # =====================================================================
                    with tab_rc:
                        if rc_mode != "off":
                            _rc_A = None
                            try:
                                _rc_A, _rc_stats = rc_show_panel(daily_prices, monthly_prices, params, results_df,
                                                                 daily_equity, strat_metrics, rc_mode, rc_extras, basket_b)
                                _ov_tiles.markdown(_ae_tiles_html(_rc_A.get("tiles"), ""), unsafe_allow_html=True)
                            except Exception as _rc_exc:
                                _rc_A = None
                                st.warning(f"The Reality Check could not run: {_rc_exc}")
                                _ov_tiles.markdown(_ae_tiles_html(None, "The Reality Check could not run this time."),
                                                   unsafe_allow_html=True)
                            # Default settings comparison (November 20, 2026): the same nearby settings
                            # on the default run, then the Overview's card again with both averages.
                            if _ov_ds is not None and ds_state.get("cmp"):
                                _ds_near = None
                                if _rc_A is not None and ds_state["cmp"]["same"]:
                                    _ds_bar = st.progress(0.0, text="Default settings: looking for saved nearby runs")
                                    try:
                                        _ds_near = ds_nearby(
                                            ds_state, _rc_A,
                                            lambda i, n: _ds_bar.progress(
                                                min(1.0, i / max(n, 1)),
                                                text=f"Default settings: {i} of {n} nearby backtests done"))
                                    except Exception:
                                        _ds_near = None
                                    finally:
                                        _ds_bar.empty()
                                _ov_ds.markdown(ds_card_html(ds_state, _ds_near, "done" if _rc_A is not None else "failed"),
                                                unsafe_allow_html=True)
                        else:
                            ae_md(ae_head("Reality Check", "Off for this run",
                                          "Turn it on in the sidebar under <b>Reality Check</b> to repeat the backtest on the settings "
                                          "one step away and see how much of the result survives."))

else:
    st.info("Configure parameters in the sidebar, then click **Run Backtest** to see results.")

    if os.path.exists(RESULTS_FILE):
        st.subheader("Previous Backtest Results")
        st.caption(f"Loaded from {os.path.basename(RESULTS_FILE)}")
        try:
            log_df = pd.read_excel(RESULTS_FILE, engine="openpyxl")
            st.dataframe(log_df, width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"Could not read results log: {e}")

        if st.button("Clear Results Log", type="secondary"):
            try:
                os.remove(RESULTS_FILE)
                st.success("Results log cleared. Refresh the page to confirm.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete file (it may be open in Excel): {e}")