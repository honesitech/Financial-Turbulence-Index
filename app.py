import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

st.set_page_config(page_title="Dual-Key Market Monitor", layout="wide")
st.title("🛡️ VIX-Filtered Turbulence Dashboard")

# --- Sidebar ---
st.sidebar.header("Parameters")
years_numeric = np.arange(0.5, 20.5, 0.5)
lookback_options = [f"{y} years" for y in years_numeric]
selected_label = st.sidebar.selectbox("Lookback Period", lookback_options, index=9)

years_val = float(selected_label.split()[0])
lookback_period = f"{int(years_val * 365)}d" if years_val < 1.0 else f"{int(years_val)}y"

window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)
vix_threshold = st.sidebar.slider("VIX Fear Threshold", 15, 40, 25)

etf_tickers = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'VWO', 'EEM', 'GLD', 'SLV',
    'USO', 'UNG', 'XLK', 'XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI',
    'XLB', 'XLU', 'SMH', 'SOXX', 'KWEB', 'ARKK', 'VGT', 'VNQ', 'RWR', 'IYR',
    'GDX', 'GDXJ', 'XOP', 'OIH', 'KRE', 'XHB', 'ITB', 'IGV', 'SKYY', 'FDN',
    'VUG', 'VTV', 'BND', 'AGG', 'LQD', 'JNK', 'HYG', 'TLT', 'IEI', 'SHY'
]

@st.cache_data(ttl=3600)
def get_data(tickers, period):
    data = yf.download(tickers + ['^VIX'], period=period)
    prices = data.xs('Close', level=0, axis=1).dropna()
    return prices

# --- Execution ---
all_prices = get_data(etf_tickers + ['SPY'], lookback_period)
returns = all_prices[etf_tickers].pct_change().dropna()

# --- CRITICAL: Handle Empty Data ---
if len(returns) < 10:
    st.error("❌ Not enough data for this period. Try a longer Lookback Period.")
    st.stop()

# Auto-adjust window if too large for selected lookback
if window >= len(returns):
    window = int(len(returns) * 0.7)
    st.sidebar.warning(f"Window auto-adjusted to {window} for data length.")

def run_analysis(ret_df, win):
    mu = ret_df.rolling(window=win).mean()
    cov = ret_df.rolling(window=win).cov()
    results = []
    valid_dates = ret_df.index[win:]
    
    if len(valid_dates) == 0: return pd.Series()
    
    for date in valid_dates:
        x = ret_df.loc[date].values
        m = mu.loc[date].values
        S = cov.loc[date].values
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, window)

# --- Final Guard against iloc error ---
if fti.empty:
    st.error("❌ Calculations resulted in an empty dataset. Increase Lookback or decrease Window.")
    st.stop()

vix = all_prices['^VIX'].reindex(fti.index)
spy = all_prices['SPY'].reindex(fti.index)
threshold_90 = fti.quantile(0.90)

# Dashboard display...
st.metric("Latest FTI", f"{fti.iloc[-1]:.2f}")
st.metric("Current VIX", f"{vix.iloc[-1]:.1f}")
# (Remainder of plotting code stays the same)
