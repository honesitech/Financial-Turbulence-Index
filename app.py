import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
import datetime

# --- App Configuration ---
st.set_page_config(page_title="Dual-Key Market Monitor", layout="wide")
st.title("🛡️ VIX-Filtered Turbulence Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")
window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)
lookback = st.sidebar.selectbox("Lookback Period", ["2y", "5y", "10y"], index=1)

st.sidebar.divider()
st.sidebar.header("VIX Filter Settings")
# Threshold for market 'Fear'. 20 is cautious, 30+ is panic.
vix_threshold = st.sidebar.slider("VIX Fear Threshold", 15, 40, 25)

# 50 ETF Tickers identified for calculation
etf_tickers = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'VWO', 'EEM', 'GLD', 'SLV',
    'USO', 'UNG', 'XLK', 'XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI',
    'XLB', 'XLU', 'SMH', 'SOXX', 'KWEB', 'ARKK', 'VGT', 'VNQ', 'RWR', 'IYR',
    'GDX', 'GDXJ', 'XOP', 'OIH', 'KRE', 'XHB', 'ITB', 'IGV', 'SKYY', 'FDN',
    'VUG', 'VTV', 'BND', 'AGG', 'LQD', 'JNK', 'HYG', 'TLT', 'IEI', 'SHY'
]

# --- Data Engine ---
@st.cache_data(ttl=3600)
def get_data(tickers, period):
    # Download ETFs + SPY + VIX (^VIX) for correlation filtering
    data = yf.download(tickers + ['^VIX'], period=period)
    prices = data.xs('Close', level=0, axis=1).dropna()
    return prices

with st.spinner("Fetching market data..."):
    all_prices = get_data(etf_tickers + ['SPY'], lookback)
    returns = all_prices[etf_tickers].pct_change().dropna()
    vix_series = all_prices['^VIX'].reindex(returns.index)

# --- Turbulence Calculation (Mahalanobis Distance) ---
def run_analysis(ret_df, win):
    mu = ret_df.rolling(window=win).mean()
    cov = ret_df.rolling(window=win).cov()
    results = []
    valid_dates = ret_df.index[win:]
    
    for date in valid_dates:
        x = ret_df.loc[date].values
        m = mu.loc[date].values
        S = cov.loc[date].values
        # Tikhonov regularization for matrix stability
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, window)
spy = all_prices['SPY'].reindex(fti.index)
vix = vix_series.reindex(fti.index)

# --- Strategy Logic ---
threshold_90 = fti.quantile(0.90)

# Base Strategy: Cash if Turbulence > 90th percentile
signal_base = (fti < threshold_90).astype(int).shift(1)

# Filtered Strategy: Move to Cash ONLY IF (FTI > 90th%) AND (VIX > threshold)
signal_filtered = ~((fti >= threshold_90) & (vix >= vix_threshold))
signal_filtered = signal_filtered.astype(int).shift(1)

# --- UI Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(fti.index, fti, color='red', label='Turbulence Index', alpha=0.6)
    ax1.set_ylabel('Turbulence (FTI)', color='red')
    ax2 = ax1.twinx()
    ax2.plot(vix.index, vix, color='orange', label='VIX (Fear Gauge)', alpha=0.4)
    ax2.set_ylabel('VIX Level', color='orange')
    plt.title("Structural Stress (FTI) vs. Market Fear (VIX)")
    st.pyplot(fig)

with col2:
    st.metric("Latest FTI", f"{fti.iloc[-1]:.2f}")
    st.metric("Current VIX", f"{vix.iloc[-1]:.1f}")
    
    # Dual-Key Logic for Alerts
    if (fti.iloc[-1] >= threshold_90) and (vix.iloc[-1] >= vix_threshold):
        st.error("🚨 DUAL-KEY ALARM: High Stress & High Fear")
    elif fti.iloc[-1] >= threshold_90:
        st.warning("⚠️ High Stress (VIX Filter Active)")
    else:
        st.success("✅ Normal Conditions")

# --- SECTION: Performance Comparison ---
st.divider()
st.subheader("📊 Performance: Base vs. VIX-Filtered Strategy")
spy_returns = spy.pct_change()
spy_cum = (1 + spy_returns).cumprod()
strat_base_cum = (1 + (spy_returns * signal_base)).cumprod()
strat_filtered_cum = (1 + (spy_returns * signal_filtered)).cumprod()

fig_f, ax_f = plt.subplots(figsize=(10, 4))
ax_f.plot(spy_cum.index, spy_cum, label="Buy & Hold SPY", alpha=0.4, linestyle='--')
ax_f.plot(strat_base_cum.index, strat_base_cum, label="Original FTI Strategy", alpha=0.6)
ax_f.plot(strat_filtered_cum.index, strat_filtered_cum, label="VIX-Filtered Strategy", linewidth=2, color='green')
ax_f.set_ylabel("Growth of $1")
ax_f.legend()
st.pyplot(fig_f)

st.write(f"**VIX-Filtered Final Wealth:** ${strat_filtered_cum.iloc[-1]:.2f}")

# --- SECTION: Lead Time Analyzer ---
st.divider()
st.subheader("⏱️ Lead Time Analysis")
spike_dates = fti[fti >= threshold_90].index
lead_times = []

for spike_date in spike_dates:
    # Check for a 2% drop in the following 14 trading days
    future_spy = spy_returns.loc[spike_date:].iloc[1:15]
    cum_drop = (1 + future_spy).cumprod() - 1
    drops = cum_drop[cum_drop <= -0.02]
    
    if not drops.empty:
        days_to_drop = (drops.index[0] - spike_date).days
        lead_times.append(days_to_drop)

if lead_times:
    avg_lead = np.mean(lead_times)
    st.write(f"Average lead time before a 2% drop: **{avg_lead:.1f} days**.")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
    ax_hist.hist(lead_times, bins=10, color='skyblue', edgecolor='black')
    ax_hist.set_xlabel("Days until 2% Drop")
    st.pyplot(fig_hist)
else:
    st.info("No 2% market drops followed high-turbulence events in this lookback.")

