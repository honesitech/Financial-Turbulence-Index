import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
import datetime

# --- App Configuration ---
st.set_page_config(page_title="ETF Turbulence Monitor", layout="wide")
st.title("📈 Financial Turbulence Index Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")
window = st.sidebar.slider("Rolling Window (Days)", 60, 500, 252)
lookback = st.sidebar.selectbox("Lookback Period", ["2y", "5y", "10y"], index=1)

# Your identified 50 ETFs
etf_tickers = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'VWO', 'EEM', 'GLD', 'SLV',
    'USO', 'UNG', 'XLK', 'XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI',
    'XLB', 'XLU', 'SMH', 'SOXX', 'KWEB', 'ARKK', 'VGT', 'VNQ', 'RWR', 'IYR',
    'GDX', 'GDXJ', 'XOP', 'OIH', 'KRE', 'XHB', 'ITB', 'IGV', 'SKYY', 'FDN',
    'VUG', 'VTV', 'BND', 'AGG', 'LQD', 'JNK', 'HYG', 'TLT', 'IEI', 'SHY'
]

# --- Data Engine ---
@st.cache_data(ttl=3600) # Only downloads data once per hour
def get_data(tickers, period):
    data = yf.download(tickers, period=period)
    return data.xs('Close', level=0, axis=1).dropna()

with st.spinner("Fetching market data..."):
    prices = get_data(etf_tickers + ['SPY'], lookback)
    returns = prices[etf_tickers].pct_change().dropna()

# --- Turbulence Calculation ---
def run_analysis(ret_df, win):
    mu = ret_df.rolling(window=win).mean()
    cov = ret_df.rolling(window=win).cov()
    
    results = []
    valid_dates = ret_df.index[win:]
    
    for date in valid_dates:
        x = ret_df.loc[date].values
        m = mu.loc[date].values
        S = cov.loc[date].values
        # Add small value to diagonal for matrix stability
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, window)
spy = prices['SPY'].reindex(fti.index)

# --- UI Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(fti.index, fti, color='red', label='Turbulence Index', alpha=0.7)
    ax1.set_ylabel('Turbulence (Mahalanobis Distance)', color='red')
    
    ax2 = ax1.twinx()
    ax2.plot(spy.index, spy, color='blue', label='S&P 500', alpha=0.5)
    ax2.set_ylabel('SPY Price', color='blue')
    
    plt.title("Turbulence vs. Market Price")
    st.pyplot(fig)

with col2:
    latest = fti.iloc[-1]
    pct = (fti < latest).mean() * 100
    st.metric("Current Turbulence", f"{latest:.2f}")
    st.metric("Percentile Rank", f"{pct:.1f}%")
    
    if pct > 90:
        st.error("⚠️ CRITICAL TURBULENCE")
    elif pct > 75:
        st.warning("⚠️ High Market Stress")
    else:

        st.success("✅ Normal Conditions")
      # --- SECTION 1: Strategy Backtest ---
st.divider()
st.subheader("🔙 Strategy Backtest")
st.markdown("Comparison: Buy & Hold SPY vs. Exiting when FTI > 90th Percentile")

# Define the strategy: 1 if below 90th percentile, 0 if above (move to cash)
threshold_90 = fti.quantile(0.90)
signal = (fti < threshold_90).astype(int).shift(1) 

# Calculate Returns and Cumulative Wealth
spy_returns = spy.pct_change()
strategy_returns = spy_returns * signal
spy_cum = (1 + spy_returns).cumprod()
strategy_cum = (1 + strategy_returns).cumprod()

# Plot Backtest
fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
ax_bt.plot(spy_cum.index, spy_cum, label="Buy & Hold SPY", alpha=0.6)
ax_bt.plot(strategy_cum.index, strategy_cum, label="Turbulence-Adjusted Strategy", linewidth=2, color='green')
ax_bt.set_ylabel("Wealth ($)")
ax_bt.legend()
st.pyplot(fig_bt)

# --- SECTION 2: Lead Time Analyzer ---
st.divider()
st.subheader("⏱️ Lead Time Analysis")

# Find every day where FTI spiked
spike_dates = fti[fti >= threshold_90].index
lead_times = []

for spike_date in spike_dates:
    # Look at the next 14 trading days after the spike
    future_spy = spy_returns.loc[spike_date:].iloc[1:15]
    cum_drop = (1 + future_spy).cumprod() - 1
    drops = cum_drop[cum_drop <= -0.02] # Find a 2% drop
    
    if not drops.empty:
        days_to_drop = (drops.index[0] - spike_date).days
        lead_times.append(days_to_drop)

if lead_times:
    avg_lead = np.mean(lead_times)
    st.write(f"The indicator provided an average lead time of **{avg_lead:.1f} days** before a 2% market drop.")
    
    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
    ax_hist.hist(lead_times, bins=10, color='skyblue', edgecolor='black')
    ax_hist.set_xlabel("Days until 2% Drop")
    st.pyplot(fig_hist)
else:
    st.info("No 2% market drops followed high-turbulence events in this lookback.")
