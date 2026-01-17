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