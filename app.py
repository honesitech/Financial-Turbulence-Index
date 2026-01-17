import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# --- App Configuration ---
st.set_page_config(page_title="Dual-Key Market Monitor", layout="wide")
st.title("🛡️ VIX-Filtered Turbulence Dashboard (50 Assets)")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")
window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)
lookback = st.sidebar.selectbox("Lookback Period", ["2y", "5y", "10y"], index=1)

st.sidebar.divider()
st.sidebar.header("VIX Filter Settings")
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
    data = yf.download(tickers + ['^VIX'], period=period)
    prices = data.xs('Close', level=0, axis=1).dropna()
    return prices

with st.spinner("Fetching market data..."):
    all_prices = get_data(etf_tickers + ['SPY'], lookback)
    returns = all_prices[etf_tickers].pct_change().dropna()
    vix_series = all_prices['^VIX'].reindex(returns.index)

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
        # Tikhonov regularization for stability
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, window)
spy = all_prices['SPY'].reindex(fti.index)
vix = vix_series.reindex(fti.index)
threshold_90 = fti.quantile(0.90)

# --- UI Layout: Metrics ---
col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Latest FTI", f"{fti.iloc[-1]:.2f}")
col2.metric("Current VIX", f"{vix.iloc[-1]:.1f}")

with col3:
    if (fti.iloc[-1] >= threshold_90) and (vix.iloc[-1] >= vix_threshold):
        st.error("🚨 DUAL-KEY ALARM: High Stress & High Fear")
    elif fti.iloc[-1] >= threshold_90:
        st.warning("⚠️ High Stress (VIX Filter Active)")
    else:
        st.success("✅ Normal Conditions")

# --- IMPROVED STACKED CHART ---
st.subheader("📈 Stress vs. Price Analysis")

# Create two stacked subplots
fig, (ax_vix, ax_main) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                      gridspec_kw={'height_ratios': [1, 2]})

# Top Subplot: VIX (The Fear Gauge)
ax_vix.plot(vix.index, vix, color='orange', alpha=0.8, label='VIX Level')
ax_vix.axhline(vix_threshold, color='orange', linestyle='--', alpha=0.5)
ax_vix.set_ylabel('VIX Level', color='orange', fontsize=10)
ax_vix.grid(alpha=0.1)
ax_vix.legend(loc='upper left')

# Bottom Subplot: SPY Price (Left Y)
ax_main.plot(spy.index, spy, color='gray', alpha=0.6, label='SPY Price', linewidth=1)
ax_main.set_ylabel('SPY Price ($)', color='gray', fontsize=10)

# Bottom Subplot Overlay: FTI (Right Y)
ax_fti = ax_main.twinx()
ax_fti.plot(fti.index, fti, color='red', alpha=0.8, label='FTI (Turbulence)', linewidth=1.2)
ax_fti.axhline(threshold_90, color='red', linestyle=':', alpha=0.5)
ax_fti.set_ylabel('Turbulence (FTI)', color='red', fontsize=10)

# HIGHLIGHT DANGER ZONES (Where both keys trigger)
danger_mask = (fti >= threshold_90) & (vix >= vix_threshold)
ax_main.fill_between(fti.index, spy.min(), spy.max(), where=danger_mask, 
                     color='red', alpha=0.2, label='Dual-Key Alarm')

ax_main.legend(loc='upper left')
fig.tight_layout()
st.pyplot(fig)

# --- SECTION: Performance Comparison ---
st.divider()
st.subheader("📊 Performance: Strategy Results")
spy_returns = spy.pct_change()
signal_base = (fti < threshold_90).astype(int).shift(1)
signal_filtered = ~((fti >= threshold_90) & (vix >= vix_threshold))
signal_filtered = signal_filtered.astype(int).shift(1)

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
