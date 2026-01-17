import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# --- App Configuration ---
st.set_page_config(page_title="Turbulence Regime Monitor", layout="wide")
st.title("🛡️ Mahalanobis Turbulence Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")

# 1. Flexible Lookback: 2.0 to 20 years in 0.5-year intervals
years_numeric = np.arange(2.0, 20.5, 0.5)
lookback_options = [f"{y} years" for y in years_numeric]
selected_label = st.sidebar.selectbox("Lookback Period", lookback_options, index=0) 

years_val = float(selected_label.split()[0])
# Convert to yfinance format safely
lookback_period = f"{int(years_val * 365)}d" if years_val < 1.0 else f"{int(years_val)}y"

# 2. Window with user input
user_window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)

# --- Asset Universe ---
etf_tickers = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'VWO', 'EEM', 'GLD', 'SLV',
    'USO', 'UNG', 'XLK', 'XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI',
    'XLB', 'XLU', 'SMH', 'SOXX', 'KWEB', 'ARKK', 'VGT', 'VNQ', 'RWR', 'IYR',
    'GDX', 'GDXJ', 'XOP', 'OIH', 'KRE', 'XHB', 'ITB', 'IGV', 'SKYY', 'FDN',
    'VUG', 'VTV', 'BND', 'AGG', 'LQD', 'JNK', 'HYG', 'TLT', 'IEI', 'SHY'
]

# --- Data Engine (Optimized with higher TTL to prevent 403 blocks) ---
@st.cache_data(ttl=86400) # Cache for 24 hours to reduce server requests
def get_data(tickers, period):
    data = yf.download(tickers + ['^GSPC'], period=period)
    prices = data.xs('Close', level=0, axis=1).dropna()
    return prices

with st.spinner("Analyzing Market Structure..."):
    all_prices = get_data(etf_tickers, lookback_period)
    returns = all_prices[etf_tickers].pct_change().dropna()
    
    # SAFETY: Adjust window if lookback is too short to prevent IndexError
    data_length = len(returns)
    if user_window >= data_length:
        active_window = max(10, int(data_length * 0.75))
        st.sidebar.warning(f"⚠️ Window adjusted to {active_window} for short lookback.")
    else:
        active_window = user_window

# --- Turbulence Calculation ---
def run_analysis(ret_df, win):
    mu = ret_df.rolling(window=win).mean()
    cov = ret_df.rolling(window=win).cov()
    results = []
    valid_dates = ret_df.index[win:]
    
    prog = st.progress(0)
    for i, date in enumerate(valid_dates):
        x = ret_df.loc[date].values
        m = mu.loc[date].values
        S = cov.loc[date].values
        # Tikhonov regularization for stability
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        if i % 100 == 0:
            prog.progress(i / len(valid_dates))
            
    prog.empty()
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, active_window)

if fti.empty:
    st.error("Insufficient data. Please increase Lookback Period.")
    st.stop()

# Using ^GSPC for actual index levels (1000s)
sp500_index = all_prices['^GSPC'].reindex(fti.index)

# --- Percentile & Signal Logic ---
threshold_90 = fti.quantile(0.90)
fti_latest = fti.iloc[-1]
fti_percentile = fti.rank(pct=True).iloc[-1] * 100

# --- Asset Health Monitor ---
st.sidebar.divider()
st.sidebar.subheader("📊 Asset Health Monitor")
downloaded_count = len(all_prices.columns) - 1
st.sidebar.success(f"All {downloaded_count} Assets Active")
st.sidebar.caption(f"Valid Data Points: {len(fti)}")

# --- UI: Top Metrics ---
col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Latest FTI", f"{fti_latest:.2f}")
col2.metric("FTI Percentile", f"{fti_percentile:.1f}%")

with col3:
    if fti_latest >= threshold_90:
        st.error("🚨 HIGH TURBULENCE: Structural Outlier Detected")
    else:
        st.success("✅ STABLE REGIME: Asset Correlations Normal")

# --- Visualizer ---
st.subheader("📈 S&P 500 Index & Structural Stress Overlay")
fig, ax_main = plt.subplots(figsize=(12, 7))

ax_main.plot(sp500_index.index, sp500_index, color='#2c3e50', alpha=0.6, label='S&P 500 Index')
ax_main.set_ylabel('S&P 500 Level', color='#2c3e50', fontweight='bold')
ax_main.grid(alpha=0.1)

ax_fti = ax_main.twinx()
ax_fti.plot(fti.index, fti, color='red', alpha=0.8, label='FTI', linewidth=1.2)
ax_fti.axhline(threshold_90, color='red', linestyle=':', alpha=0.5, label='90th Pct')
ax_fti.set_ylabel('Turbulence (FTI)', color='red', fontweight='bold')

danger_mask = (fti >= threshold_90)
ax_main.fill_between(fti.index, sp500_index.min(), sp500_index.max(), where=danger_mask, 
                     color='red', alpha=0.1, label='Turbulent Period')

ax_main.legend(loc='upper left')
st.pyplot(fig)

# --- Documentation ---
with st.expander("📖 Documentation: Understanding the Strategy"):
    st.write(f"""
    ### 1. Mahalanobis Distance (FTI)
    The **Financial Turbulence Index (FTI)** measures the relationship between 50 diverse ETFs statistically. 
    It identifies 'structural outliers' where asset correlations deviate from historical norms. 
    

    ### 2. The 90th Percentile Threshold
    We flag the top **10%** of most stressful days relative to your lookback period. 
    This adaptive filter helps isolate regime shifts from normal market noise.

    ### 3. Asset Universe (50 ETFs)
    Includes Broad Indices, Sector SPDRs, Commodities, Bonds, and Factor ETFs to ensure a global view of risk.
    """)
