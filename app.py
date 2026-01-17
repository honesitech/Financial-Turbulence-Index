import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# --- App Configuration ---
st.set_page_config(page_title="Dual-Key Market Monitor", layout="wide")
st.title("🛡️ VIX-Filtered Turbulence Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")

# New Lookback Logic: 0.5 to 20 years in 0.5 intervals
years_numeric = np.arange(0.5, 20.5, 0.5)
lookback_options = [f"{y} years" for y in years_numeric]
selected_label = st.sidebar.selectbox("Lookback Period", lookback_options, index=9) # Default 5.0y

# Convert selection to yfinance format
years_val = float(selected_label.split()[0])
if years_val < 1.0:
    lookback_period = f"{int(years_val * 365)}d"
else:
    lookback_period = f"{int(years_val)}y"

window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)

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
    # Download ETFs + SPY + VIX (^VIX)
    data = yf.download(tickers + ['^VIX'], period=period)
    prices = data.xs('Close', level=0, axis=1).dropna()
    return prices

with st.spinner("Fetching market data..."):
    all_prices = get_data(etf_tickers + ['SPY'], lookback_period)
    returns = all_prices[etf_tickers].pct_change().dropna()
    vix_series = all_prices['^VIX'].reindex(returns.index)

# --- Turbulence Calculation (Mahalanobis Distance) ---
def run_analysis(ret_df, win):
    mu = ret_df.rolling(window=win).mean()
    cov = ret_df.rolling(window=win).cov()
    results = []
    valid_dates = ret_df.index[win:]
    
    # Progress tracking for user feedback
    prog = st.progress(0)
    for i, date in enumerate(valid_dates):
        x = ret_df.loc[date].values
        m = mu.loc[date].values
        S = cov.loc[date].values
        
        # Tikhonov regularization for matrix stability
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        
        if i % 100 == 0:
            prog.progress(i / len(valid_dates))
    
    prog.empty()
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, window)
spy = all_prices['SPY'].reindex(fti.index)
vix = vix_series.reindex(fti.index)
threshold_90 = fti.quantile(0.90)

# --- UI Layout: Status Metrics ---
col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Latest FTI", f"{fti.iloc[-1]:.2f}")
col2.metric("Current VIX", f"{vix.iloc[-1]:.1f}")

with col3:
    if (fti.iloc[-1] >= threshold_90) and (vix.iloc[-1] >= vix_threshold):
        st.error("🚨 DUAL-KEY ALARM: Structural Stress & Market Panic")
    elif fti.iloc[-1] >= threshold_90:
        st.warning("⚠️ High Structural Stress (VIX below threshold)")
    else:
        st.success("✅ Normal Conditions: Stable Regime")

# --- SECTION: Stacked Integrated Visualizer ---
st.subheader("📊 Stress and Sentiment Analysis")

fig, (ax_vix, ax_main) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, 
                                      gridspec_kw={'height_ratios': [1, 2]})

# 1. Top Subplot: VIX (The Fear Gauge)
ax_vix.plot(vix.index, vix, color='orange', alpha=0.8, label='VIX Level')
ax_vix.axhline(vix_threshold, color='orange', linestyle='--', alpha=0.4)
ax_vix.fill_between(vix.index, vix, vix_threshold, where=(vix >= vix_threshold), 
                    color='orange', alpha=0.1)
ax_vix.set_ylabel('VIX (Sentiment)', color='orange', fontsize=10)
ax_vix.grid(alpha=0.1)
ax_vix.set_title("Market Fear (VIX)")

# 2. Bottom Subplot: SPY Price (Left Axis)
ax_main.plot(spy.index, spy, color='gray', alpha=0.5, label='SPY Price', linewidth=1)
ax_main.set_ylabel('SPY Price ($)', color='gray', fontsize=10)

# 3. Bottom Subplot Overlay: Turbulence (Right Axis)
ax_fti = ax_main.twinx()
ax_fti.plot(fti.index, fti, color='red', alpha=0.8, label='Turbulence (FTI)', linewidth=1.2)
ax_fti.axhline(threshold_90, color='red', linestyle=':', alpha=0.5, label='90th Percentile')
ax_fti.set_ylabel('Turbulence (FTI)', color='red', fontsize=10)

# 4. Highlight Dual-Key "Danger Zones"
danger_mask = (fti >= threshold_90) & (vix >= vix_threshold)
ax_main.fill_between(fti.index, spy.min(), spy.max(), where=danger_mask, 
                     color='red', alpha=0.2, label='Dual-Key Danger Zone')

ax_main.set_title("Structural Stress (FTI) vs. Price (SPY)")
ax_main.legend(loc='upper left')
fig.tight_layout()
st.pyplot(fig)

# --- SECTION: Strategy Performance ---
st.divider()
st.subheader("📈 Performance: VIX-Filtered FTI Strategy")

spy_returns = spy.pct_change()
# Strategy: Cash if Turbulence > 90% AND VIX > threshold
signal_filtered = ~((fti >= threshold_90) & (vix >= vix_threshold))
signal_filtered = signal_filtered.astype(int).shift(1)

spy_cum = (1 + spy_returns).cumprod()
strat_filtered_cum = (1 + (spy_returns * signal_filtered)).cumprod()

fig_perf, ax_p = plt.subplots(figsize=(10, 4))
ax_p.plot(spy_cum.index, spy_cum, label="Buy & Hold SPY", alpha=0.3, linestyle='--')
ax_p.plot(strat_filtered_cum.index, strat_filtered_cum, label="VIX-Filtered Strategy", color='green', linewidth=2)
ax_p.set_ylabel("Growth of $1")
ax_p.legend()
st.pyplot(fig_perf)

st.write(f"**Final Strategy Wealth:** ${strat_filtered_cum.iloc[-1]:.2f}")

# --- Historical Background ---
with st.expander("🔍 Understanding the Data"):
    st.write(f"""
    - **FTI Values (e.g., 4 vs 12)**: A value of 4 is normal. A value of 12 represents extreme statistical unusualness, such as the COVID-19 crash.
    - **Mahalanobis Distance**: This measures how far current market moves are from the historical average, accounting for correlations.
    - **Dual-Key Logic**: We only move to cash when both structural stress is high (FTI) and market sentiment is panicked (VIX).
    """)
