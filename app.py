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

# 1. Flexible Lookback: 0.5 to 20 years in 0.5-year intervals
years_numeric = np.arange(2, 20.5, 0.5)
lookback_options = [f"{y} years" for y in years_numeric]
# Defaulting to 5 years (index 9)
selected_label = st.sidebar.selectbox("Lookback Period", lookback_options, index=9) 

years_val = float(selected_label.split()[0])
# Convert to yfinance format (e.g., '182d' or '5y')
lookback_period = f"{int(years_val * 365)}d" if years_val < 1.0 else f"{int(years_val)}y"

# 2. Window with user input
user_window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)

st.sidebar.divider()
st.sidebar.header("VIX Filter Settings")
vix_threshold = st.sidebar.slider("VIX Fear Threshold", 15, 40, 25)

# --- Asset Universe ---
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
    data = yf.download(tickers + ['^VIX', '^GSPC'], period=period)
    prices = data.xs('Close', level=0, axis=1).dropna()
    return prices

with st.spinner("Analyzing Global Markets..."):
    all_prices = get_data(etf_tickers, lookback_period)
    returns = all_prices[etf_tickers].pct_change().dropna()
    
    # SAFETY: Adjust window if lookback is too short to prevent IndexError
    data_length = len(returns)
    if user_window >= data_length:
        active_window = max(10, int(data_length * 0.75))
        st.sidebar.warning(f"⚠️ Window auto-adjusted to {active_window} for short lookback.")
    else:
        active_window = user_window

# --- Turbulence Calculation (Mahalanobis Distance) ---
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
        # Tikhonov regularization for matrix stability
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        if i % 100 == 0:
            prog.progress(i / len(valid_dates))
            
    prog.empty()
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, active_window)

if fti.empty:
    st.error("Insufficient data for calculations. Increase Lookback or decrease Window.")
    st.stop()

vix = all_prices['^VIX'].reindex(fti.index)
sp500_index = all_prices['^GSPC'].reindex(fti.index)

# --- Percentile & Strategy Logic ---
threshold_90 = fti.quantile(0.90)
fti_latest = fti.iloc[-1]
fti_percentile = fti.rank(pct=True).iloc[-1] * 100

# --- UI: Top Metrics ---
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
col1.metric("Latest FTI", f"{fti_latest:.2f}")
col2.metric("FTI Percentile", f"{fti_percentile:.1f}%")
col3.metric("Current VIX", f"{vix.iloc[-1]:.1f}")

with col4:
    if (fti_latest >= threshold_90) and (vix.iloc[-1] >= vix_threshold):
        st.error("🚨 DUAL-KEY ALARM: Systemic Stress & Panic")
    elif fti_latest >= threshold_90:
        st.warning("⚠️ High Turbulence (VIX below threshold)")
    else:
        st.success("✅ Normal Market Regime")

# --- Integrated Dashboard Chart ---
st.subheader("📈 Integrated Market Stress Visualizer")
fig, (ax_vix, ax_main) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                      gridspec_kw={'height_ratios': [1, 2.5]})

ax_vix.plot(vix.index, vix, color='orange', alpha=0.8, label='VIX (Fear)')
ax_vix.axhline(vix_threshold, color='orange', linestyle='--', alpha=0.5)
ax_vix.set_ylabel('VIX', color='orange', fontweight='bold')
ax_vix.set_title("Market Fear Gauge (VIX)")

ax_main.plot(sp500_index.index, sp500_index, color='#2c3e50', alpha=0.6, label='S&P 500 Index')
ax_main.set_ylabel('S&P 500 Level', color='#2c3e50', fontweight='bold')

ax_fti = ax_main.twinx()
ax_fti.plot(fti.index, fti, color='red', alpha=0.8, label='FTI (Stress)', linewidth=1.2)
ax_fti.axhline(threshold_90, color='red', linestyle=':', alpha=0.5, label='90th Percentile')
ax_fti.set_ylabel('Turbulence (FTI)', color='red', fontweight='bold')

danger_mask = (fti >= threshold_90) & (vix >= vix_threshold)
ax_main.fill_between(fti.index, sp500_index.min(), sp500_index.max(), where=danger_mask, 
                     color='red', alpha=0.15, label='Dual-Key Warning')

ax_main.legend(loc='upper left')
fig.tight_layout()
st.pyplot(fig)

# --- Performance Logic ---
st.divider()
st.subheader("📊 Strategy Backtest: Growth of $1")
sp_returns = sp500_index.pct_change()
signal = ~((fti >= threshold_90) & (vix >= vix_threshold))
signal = signal.astype(int).shift(1)

strat_cum = (1 + (sp_returns * signal)).cumprod()
mkt_cum = (1 + sp_returns).cumprod()

fig_perf, ax_p = plt.subplots(figsize=(10, 4))
ax_p.plot(mkt_cum.index, mkt_cum, label="Buy & Hold S&P 500", alpha=0.4, linestyle='--')
ax_p.plot(strat_cum.index, strat_cum, label="Dual-Key Strategy", color='green', linewidth=2)
ax_p.set_ylabel("Portfolio Value")
ax_p.legend()
st.pyplot(fig_perf)

# --- Documentation Section ---
with st.expander("📖 Documentation: Understanding the Strategy"):
    st.write(f"""
    ### 1. The Mahalanobis Distance (FTI)
    The **Financial Turbulence Index (FTI)** is calculated using the **Mahalanobis Distance**. Unlike standard volatility, it measures how 'statistically unusual' today’s returns are compared to historical averages by looking at the interaction (correlations) of 50 different assets simultaneously. 
    - A low FTI (~4) means the market is behaving normally. 
    - A high FTI (~12+) signals a potential systemic breakdown or 'Black Swan' event.

    ### 2. The 90th Percentile Threshold
    The red dotted line on the chart represents the **90th percentile** of the FTI over your selected lookback. This means we only flag the top **10%** most stressful days as turbulent. This threshold is adaptive—it changes based on the period you view to capture only the most extreme relative outliers.

    ### 3. Asset Universe (50 ETFs Used)
    The FTI monitors systemic stress across these key categories:
    * **Broad US Indices**: SPY, IVV, VOO, QQQ, DIA, IWM
    * **Sectors (SPDRs)**: XLK, XLF, XLC, XLY, XLP, XLE, XLV, XLI, XLB, XLU
    * **Global & Emerging**: VWO, EEM, KWEB
    * **Commodities**: GLD, SLV, USO, UNG, GDX, GDXJ
    * **Fixed Income**: BND, AGG, LQD, JNK, HYG, TLT, IEI, SHY
    * **Thematic & Factors**: SMH, SOXX, ARKK, VGT, VNQ, RWR, IYR, XOP, OIH, KRE, XHB, ITB, IGV, SKYY, FDN, VUG, VTV
    """)
