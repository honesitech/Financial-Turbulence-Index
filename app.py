import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# --- Page Config ---
st.set_page_config(page_title="FTI Dashboard", layout="wide")

# --- CUSTOM CSS: Tightens the UI spacing ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem !important;}
    [data-testid="stMetricLabel"] {font-size: 0.9rem !important;}
    div[data-testid="stVerticalBlock"] {gap: 0.5rem !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Mahalanobis Turbulence Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")
years_numeric = np.arange(2, 20.5, 0.5)
selected_label = st.sidebar.selectbox("Lookback Period", [f"{y} years" for y in years_numeric], index=0) 
years_val = float(selected_label.split()[0])
lookback_period = f"{int(years_val)}y"
user_window = st.sidebar.slider("FTI Window (Days)", 60, 500, 252)

# --- Asset Universe ---
etf_tickers = [
    "SPY", "VUG", "QQQ", "DIA", "IWM", "VTV",
    "XLK", "XLF", "XLC", "XLY", "XLP", "XLE", "XLV", "XLI", "XLB", "XLU", "XLRE",
    "EFA", "VWO", "EWJ", "EWZ", "MCHI", "INDA", "FXI", "KWEB",
    "GLD", "SLV", "USO", "UNG", "GDX", "GDXJ", "SMH", "KRE", "XBI", "ITA", "XHB", "XOP",
    "BND", "AGG", "LQD", "JNK", "HYG", "TLT", "IEF", "SHY", "EMB", "TIP",
    "VNQ", "ARKK", "VGT", "UUP", "BTC-USD", "ETH-USD"
]

@st.cache_data(ttl=3600)
def get_data(tickers, period):
    data = yf.download(tickers + ['^GSPC'], period=period)
    return data.xs('Close', level=0, axis=1).dropna()

with st.spinner("Analyzing Market Structure..."):
    all_prices = get_data(etf_tickers, lookback_period)
    returns = all_prices[etf_tickers].pct_change().dropna()
    active_window = min(user_window, int(len(returns) * 0.75))

def run_analysis(ret_df, win):
    mu = ret_df.rolling(window=win).mean()
    cov = ret_df.rolling(window=win).cov()
    valid_dates = ret_df.index[win:]
    results = []
    
    prog = st.progress(0)
    for i, date in enumerate(valid_dates):
        x, m, S = ret_df.loc[date].values, mu.loc[date].values, cov.loc[date].values
        S_inv = np.linalg.inv(S + np.eye(len(etf_tickers)) * 1e-6)
        results.append(mahalanobis(x, m, S_inv))
        if i % 100 == 0: prog.progress(i / len(valid_dates))
    prog.empty()
    return pd.Series(results, index=valid_dates)

fti = run_analysis(returns, active_window)
sp500_index = all_prices['^GSPC'].reindex(fti.index)

# --- Percentile & Signal ---
threshold_90 = fti.quantile(0.90)
fti_latest = fti.iloc[-1]
fti_percentile = fti.rank(pct=True).iloc[-1] * 100

# --- UI: Compact Metrics ---
# Using gap="small" to keep metrics close together
col1, col2, col3 = st.columns([1, 1, 3], gap="small")
col1.metric("Latest FTI", f"{fti_latest:.2f}")
col2.metric("FTI Rank", f"{fti_percentile:.1f}%")

with col3:
    if fti_latest >= threshold_90:
        st.error("🚨 HIGH TURBULENCE REGIME")
    else:
        st.success("✅ STABLE REGIME")

# --- Main Chart ---
st.subheader("📈 Stress Overlay")
fig, ax_main = plt.subplots(figsize=(12, 4)) # Reduced height for better fit
ax_main.plot(sp500_index.index, sp500_index, color='#2c3e50', alpha=0.6, label='SP500')
ax_fti = ax_main.twinx()
ax_fti.plot(fti.index, fti, color='red', alpha=0.8, label='FTI', linewidth=1)
ax_main.fill_between(fti.index, sp500_index.min(), sp500_index.max(), 
                     where=(fti >= threshold_90), color='red', alpha=0.15)
st.pyplot(fig)

# --- Documentation ---
with st.expander("📖 Documentation: Understanding the Universe"):
    st.markdown("""
    ### 3. Asset Universe (50 ETFs Used)
    * **Broad US Indices**: SPY, VUG, QQQ, DIA, IWM, VTV
    * **Sectors (SPDRs)**: XLK, XLF, XLC, XLY, XLP, XLE, XLV, XLI, XLB, XLU, XLRE
    * **Global & Emerging**: EFA, VWO, EWJ, EWZ, MCHI, INDA, FXI, KWEB
    * **Commodities & Industry**: GLD, SLV, USO, UNG, GDX, GDXJ, SMH, KRE, XBI, ITA, XHB, XOP
    * **Fixed Income**: BND, AGG, LQD, JNK, HYG, TLT, IEF, SHY, EMB, TIP
    * **Thematic & Factors**: VNQ, ARKK, VGT, UUP, BTC-USD, ETH-USD
    """)
