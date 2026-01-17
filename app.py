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



# 1. Flexible Lookback: 2 to 20 years

years_numeric = np.arange(2, 20.5, 0.5)

lookback_options = [f"{y} years" for y in years_numeric]

selected_label = st.sidebar.selectbox("Lookback Period", lookback_options, index=0) 



years_val = float(selected_label.split()[0])

lookback_period = f"{int(years_val * 365)}d" if years_val < 1.0 else f"{int(years_val)}y"



# 2. Window with user input

user_window = st.sidebar.slider("FTI Rolling Window (Days)", 60, 500, 252)



# --- Asset Universe ---

etf_tickers = [

    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
    "EFA", "VWO", "EWJ", "EWZ", "MCHI", "INDA", "FXI", "KRE", "XBI", "SMH", "ITA",
    "AGG", "TLT", "IEF", "SHY", "LQD", "HYG", "BNDX", "EMB", "TIP", "MUB", "JNK",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "CPER", "PALL", "VNQ", "UUP", "FXE", "FXY", 
    "BTC-USD", "ETH-USD", "IAU"

]



# --- Data Engine ---

@st.cache_data(ttl=3600)

def get_data(tickers, period):

    # Fetching ETFs + S&P 500 Index (^GSPC)

    data = yf.download(tickers + ['^GSPC'], period=period)

    prices = data.xs('Close', level=0, axis=1).dropna()

    return prices



with st.spinner("Analyzing Market Structure..."):

    all_prices = get_data(etf_tickers, lookback_period)

    returns = all_prices[etf_tickers].pct_change().dropna()

    

    # SAFETY: Adjust window if lookback is too short

    data_length = len(returns)

    if user_window >= data_length:

        active_window = max(10, int(data_length * 0.75))

        st.sidebar.warning(f"⚠️ Window adjusted to {active_window} for short lookback.")

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

        # Stability regularization

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



sp500_index = all_prices['^GSPC'].reindex(fti.index)



# --- Percentile & Signal Logic ---

threshold_90 = fti.quantile(0.90)

fti_latest = fti.iloc[-1]

fti_percentile = fti.rank(pct=True).iloc[-1] * 100



# --- UI: Top Metrics ---

col1, col2, col3 = st.columns([1, 1, 2])

col1.metric("Latest FTI", f"{fti_latest:.2f}")

col2.metric("FTI Percentile", f"{fti_percentile:.1f}%")



with col3:

    if fti_latest >= threshold_90:

        st.error("🚨 HIGH TURBULENCE REGIME: Structural Outlier Detected")

    else:

        st.success("✅ STABLE REGIME: Asset Correlations Normal")



# --- Integrated Visualizer ---

st.subheader("📈 S&P 500 Index & Structural Stress Overlay")



fig, ax_main = plt.subplots(figsize=(12, 7))



# S&P 500 Index (Left Y)

ax_main.plot(sp500_index.index, sp500_index, color='#2c3e50', alpha=0.6, label='S&P 500 Index')

ax_main.set_ylabel('S&P 500 Level', color='#2c3e50', fontweight='bold')

ax_main.grid(alpha=0.1)



# FTI (Right Y)

ax_fti = ax_main.twinx()

ax_fti.plot(fti.index, fti, color='red', alpha=0.8, label='FTI (Stress)', linewidth=1.2)

ax_fti.axhline(threshold_90, color='red', linestyle=':', alpha=0.5, label='90th Percentile')

ax_fti.set_ylabel('Turbulence (FTI)', color='red', fontweight='bold')



# Danger Zone Highlight

danger_mask = (fti >= threshold_90)

ax_main.fill_between(fti.index, sp500_index.min(), sp500_index.max(), where=danger_mask, 

                     color='red', alpha=0.1, label='Turbulent Period')



plt.title("Price Action vs. Mahalanobis Stress")

ax_main.legend(loc='upper left')

fig.tight_layout()

st.pyplot(fig)



# --- Performance Logic ---

st.divider()

st.subheader("📊 Performance: Turbulence-Aware Strategy")

sp_returns = sp500_index.pct_change()



# Strategy: Exit Market if FTI > 90th percentile

signal = (fti < threshold_90).astype(int).shift(1)



strat_cum = (1 + (sp_returns * signal)).cumprod()

mkt_cum = (1 + sp_returns).cumprod()



fig_perf, ax_p = plt.subplots(figsize=(10, 4))

ax_p.plot(mkt_cum.index, mkt_cum, label="Buy & Hold S&P 500", alpha=0.4, linestyle='--')

ax_p.plot(strat_cum.index, strat_cum, label="FTI Timing Strategy", color='green', linewidth=2)

ax_p.set_ylabel("Portfolio Growth")

ax_p.legend()

st.pyplot(fig_perf)

# --- Asset Health Monitor in Sidebar ---

st.sidebar.divider()

st.sidebar.subheader("📊 Asset Health Monitor")



# Count assets that successfully downloaded

downloaded_count = len(all_prices.columns) - 1  # Subtracting 1 for ^GSPC

expected_count = len(etf_tickers)



if downloaded_count == expected_count:

    st.sidebar.success(f"All {downloaded_count} Assets Active")

else:

    st.sidebar.warning(f"Processing {downloaded_count}/{expected_count} Assets")

    missing = set(etf_tickers) - set(all_prices.columns)

    if missing:

        st.sidebar.write(f"Missing: {', '.join(missing)}")



# Show Data Density

st.sidebar.caption(f"Total Data Points: {len(returns):,}")



# --- Documentation ---

with st.expander("📖 Documentation: Understanding the Turbulence Index"):

    st.write(f"""

   

    ### Mahalanobis Distance (FTI)

    The FTI uses the **Mahalanobis Distance** to measure the relationship between 50 diverse ETFs statistically. 



    ### 90th Percentile Threshold

    The red shading highlights the top **10%** of most unusual days.



### 3. Asset Universe (50 ETFs Used)
                
    The FTI monitors systemic stress across these key categories:

    * **Broad US Indices**: SPY, VUG, QQQ, DIA, IWM, VTV
    * **Sectors (SPDRs)**: XLK, XLF, XLC, XLY, XLP, XLE, XLV, XLI, XLB, XLU, XLRE
    * **Global & Emerging**: EFA, VWO, EWJ, EWZ, MCHI, INDA, FXI, KWEB
    * **Commodities & Industry**: GLD, SLV, USO, UNG, GDX, GDXJ, SMH, KRE, XBI, ITA, XHB, XOP
    * **Fixed Income**: BND, AGG, LQD, JNK, HYG, TLT, IEF, SHY, EMB, TIP
    * **Thematic & Factors**: VNQ, ARKK, VGT, UUP, BTC-USD, ETH-USD
    """)


