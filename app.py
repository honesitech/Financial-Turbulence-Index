import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration & Data ---
st.set_page_config(page_title="Adaptive Market Stress Dashboard", layout="wide")

@st.cache_data
def get_market_data():
    # 30 Liquid ETFs across all asset classes
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 
        'XLI', 'XLU', 'XLE', 'XLB', 'XLRE', 'AGG', 'BND', 'LQD', 'HYG', 'TLT', 
        'GLD', 'SLV', 'USO', 'DBA', 'UUP', 'VNQ', 'VGT', 'VUG', 'VTV', 'VXUS'
    ]
    data = yf.download(tickers, period="5y")['Adj Close'].dropna()
    vix = yf.download('^VIX', period="5y")['Adj Close']
    return data, vix

data, vix = get_market_data()

# --- Sidebar Controls ---
st.sidebar.header("Strategy Settings")
window = st.sidebar.slider("Rolling Window (Days)", 60, 504, 252, help="252 days = 1 trading year")
vix_threshold = st.sidebar.slider("VIX Fear Threshold", 15, 40, 20)
fti_quantile = st.sidebar.slider("FTI Sensitivity (Percentile)", 0.80, 0.99, 0.95)

# --- Adaptive Math Logic (Rolling Mahalanobis) ---
def calculate_rolling_fti(df, window_size):
    returns = df.pct_change().dropna()
    fti_values = []
    
    # We iterate through the data to compute the rolling Mahalanobis distance
    for i in range(len(returns)):
        if i < window_size:
            fti_values.append(np.nan)
            continue
        
        # Lookback window for mean and covariance
        window_data = returns.iloc[i-window_size:i]
        mu = window_data.mean()
        try:
            sigma_inv = np.linalg.inv(window_data.cov().values)
            # Current day's deviation from the rolling average
            diff = returns.iloc[i] - mu
            dist = diff.dot(sigma_inv).dot(diff.T)
            fti_values.append(dist)
        except: # Handle singular matrices if data is flat
            fti_values.append(np.nan)
            
    return pd.Series(fti_values, index=returns.index)

fti_series = calculate_rolling_fti(data, window)
df = pd.DataFrame({'FTI': fti_series, 'VIX': vix, 'SPY': data['SPY']}).dropna()

# --- Signal & Performance Logic ---
fti_threshold = df['FTI'].quantile(fti_quantile)
df['Signal'] = (df['FTI'] > fti_threshold) & (df['VIX'] > vix_threshold)
df['Market_Ret'] = df['SPY'].pct_change()
# Shift signal by 1 day to prevent 'look-ahead bias'
df['Strat_Ret'] = np.where(df['Signal'].shift(1), 0, df['Market_Ret'])

# --- UI Layout ---
st.title("🛡️ Adaptive Dual-Key Market Stress")
st.info(f"Currently analyzing structural stress relative to a {window}-day rolling window.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Adaptive Structural Stress (FTI)")
    st.plotly_chart(px.line(df, y='FTI', title=f"Rolling {window}-Day Mahalanobis Distance"), use_container_width=True)

with col2:
    st.subheader("2. Market Fear (VIX)")
    fig_vix = px.line(df, y='VIX', title="VIX Index")
    fig_vix.add_hline(y=vix_threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig_vix, use_container_width=True)

# --- Scatter Analysis ---
st.divider()
st.subheader("3. Regime Analysis: The Danger Zone")
df['Regime'] = "Normal"
df.loc[df['FTI'] > fti_threshold, 'Regime'] = "Turbulent (High FTI)"
df.loc[df['VIX'] > vix_threshold, 'Regime'] = "Fearful (High VIX)"
df.loc[df['Signal'], 'Regime'] = "CRASH WARNING"

fig_scatter = px.scatter(df, x='FTI', y='VIX', color='Regime', 
                 color_discrete_map={"Normal": "gray", "Turbulent (High FTI)": "blue", 
                                     "Fearful (High VIX)": "orange", "CRASH WARNING": "red"})
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Performance Chart ---
st.subheader("4. Strategy: Adaptive Filter vs. Buy & Hold")
cum_market = (1 + df['Market_Ret'].fillna(0)).cumprod()
cum_strat = (1 + df['Strat_Ret'].fillna(0)).cumprod()

fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=df.index, y=cum_market, name="Buy & Hold SPY", line=dict(color='gray')))
fig_perf.add_trace(go.Scatter(x=df.index, y=cum_strat, name="Filtered Strategy", line=dict(color='orange', width=2)))
st.plotly_chart(fig_perf, use_container_width=True)
