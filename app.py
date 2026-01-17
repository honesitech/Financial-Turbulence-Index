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
    # Reduced list to ensure higher reliability and faster calculation
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 
        'XLI', 'XLU', 'XLE', 'XLB', 'AGG', 'LQD', 'HYG', 'TLT', 
        'GLD', 'USO', 'UUP', 'VNQ'
    ]
    
    # Updated yfinance logic to avoid the KeyError
    raw_data = yf.download(tickers, period="5y", group_by='column')
    
    # Robustly extract Adj Close
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close'].dropna()
    else:
        # Fallback for different yfinance return structures
        data = raw_data.xs('Adj Close', axis=1, level=0).dropna()
        
    vix = yf.download('^VIX', period="5y")['Adj Close']
    return data, vix

data, vix = get_market_data()

# --- Sidebar Controls ---
st.sidebar.header("Strategy Settings")
window = st.sidebar.slider("Rolling Window (Days)", 60, 504, 252)
vix_threshold = st.sidebar.slider("VIX Fear Threshold", 15, 40, 20)
fti_quantile = st.sidebar.slider("FTI Sensitivity (Percentile)", 0.80, 0.99, 0.95)

# --- Adaptive Math Logic (Rolling Mahalanobis) ---
def calculate_rolling_fti(df, window_size):
    returns = df.pct_change().dropna()
    fti_values = []
    
    for i in range(len(returns)):
        if i < window_size:
            fti_values.append(np.nan)
            continue
        
        window_data = returns.iloc[i-window_size:i]
        mu = window_data.mean()
        try:
            # Use pseudo-inverse for better stability with rolling windows
            sigma_inv = np.linalg.pinv(window_data.cov().values)
            diff = returns.iloc[i] - mu
            dist = diff.dot(sigma_inv).dot(diff.T)
            fti_values.append(dist)
        except:
            fti_values.append(np.nan)
            
    return pd.Series(fti_values, index=returns.index)

fti_series = calculate_rolling_fti(data, window)
df = pd.DataFrame({'FTI': fti_series, 'VIX': vix, 'SPY': data['SPY']}).dropna()

# --- Signal & Performance Logic ---
fti_threshold = df['FTI'].quantile(fti_quantile)
df['Signal'] = (df['FTI'] > fti_threshold) & (df['VIX'] > vix_threshold)
df['Market_Ret'] = df['SPY'].pct_change()
df['Strat_Ret'] = np.where(df['Signal'].shift(1), 0, df['Market_Ret'])

# --- Drawdown Calculation ---
def get_max_drawdown(returns):
    cum_rets = (1 + returns.fillna(0)).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    return drawdown

df['Market_DD'] = get_max_drawdown(df['Market_Ret'])
df['Strat_DD'] = get_max_drawdown(df['Strat_Ret'])

# --- UI Layout ---
st.title("🛡️ Adaptive Dual-Key Market Stress")

# Performance Metrics Row
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    total_mkt = (1 + df['Market_Ret']).prod() - 1
    st.metric("SPY Total Return", f"{total_mkt:.2%}")
with m_col2:
    total_strat = (1 + df['Strat_Ret']).prod() - 1
    st.metric("Strategy Total Return", f"{total_strat:.2%}", delta=f"{(total_strat - total_mkt):.2%}")
with m_col3:
    st.metric("Strategy Max Drawdown", f"{df['Strat_DD'].min():.2%}", delta=f"{(df['Strat_DD'].min() - df['Market_DD'].min()):.2%}", delta_color="inverse")

# Charts
st.subheader("1. Equity Curve & Drawdown")
fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=df.index, y=(1+df['Market_Ret']).cumprod(), name="Buy & Hold SPY", line=dict(color='gray', dash='dot')))
fig_perf.add_trace(go.Scatter(x=df.index, y=(1+df['Strat_Ret']).cumprod(), name="Filtered Strategy", line=dict(color='orange', width=3)))
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("2. Regime Analysis: The Danger Zone")
df['Regime'] = "Normal"
df.loc[df['FTI'] > fti_threshold, 'Regime'] = "Turbulent (High FTI)"
df.loc[df['VIX'] > vix_threshold, 'Regime'] = "Fearful (High VIX)"
df.loc[df['Signal'], 'Regime'] = "CRASH WARNING"

fig_scatter = px.scatter(df, x='FTI', y='VIX', color='Regime', 
                 color_discrete_map={"Normal": "gray", "Turbulent (High FTI)": "blue", 
                                     "Fearful (High VIX)": "orange", "CRASH WARNING": "red"})
st.plotly_chart(fig_scatter, use_container_width=True)
