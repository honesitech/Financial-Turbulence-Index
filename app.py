import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="Adaptive Market Stress Dashboard", layout="wide")

@st.cache_data
def get_market_data():
    # A robust list of major asset classes
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 
        'XLI', 'XLU', 'XLE', 'XLB', 'AGG', 'LQD', 'HYG', 'TLT', 
        'GLD', 'USO', 'UUP', 'VNQ'
    ]
    
    # Download data individually to avoid MultiIndex KeyError issues
    adj_close_data = {}
    for ticker in tickers:
        try:
            tmp = yf.download(ticker, period="5y", progress=False)
            if not tmp.empty:
                # Force extraction of 'Adj Close' safely
                if 'Adj Close' in tmp.columns:
                    adj_close_data[ticker] = tmp['Adj Close']
                else:
                    adj_close_data[ticker] = tmp['Close'] # Fallback
        except Exception:
            continue
            
    df_final = pd.DataFrame(adj_close_data).dropna()
    
    # Download VIX separately
    vix = yf.download('^VIX', period="5y", progress=False)
    vix_series = vix['Adj Close'] if 'Adj Close' in vix.columns else vix['Close']
    
    return df_final, vix_series

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
    
    # Pre-calculate values for speed
    for i in range(len(returns)):
        if i < window_size:
            fti_values.append(np.nan)
            continue
        
        window_data = returns.iloc[i-window_size:i]
        mu = window_data.mean()
        try:
            # pinv (Pseudo-inverse) is more stable for rolling matrix math
            sigma_inv = np.linalg.pinv(window_data.cov().values)
            diff = returns.iloc[i] - mu
            dist = diff.dot(sigma_inv).dot(diff.T)
            fti_values.append(dist)
        except:
            fti_values.append(np.nan)
            
    return pd.Series(fti_values, index=returns.index)

fti_series = calculate_rolling_fti(data, window)
# Align all dataframes
common_index = data.index.intersection(fti_series.dropna().index).intersection(vix.index)
df = pd.DataFrame({
    'FTI': fti_series, 
    'VIX': vix, 
    'SPY': data['SPY']
}, index=common_index).dropna()

# --- Signal & Drawdown Logic ---
fti_threshold = df['FTI'].quantile(fti_quantile)
df['Signal'] = (df['FTI'] > fti_threshold) & (df['VIX'] > vix_threshold)
df['Market_Ret'] = df['SPY'].pct_change()
df['Strat_Ret'] = np.where(df['Signal'].shift(1), 0, df['Market_Ret'])

def get_max_drawdown(returns):
    cum_rets = (1 + returns.fillna(0)).cumprod()
    peak = cum_rets.cummax()
    return (cum_rets - peak) / peak

df['Market_DD'] = get_max_drawdown(df['Market_Ret'])
df['Strat_DD'] = get_max_drawdown(df['Strat_Ret'])

# --- UI Layout ---
st.title("🛡️ Adaptive Dual-Key Market Stress")

# Performance Metrics
m1, m2, m3 = st.columns(3)
with m1:
    total_mkt = (1 + df['Market_Ret'].dropna()).prod() - 1
    st.metric("SPY Total Return", f"{total_mkt:.2%}")
with m2:
    total_strat = (1 + df['Strat_Ret'].dropna()).prod() - 1
    st.metric("Strategy Return", f"{total_strat:.2%}", delta=f"{(total_strat - total_mkt):.2%}")
with m3:
    st.metric("Strategy Max DD", f"{df['Strat_DD'].min():.2%}", delta=f"{(df['Strat_DD'].min() - df['Market_DD'].min()):.2%}", delta_color="inverse")

# Charts
st.subheader("1. Equity Curve (Log Scale Available)")
fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=df.index, y=(1+df['Market_Ret'].fillna(0)).cumprod(), name="SPY", line=dict(color='gray')))
fig_perf.add_trace(go.Scatter(x=df.index, y=(1+df['Strat_Ret'].fillna(0)).cumprod(), name="Filtered Strategy", line=dict(color='orange', width=2)))
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("2. Regime Analysis: The Danger Zone")
df['Regime'] = "Normal"
df.loc[df['FTI'] > fti_threshold, 'Regime'] = "Structural Stress"
df.loc[df['VIX'] > vix_threshold, 'Regime'] = "Market Fear"
df.loc[df['Signal'], 'Regime'] = "CRASH WARNING"

fig_scatter = px.scatter(df, x='FTI', y='VIX', color='Regime', 
                         color_discrete_map={"Normal": "gray", "Structural Stress": "blue", 
                                             "Market Fear": "purple", "CRASH WARNING": "red"})
st.plotly_chart(fig_scatter, use_container_width=True)
