import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Market Stress Dashboard", layout="wide")

@st.cache_data
def get_market_data():
    # Core list of tickers
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'GLD', 'TLT']
    
    adj_close_dict = {}
    
    # Download SPY first to establish a master index
    spy_data = yf.download('SPY', period="5y", progress=False)
    if spy_data.empty:
        st.error("Could not connect to Yahoo Finance. Please refresh.")
        st.stop()
        
    # Standardize SPY series
    master_index = spy_data.index
    adj_close_dict['SPY'] = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']

    # Download others and align to master index
    for t in tickers:
        if t == 'SPY': continue
        try:
            tmp = yf.download(t, period="5y", progress=False)
            if not tmp.empty:
                val = tmp['Adj Close'] if 'Adj Close' in tmp.columns else tmp['Close']
                adj_close_dict[t] = val
        except:
            continue

    # Build DataFrame safely
    df_final = pd.DataFrame(adj_close_dict).dropna()
    
    # Get VIX
    vix_raw = yf.download('^VIX', period="5y", progress=False)
    vix_series = vix_raw['Adj Close'] if 'Adj Close' in vix_raw.columns else vix_raw['Close']
    
    return df_final, vix_series

# --- Load Data ---
try:
    data, vix = get_market_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Settings")
window = st.sidebar.slider("Rolling Window", 60, 504, 252)
vix_threshold = st.sidebar.slider("VIX Threshold", 15, 40, 20)
fti_quantile = st.sidebar.slider("FTI Sensitivity", 0.80, 0.99, 0.95)

# --- Logic: Rolling Mahalanobis (FTI) ---
def calculate_fti(df, window_size):
    rets = df.pct_change().dropna()
    fti_vals = []
    
    for i in range(len(rets)):
        if i < window_size:
            fti_vals.append(np.nan)
            continue
        
        subset = rets.iloc[i-window_size:i]
        mu = subset.mean()
        try:
            # Use pseudo-inverse for maximum stability
            inv_cov = np.linalg.pinv(subset.cov().values)
            diff = rets.iloc[i] - mu
            dist = diff.dot(inv_cov).dot(diff.T)
            fti_vals.append(dist)
        except:
            fti_vals.append(np.nan)
            
    return pd.Series(fti_vals, index=rets.index)

fti_series = calculate_fti(data, window)

# Merge and clean
df = pd.DataFrame({'FTI': fti_series, 'VIX': vix, 'SPY': data['SPY']}).dropna()
threshold = df['FTI'].quantile(fti_quantile)

# --- Signals & Performance ---
df['Signal'] = (df['FTI'] > threshold) & (df['VIX'] > vix_threshold)
df['Mkt_Ret'] = df['SPY'].pct_change()
df['Strat_Ret'] = np.where(df['Signal'].shift(1), 0, df['Mkt_Ret'])

# Drawdowns
def calc_dd(rets):
    cum = (1 + rets.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()

df['Mkt_DD'] = calc_dd(df['Mkt_Ret'])
df['Strat_DD'] = calc_dd(df['Strat_Ret'])

# --- UI ---
st.title("🛡️ Market Turbulence Dashboard")

c1, c2, c3 = st.columns(3)
m_ret = (1 + df['Mkt_Ret']).prod() - 1
s_ret = (1 + df['Strat_Ret']).prod() - 1
c1.metric("SPY Return", f"{m_ret:.1%}")
c2.metric("Strat Return", f"{s_ret:.1%}", f"{s_ret-m_ret:.1%}")
c3.metric("Strat Max DD", f"{df['Strat_DD'].min():.1%}")

st.subheader("Equity Curve")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=(1+df['Mkt_Ret']).cumprod(), name="SPY", line=dict(color='gray')))
fig.add_trace(go.Scatter(x=df.index, y=(1+df['Strat_Ret']).cumprod(), name="Strategy", line=dict(color='orange')))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Danger Zone Analysis")
df['Regime'] = "Normal"
df.loc[df['Signal'], 'Regime'] = "CRASH WARNING"
fig_scat = px.scatter(df, x='FTI', y='VIX', color='Regime', color_discrete_map={"Normal":"gray", "CRASH WARNING":"red"})
st.plotly_chart(fig_scat, use_container_width=True)
