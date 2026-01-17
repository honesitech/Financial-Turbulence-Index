import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from datetime import datetime, timedelta

# --- App Configuration ---
st.set_page_config(page_title="Professional Turbulence Monitor", layout="wide")

# --- Asset Universe (Optimized Global 50) ---
# Removed redundancy (IVV, VOO) to improve mathematical stability
tickers = [
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
    "EFA", "VWO", "EWJ", "EWZ", "MCHI", "INDA", "FXI", "KRE", "XBI", "SMH", "ITA",
    "AGG", "TLT", "IEF", "SHY", "LQD", "HYG", "BNDX", "EMB", "TIP", "MUB", "JNK",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "CPER", "PALL", "VNQ", "UUP", "FXE", "FXY", 
    "BTC-USD", "ETH-USD", "IAU"
]

# --- Sidebar Inputs ---
st.sidebar.header("🕹️ Controls")
lookback_years = st.sidebar.slider("Lookback Years", 2, 10, 5)
user_window = st.sidebar.slider("Rolling Window (Days)", 60, 500, 252)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, years):
    start_date = datetime.now() - timedelta(days=years*365)
    data = yf.download(tickers, start=start_date)['Close']
    # If any tickers failed, remove them to avoid math errors
    data = data.dropna(axis=1, how='all').ffill().dropna()
    return data

with st.spinner("Synchronizing Global Markets..."):
    prices = get_market_data(tickers, lookback_years)
    returns = prices.pct_change().dropna()

# --- Turbulence Engine (Mahalanobis Distance) ---
def calculate_turbulence(ret_df, window):
    # Using the pseudo-inverse (pinv) prevents the app from crashing 
    # if two assets become perfectly correlated during a crash.
    turb_results = []
    dates = ret_df.index[window:]
    
    for i in range(window, len(ret_df)):
        history = ret_df.iloc[i-window : i]
        mu = history.mean().values
        inv_cov = np.linalg.pinv(history.cov().values)
        y_t = ret_df.iloc[i].values
        
        # Mahalanobis Distance Squared
        d_t = mahalanobis(y_t, mu, inv_cov) ** 2
        turb_results.append(d_t)
        
    return pd.Series(turb_results, index=dates)

fti_raw = calculate_turbulence(returns, user_window)
# Convert to Percentile Rank (0-100) for UX clarity
fti_percentile = fti_raw.rank(pct=True) * 100
current_score = fti_percentile.iloc[-1]

# --- UI: Risk Gauge ---
st.title("🌪️ Financial Turbulence Index")

col1, col2 = st.columns([1, 2])

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_score,
        title = {'text': "Current Risk Percentile"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 70], 'color': "lightgreen"},
                {'range': [70, 90], 'color': "orange"},
                {'range': [90, 100], 'color': "red"}
            ],
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("### Market Regime Summary")
    if current_score > 90:
        st.error(f"**CRITICAL STRESS ({current_score:.1f}%)**: Asset correlations are breaking down. This is typical of systemic shocks.")
    elif current_score > 70:
        st.warning(f"**ELEVATED TURBULENCE ({current_score:.1f}%)**: Unusual market activity. Diversification may be less effective.")
    else:
        st.success(f"**NORMAL REGIME ({current_score:.1f}%)**: Market interactions are within historical norms.")

# --- Interactive Main Chart ---
st.subheader("📈 FTI Percentile vs. S&P 500")
bench_price = prices['SPY'].reindex(fti_percentile.index)

fig_main = go.Figure()
# S&P 500 Price (Primary Axis)
fig_main.add_trace(go.Scatter(x=bench_price.index, y=bench_price, name="S&P 500 (SPY)", line=dict(color='gray', width=1.5), opacity=0.4))
# FTI (Secondary Axis)
fig_main.add_trace(go.Scatter(x=fti_percentile.index, y=fti_percentile, name="Turbulence %", line=dict(color='red', width=2), yaxis="y2"))

fig_main.update_layout(
    template="plotly_dark",
    yaxis=dict(title="S&P 500 Price"),
    yaxis2=dict(title="Turbulence Percentile", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_main, use_container_width=True)


