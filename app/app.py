import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Live Stock Dashboard", layout="centered")
st.title("Live Stock Dashboard")

# Stock list
stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS"
]

# Dropdowns
ticker = st.selectbox("Choose a Stock", stocks)

chart_type = st.selectbox(
    "Choose Chart Type",
    ["Line Chart", "Candlestick Chart"]
)

# Load data
@st.cache_data(ttl=300)
def load_data(ticker):
    df = yf.download(ticker, period="1y", auto_adjust=False)
    return df

data = load_data(ticker)

# Check data
if data.empty:
    st.error("No data found.")
    st.stop()

# Fix multi-columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# -----------------------
# INDICATORS
# -----------------------

# Moving Averages
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# -----------------------
# RETURNS & RISK (NEW)
# -----------------------

# Daily Return
data["Daily_Return"] = data["Close"].pct_change()

# Cumulative Return
data["Cumulative_Return"] = (1 + data["Daily_Return"]).cumprod() - 1

# Volatility (Annualized Risk)
volatility = data["Daily_Return"].std() * np.sqrt(252)

# Total Return (1 year)
total_return = data["Cumulative_Return"].iloc[-1] * 100

# -----------------------
# LATEST PRICE
# -----------------------

latest_price = float(data["Close"].iloc[-1])

# -----------------------
# METRICS
# -----------------------

st.subheader("Key Performance Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Latest Price", f"₹ {latest_price:.2f}")
c2.metric("1Y Return", f"{total_return:.2f}%")
c3.metric("Volatility (Risk)", f"{volatility*100:.2f}%")

# -----------------------
# PRICE CHART
# -----------------------

st.subheader("Price Chart (1 Year)")

if chart_type == "Line Chart":

    chart_data = data[["Close", "MA20", "MA50"]]
    st.line_chart(chart_data)

else:

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            name="MA20",
            line=dict(color="blue")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="MA50",
            line=dict(color="orange")
        )
    )

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# RETURNS CHART
# -----------------------

st.subheader("Cumulative Return (Growth of ₹1)")

st.line_chart(data[["Cumulative_Return"]])

# -----------------------
# RSI CHART
# -----------------------

st.subheader("RSI Indicator")

st.line_chart(data[["RSI"]])

st.caption("RSI > 70 = Overbought | RSI < 30 = Oversold")

# -----------------------
# RAW DATA
# -----------------------

with st.expander("Show Raw Data"):
    st.dataframe(data.tail(20))
