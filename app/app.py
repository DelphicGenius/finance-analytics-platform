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

# Load stock + NIFTY data
@st.cache_data(ttl=300)
def load_data(stock):
    stock_df = yf.download(stock, period="1y", auto_adjust=False)
    nifty_df = yf.download("^NSEI", period="1y", auto_adjust=False)
    return stock_df, nifty_df

data, nifty = load_data(ticker)

# Check data
if data.empty or nifty.empty:
    st.error("Data not available.")
    st.stop()

# Fix multi-columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

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
# RETURNS
# -----------------------

# Stock returns
data["Return\db"] = data["Close"].pct_change()
data["Return"] = data["Close"].pct_change()
data["CumReturn"] = (1 + data["Return"]).cumprod()

# NIFTY returns
nifty["Return"] = nifty["Close"].pct_change()
nifty["CumReturn"] = (1 + nifty["Return"]).cumprod()

# -----------------------
# BETA CALCULATION
# -----------------------

merged = pd.concat(
    [data["Return"], nifty["Return"]],
    axis=1
)

merged.columns = ["Stock", "Market"]
merged = merged.dropna()

cov = np.cov(merged["Stock"], merged["Market"])[0][1]
var = np.var(merged["Market"])

beta = cov / var

# -----------------------
# METRICS
# -----------------------

latest_price = float(data["Close"].iloc[-1])
total_return = (data["CumReturn"].iloc[-1] - 1) * 100
volatility = data["Return"].std() * np.sqrt(252)

st.subheader("Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Price", f"₹ {latest_price:.2f}")
c2.metric("1Y Return", f"{total_return:.2f}%")
c3.metric("Volatility", f"{volatility*100:.2f}%")
c4.metric("Beta vs NIFTY", f"{beta:.2f}")

# -----------------------
# PRICE CHART
# -----------------------

st.subheader("Price Chart")

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
            name="MA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="MA50"
        )
    )

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# BENCHMARK COMPARISON
# -----------------------

st.subheader("Stock vs NIFTY (Growth of ₹1)")

compare_df = pd.DataFrame({
    "Stock": data["CumReturn"],
    "NIFTY": nifty["CumReturn"]
})

st.line_chart(compare_df)

# -----------------------
# RSI
# -----------------------

st.subheader("RSI Indicator")

st.line_chart(data[["RSI"]])

st.caption("Beta > 1 = More risky than market | Beta < 1 = Less risky")

# -----------------------
# RAW DATA
# -----------------------

with st.expander("Show Raw Data"):
    st.dataframe(data.tail(20))
