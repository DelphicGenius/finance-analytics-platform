import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page setup
st.set_page_config(page_title="Portfolio Dashboard", layout="centered")
st.title("Portfolio Analysis Dashboard")

# Stock list
stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS"
]

# Multi-select
selected_stocks = st.multiselect(
    "Select Stocks for Portfolio",
    stocks,
    default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
)

# Load data
@st.cache_data(ttl=300)
def load_data(tickers):
    df = yf.download(tickers, period="1y", auto_adjust=True)["Close"]
    return df

if len(selected_stocks) == 0:
    st.warning("Please select at least one stock.")
    st.stop()

data = load_data(selected_stocks)

# Fix if single stock
if isinstance(data, pd.Series):
    data = data.to_frame()

# -----------------------
# RETURNS
# -----------------------

returns = data.pct_change().dropna()

# Equal weights
n = len(selected_stocks)
weights = np.array([1/n] * n)

# Portfolio daily return
portfolio_return = (returns * weights).sum(axis=1)

# Cumulative return
portfolio_cum = (1 + portfolio_return).cumprod()

# Volatility
portfolio_vol = portfolio_return.std() * np.sqrt(252)

# Total return
total_return = (portfolio_cum.iloc[-1] - 1) * 100

# -----------------------
# METRICS
# -----------------------

st.subheader("Portfolio Performance")

c1, c2, c3 = st.columns(3)

c1.metric("No. of Stocks", n)
c2.metric("1Y Return", f"{total_return:.2f}%")
c3.metric("Volatility (Risk)", f"{portfolio_vol*100:.2f}%")

# -----------------------
# CHARTS
# -----------------------

st.subheader("Portfolio Growth (₹1 Invested)")

st.line_chart(portfolio_cum)

st.subheader("Individual Stock Prices")

st.line_chart(data)

# -----------------------
# WEIGHTS
# -----------------------

st.subheader("Portfolio Weights (Equal)")

weights_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Weight (%)": weights * 100
})

st.table(weights_df)

# -----------------------
# RAW DATA
# -----------------------

with st.expander("Show Returns Data"):
    st.dataframe(returns.tail(20))
