import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Page setup
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Finance Analytics Platform")

# Stock list
stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS"
]

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📈 Individual Stock", "💼 Portfolio", "📊 DCF Valuation"]
)


# ===============================
# DATA LOADER
# ===============================

@st.cache_data(ttl=300)
def load_data(ticker):
    df = yf.download(ticker, period="1y", auto_adjust=False)
    return df

@st.cache_data(ttl=300)
def load_multi(tickers):
    df = yf.download(tickers, period="1y", auto_adjust=True)["Close"]
    return df


# ==================================================
# TAB 1: INDIVIDUAL STOCK (MA + RSI + CANDLE)
# ==================================================

with tab1:

    st.header("Individual Stock Analysis")

    ticker = st.selectbox("Choose Stock", stocks)

    chart_type = st.selectbox(
        "Chart Type",
        ["Line Chart", "Candlestick Chart"]
    )

    data = load_data(ticker)

    if data.empty:
        st.error("No data found")
        st.stop()

    # Fix columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # -----------------------
    # INDICATORS
    # -----------------------

    # MA
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

    # Returns
    data["Return"] = data["Close"].pct_change()

    # -----------------------
    # METRICS
    # -----------------------

    latest_price = float(data["Close"].iloc[-1])
    vol = data["Return"].std() * np.sqrt(252)

    c1, c2 = st.columns(2)

    c1.metric("Latest Price", f"₹ {latest_price:.2f}")
    c2.metric("Volatility", f"{vol*100:.2f}%")

    # -----------------------
    # PRICE CHART
    # -----------------------

    st.subheader("Price Chart")

    if chart_type == "Line Chart":

        st.line_chart(data[["Close", "MA20", "MA50"]])

    else:

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA20"],
            name="MA20"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="MA50"
        ))

        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # RSI
    # -----------------------

    st.subheader("RSI")

    st.line_chart(data[["RSI"]])

    st.caption("RSI > 70 = Overbought | RSI < 30 = Oversold")


# ==================================================
# TAB 2: PORTFOLIO (SMART)
# ==================================================

with tab2:

    st.header("Portfolio Optimizer")

    selected = st.multiselect(
        "Select Stocks",
        stocks,
        default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    )

    risk_free = st.number_input(
        "Risk Free Rate (0.06 = 6%)",
        value=0.06
    )

    if len(selected) < 2:
        st.warning("Select at least 2 stocks")
        st.stop()

    data_p = load_multi(selected)

    if isinstance(data_p, pd.Series):
        data_p = data_p.to_frame()

    # Returns
    returns = data_p.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # -----------------------
    # OPTIMIZATION
    # -----------------------

    def portfolio_perf(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol

    def neg_sharpe(weights):
        ret, vol = portfolio_perf(weights)
        return -(ret - risk_free) / vol

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    })

    bounds = tuple((0, 1) for _ in selected)

    init = [1/len(selected)] * len(selected)

    result = minimize(
        neg_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x

    # Performance
    ret, vol = portfolio_perf(weights)
    sharpe = (ret - risk_free) / vol

    # -----------------------
    # METRICS
    # -----------------------

    c1, c2, c3 = st.columns(3)

    c1.metric("Return", f"{ret*100:.2f}%")
    c2.metric("Risk", f"{vol*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}")

    # -----------------------
    # WEIGHTS
    # -----------------------

    st.subheader("Optimal Weights")

    wdf = pd.DataFrame({
        "Stock": selected,
        "Weight %": weights * 100
    })

    st.table(wdf)

    # -----------------------
    # PORTFOLIO GROWTH
    # -----------------------

    portfolio_ret = (returns * weights).sum(axis=1)
    portfolio_cum = (1 + portfolio_ret).cumprod()

    st.subheader("Portfolio Growth (₹1)")

    st.line_chart(portfolio_cum)

    # ==================================================
# TAB 3: DCF VALUATION
# ==================================================

with tab3:

    st.header("DCF Valuation Model")

    st.write("Estimate intrinsic value using Discounted Cash Flow")

    # -----------------------
    # INPUTS
    # -----------------------

    st.subheader("Input Assumptions")

    col1, col2 = st.columns(2)

    with col1:
        fcf0 = st.number_input(
            "Latest Free Cash Flow (₹ Cr)",
            value=1000.0,
            step=50.0
        )

        wacc = st.number_input(
            "WACC (Cost of Capital, %)",
            value=12.0
        ) / 100

    with col2:
        terminal_growth = st.number_input(
            "Terminal Growth (%)",
            value=4.0
        ) / 100

        shares = st.number_input(
            "Shares Outstanding (Cr)",
            value=100.0
        )

    st.divider()

    st.subheader("Growth Assumptions (Next 5 Years)")

    growth_rates = []

    for i in range(5):
        g = st.number_input(
            f"Year {i+1} Growth (%)",
            value=10.0 - i,
            key=f"g{i}"
        ) / 100

        growth_rates.append(g)

    # -----------------------
    # DCF CALCULATION
    # -----------------------

    if st.button("Run DCF Valuation"):

        # Project FCFs
        fcfs = []
        f = fcf0

        for g in growth_rates:
            f = f * (1 + g)
            fcfs.append(f)

        # Discount FCFs
        pv_fcfs = 0

        for i in range(len(fcfs)):
            pv = fcfs[i] / ((1 + wacc) ** (i + 1))
            pv_fcfs += pv

        # Terminal Value (Gordon Model)
        terminal_value = (
            fcfs[-1] * (1 + terminal_growth)
        ) / (wacc - terminal_growth)

        pv_terminal = terminal_value / ((1 + wacc) ** 5)

        # Enterprise Value
        enterprise_value = pv_fcfs + pv_terminal

        # Per Share Value
        intrinsic_price = enterprise_value / shares

        # -----------------------
        # OUTPUT
        # -----------------------

        st.divider()
        st.subheader("DCF Result")

        c1, c2, c3 = st.columns(3)

        c1.metric("PV of FCF (₹ Cr)", f"{pv_fcfs:,.0f}")
        c2.metric("PV of Terminal (₹ Cr)", f"{pv_terminal:,.0f}")
        c3.metric("Enterprise Value (₹ Cr)", f"{enterprise_value:,.0f}")

        st.success(
            f"📌 Intrinsic Value per Share ≈ ₹ {intrinsic_price:,.2f}"
        )

        st.caption(
            "Note: This is an estimate based on assumptions."
        )

