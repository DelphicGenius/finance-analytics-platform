import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from fpdf import FPDF

st.set_page_config(
    page_title="Finance Dashboard",
    layout="wide"
)

st.markdown("### 📊 End-to-End Finance Analytics Platform")
st.title("Finance Analytics Platform")

# Stock list
stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS"
]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Individual Stock",
        "💼 Portfolio",
        "📊 DCF Valuation",
        "💰 DDM Valuation","⚖️ Valuation Dashboard"
    ]
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

    # -----------------------
# EXPORT (CSV + PDF)
# -----------------------

st.subheader("Download Report")

# CSV Export
csv = data.to_csv().encode("utf-8")

st.download_button(
    "Download Stock Data (CSV)",
    csv,
    file_name=f"{ticker}_data.csv",
    mime="text/csv"
)

# PDF Export
if st.button("Download Summary (PDF)"):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Stock Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Stock: {ticker}", ln=True)
    pdf.cell(200, 10, txt=f"Latest Price: ₹ {latest_price:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Volatility: {vol*100:.2f}%", ln=True)

    pdf.output("stock_report.pdf")

    with open("stock_report.pdf", "rb") as f:
        st.download_button(
            "Click to Download PDF",
            f,
            file_name="stock_report.pdf",
            mime="application/pdf"
        )



# ==================================================
# TAB 2: PORTFOLIO OPTIMIZER
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

    returns = data_p.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # -----------------------
    # OPTIMIZATION
    # -----------------------

    def portfolio_perf(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(
            np.dot(weights.T,
            np.dot(cov_matrix, weights))
        )
        return ret, vol

    def neg_sharpe(weights):
        ret, vol = portfolio_perf(weights)
        return -(ret - risk_free) / vol

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    })

    bounds = tuple(
        (0, 1) for _ in selected
    )

    init = [1 / len(selected)] * len(selected)

    result = minimize(
        neg_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x

    ret, vol = portfolio_perf(weights)

    sharpe = (
        ret - risk_free
    ) / vol

    # -----------------------
    # METRICS
    # -----------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Return",
        f"{ret*100:.2f}%"
    )

    c2.metric(
        "Risk",
        f"{vol*100:.2f}%"
    )

    c3.metric(
        "Sharpe",
        f"{sharpe:.2f}"
    )

    # -----------------------
    # WEIGHTS
    # -----------------------

    st.subheader(
        "Optimal Weights"
    )

    wdf = pd.DataFrame({
        "Stock": selected,
        "Weight %": weights * 100
    })

    st.table(wdf)

    # -----------------------
    # PORTFOLIO GROWTH
    # -----------------------

    portfolio_ret = (
        returns * weights
    ).sum(axis=1)

    portfolio_cum = (
        1 + portfolio_ret
    ).cumprod()

    st.subheader(
        "Portfolio Growth (₹1)"
    )

    st.line_chart(portfolio_cum)

    # -----------------------
    # EFFICIENT FRONTIER
    # -----------------------

    frontier_returns = []
    frontier_volatility = []
    frontier_sharpe = []

    for _ in range(5000):

        random_weights = np.random.random(
            len(selected)
        )

        random_weights /= np.sum(
            random_weights
        )

        p_return = np.dot(
            random_weights,
            mean_returns
        )

        p_vol = np.sqrt(
            np.dot(
                random_weights.T,
                np.dot(
                    cov_matrix,
                    random_weights
                )
            )
        )

        p_sharpe = (
            p_return -
            risk_free
        ) / p_vol

        frontier_returns.append(
            p_return
        )

        frontier_volatility.append(
            p_vol
        )

        frontier_sharpe.append(
            p_sharpe
        )

    st.subheader(
        "Efficient Frontier"
    )

    frontier_df = pd.DataFrame({
        "Return": frontier_returns,
        "Risk": frontier_volatility,
        "Sharpe": frontier_sharpe
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frontier_df["Risk"],
            y=frontier_df["Return"],
            mode="markers",
            marker=dict(
                color=frontier_df["Sharpe"],
                colorscale="Viridis",
                size=5,
                colorbar=dict(
                    title="Sharpe"
                )
            ),
            name="Portfolios"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[vol],
            y=[ret],
            mode="markers",
            marker=dict(
                color="red",
                size=14
            ),
            name="Max Sharpe"
        )
    )

    fig.update_layout(
        xaxis_title="Risk",
        yaxis_title="Return",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------
    # EXPORT
    # -----------------------

    st.subheader(
        "Download Portfolio Report"
    )

    portfolio_csv = (
        wdf.to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download Weights (CSV)",
        portfolio_csv,
        file_name="portfolio_weights.csv",
        mime="text/csv"
    )

    if st.button(
        "Download Portfolio PDF"
    ):

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font(
            "Arial",
            size=12
        )

        pdf.cell(
            200,
            10,
            txt="Portfolio Report",
            ln=True,
            align="C"
        )

        pdf.ln(10)

        pdf.cell(
            200,
            10,
            txt=f"Return: {ret*100:.2f}%",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Risk: {vol*100:.2f}%",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Sharpe: {sharpe:.2f}",
            ln=True
        )

        pdf.output(
            "portfolio_report.pdf"
        )

        with open(
            "portfolio_report.pdf",
            "rb"
        ) as f:

            st.download_button(
                "Click to Download PDF",
                f,
                file_name="portfolio_report.pdf",
                mime="application/pdf"
            )

# ==================================================
# TAB 3: DCF VALUATION
# ==================================================

with tab3:

    st.header("DCF Valuation Model")

    st.write(
        "Estimate intrinsic value using Discounted Cash Flow"
    )

    st.subheader("Input Assumptions")

    col1, col2 = st.columns(2)

    with col1:

        fcf0 = st.number_input(
            "Latest Free Cash Flow (₹ Cr)",
            value=1000.0,
            step=50.0,
            key="fcf0"
        )

        wacc = st.number_input(
            "WACC (%)",
            value=12.0,
            key="wacc"
        ) / 100

    with col2:

        terminal_growth = st.number_input(
            "Terminal Growth (%)",
            value=4.0,
            key="terminal_growth"
        ) / 100

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

    st.divider()

    st.subheader("Capital Structure")

    cash = st.number_input(
        "Cash & Cash Equivalents (₹ Cr)",
        value=500.0,
        key="cash"
    )

    debt = st.number_input(
        "Total Debt (₹ Cr)",
        value=1000.0,
        key="debt"
    )

    shares = st.number_input(
        "Shares Outstanding (Cr)",
        value=100.0,
        key="shares"
    )

    # Validation

    if terminal_growth >= wacc:

        st.error(
            "Terminal Growth must be lower than WACC"
        )

        st.stop()

    if shares <= 0:

        st.error(
            "Shares Outstanding must be greater than zero"
        )

        st.stop()

    # Run DCF

    if st.button(
        "Run DCF Valuation",
        key="run_dcf"
    ):

        fcfs = []

        f = fcf0

        for g in growth_rates:

            f = f * (1 + g)

            fcfs.append(f)

        pv_fcfs = 0

        for i in range(len(fcfs)):

            pv_fcfs += (
                fcfs[i] /
                ((1 + wacc) ** (i + 1))
            )

        terminal_value = (
            fcfs[-1] *
            (1 + terminal_growth)
        ) / (
            wacc - terminal_growth
        )

        pv_terminal = (
            terminal_value /
            ((1 + wacc) ** 5)
        )

        enterprise_value = (
            pv_fcfs +
            pv_terminal
        )

        equity_value = (
            enterprise_value +
            cash -
            debt
        )

        intrinsic_price = (
            equity_value /
            shares
        )

        st.divider()

        st.subheader("DCF Result")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "PV of FCF",
            f"₹ {pv_fcfs:,.0f} Cr"
        )

        c2.metric(
            "PV Terminal",
            f"₹ {pv_terminal:,.0f} Cr"
        )

        c3.metric(
            "Enterprise Value",
            f"₹ {enterprise_value:,.0f} Cr"
        )

        c4.metric(
            "Equity Value",
            f"₹ {equity_value:,.0f} Cr"
        )

        st.success(
            f"📌 Intrinsic Value Per Share ≈ ₹ {intrinsic_price:,.2f}"
        )

        # -------------------
        # SENSITIVITY TABLE
        # -------------------

        st.subheader(
            "DCF Sensitivity Analysis"
        )

        growth_rates_sens = [
            0.03,
            0.04,
            0.05
        ]

        wacc_rates_sens = [
            0.10,
            0.11,
            0.12,
            0.13,
            0.14
        ]

        sensitivity_data = []

        for w in wacc_rates_sens:

            row = []

            for g in growth_rates_sens:

                tv = (
                    fcfs[-1] *
                    (1 + g)
                ) / (
                    w - g
                )

                pv_tv = (
                    tv /
                    ((1 + w) ** 5)
                )

                ev = pv_fcfs + pv_tv

                eq = ev + cash - debt

                value_per_share = (
                    eq / shares
                )

                row.append(
                    round(
                        value_per_share,
                        2
                    )
                )

            sensitivity_data.append(
                row
            )

        sensitivity_df = pd.DataFrame(
            sensitivity_data,
            index=[
                "10%",
                "11%",
                "12%",
                "13%",
                "14%"
            ],
            columns=[
                "3%",
                "4%",
                "5%"
            ]
        )

        st.dataframe(
            sensitivity_df.style.background_gradient(
                cmap="RdYlGn"
            )
        )

# ==================================================
# TAB 4: DDM VALUATION
# ==================================================

with tab4:

    st.header(
        "Dividend Discount Model (DDM)"
    )

    st.write(
        "Estimate intrinsic value using the Gordon Growth Model"
    )

    st.divider()

    st.subheader("Input Assumptions")

    dividend = st.number_input(
        "Expected Dividend Next Year (₹)",
        value=10.0,
        key="ddm_dividend"
    )

    growth = st.number_input(
        "Dividend Growth Rate (%)",
        value=5.0,
        key="ddm_growth"
    ) / 100

    cost_equity = st.number_input(
        "Cost of Equity (%)",
        value=12.0,
        key="ddm_cost_equity"
    ) / 100

    if growth >= cost_equity:

        st.error(
            "Growth Rate must be lower than Cost of Equity"
        )

        st.stop()

    if st.button(
        "Run DDM Valuation",
        key="run_ddm"
    ):

        intrinsic_value = (
            dividend /
            (
                cost_equity -
                growth
            )
        )

        st.divider()

        st.subheader(
            "DDM Result"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Dividend",
            f"₹ {dividend:.2f}"
        )

        c2.metric(
            "Growth Rate",
            f"{growth*100:.2f}%"
        )

        c3.metric(
            "Cost of Equity",
            f"{cost_equity*100:.2f}%"
        )

        st.success(
            f"📌 Intrinsic Value ≈ ₹ {intrinsic_value:,.2f}"
        )

        # -------------------
        # SENSITIVITY TABLE
        # -------------------

        st.subheader(
            "DDM Sensitivity Analysis"
        )

        growth_list = [
            0.03,
            0.04,
            0.05,
            0.06
        ]

        ke_list = [
            0.10,
            0.11,
            0.12,
            0.13,
            0.14
        ]

        sensitivity_data = []

        for ke in ke_list:

            row = []

            for g in growth_list:

                if g >= ke:

                    row.append(
                        np.nan
                    )

                else:

                    value = (
                        dividend /
                        (ke - g)
                    )

                    row.append(
                        round(
                            value,
                            2
                        )
                    )

            sensitivity_data.append(
                row
            )

        sensitivity_df = pd.DataFrame(
            sensitivity_data,
            index=[
                "10%",
                "11%",
                "12%",
                "13%",
                "14%"
            ],
            columns=[
                "3%",
                "4%",
                "5%",
                "6%"
            ]
        )

        st.dataframe(
            sensitivity_df.style.background_gradient(
                cmap="RdYlGn"
            )
        )

    st.divider()

    with st.expander(
        "📘 What is DDM?"
    ):

        st.markdown("""
### Dividend Discount Model

Formula:

Value = D₁ / (Ke − g)

Where:

- D₁ = Expected Dividend Next Year
- Ke = Cost of Equity
- g = Dividend Growth Rate

### Best Used For

- Banks
- Insurance Companies
- Mature Dividend-Paying Companies

### Limitation

Not suitable for companies that do not pay dividends.
""")
st.divider()

st.markdown("""
## About This Project

Finance Analytics Platform built using:

- Python
- Streamlit
- Yahoo Finance API
- Pandas
- NumPy
- Plotly
- SciPy

### Features

✅ Stock Analysis

✅ Technical Indicators (MA, RSI)

✅ Portfolio Optimization

✅ Efficient Frontier

✅ DCF Valuation

✅ DDM Valuation

Created by Varad and Vedant Jogadia.
""")