"""
===========================================================
TradeVision AI
Interactive Dashboard
===========================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
import streamlit as st
import pandas as pd

from src.models.load_model import load_model
from src.models.prepare_data import prepare_dataset


st.set_page_config(
    page_title="TradeVision AI",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------------

model = load_model("tradevision_final.pkl")

X, y = prepare_dataset("GC=F")

latest_features = X.iloc[[-1]]
latest_date = latest_features.index[0]

prediction = model.predict(
    latest_features
)[0]

probability = model.predict_proba(
    latest_features
)[0][1]


# ---------------------------------------------------------
# DASHBOARD TITLE
# ---------------------------------------------------------

st.title("TradeVision AI")

st.subheader(
    "Machine Learning Market Prediction Dashboard"
)

st.caption(
    "Gold futures (GC=F) — Logistic Regression Model"
)


# ---------------------------------------------------------
# KEY METRICS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest Close",
        f"{latest_features['Close'].iloc[0]:.2f}",
    )

with col2:
    st.metric(
        "Upward Probability",
        f"{probability:.2%}",
    )

with col3:
    st.metric(
        "RSI",
        f"{latest_features['RSI'].iloc[0]:.2f}",
    )

with col4:
    st.metric(
        "Volatility",
        f"{latest_features['Volatility'].iloc[0]:.2%}",
    )


# ---------------------------------------------------------
# MODEL SIGNAL
# ---------------------------------------------------------

st.divider()

st.header("Latest Model Signal")

if prediction == 1:

    st.success("BULLISH")

else:

    st.warning("NOT BULLISH")


st.write(
    f"Latest available market date: "
    f"**{latest_date.date()}**"
)


# ---------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------

st.divider()

st.header("Technical Indicators")

indicator_data = pd.DataFrame(
    {
        "Indicator": [
            "RSI",
            "MACD",
            "Volatility",
            "SMA 20",
            "SMA 50",
            "SMA 200",
        ],
        "Value": [
            latest_features["RSI"].iloc[0],
            latest_features["MACD"].iloc[0],
            latest_features["Volatility"].iloc[0],
            latest_features["SMA_20"].iloc[0],
            latest_features["SMA_50"].iloc[0],
            latest_features["SMA_200"].iloc[0],
        ],
    }
)

st.dataframe(
    indicator_data,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# HISTORICAL PRICE CHART
# ---------------------------------------------------------

st.divider()

st.header("Gold Futures Price History")

price_data = X.copy()

st.line_chart(
    price_data["Close"]
)

# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------

st.divider()

st.header("Model Performance Comparison")

comparison_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)

comparison_data = pd.read_csv(
    comparison_path
)

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# BACKTEST PERFORMANCE
# ---------------------------------------------------------

st.divider()

st.header("Historical Backtest Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Strategy Return",
        "98.85%"
    )

with col2:
    st.metric(
        "Buy & Hold Return",
        "108.13%"
    )

with col3:
    st.metric(
        "Signal Win Rate",
        "55.79%"
    )

st.caption(
    "Historical backtest period: 2023-06-06 to 2026-07-31"
)

# ---------------------------------------------------------
# BACKTEST RESULTS
# ---------------------------------------------------------

st.divider()

st.header("Backtest Results")

backtest_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "backtest_results.csv"
)

backtest_data = pd.read_csv(
    backtest_path
)

st.dataframe(
    backtest_data.tail(20),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Showing the 20 most recent observations from the historical backtest."
)

# ---------------------------------------------------------
# PREDICTION PROBABILITY
# ---------------------------------------------------------

st.divider()

st.header("Prediction Confidence")

st.progress(
    float(probability)
)

st.write(
    f"Probability of upward movement: **{probability:.2%}**"
)

# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.divider()

st.caption(
    "TradeVision AI is an experimental machine learning "
    "project for educational and research purposes. "
    "Model predictions are not guaranteed and should not "
    "be treated as financial advice."
)

st.subheader("Historical Backtest Performance")

try:
    backtest = pd.read_csv("data/processed/backtest_results.csv")

    backtest["Date"] = pd.to_datetime(backtest.iloc[:, 0])

    chart_data = backtest[
        ["Date", "Strategy_Equity", "Buy_Hold_Equity"]
    ].copy()

    chart_data = chart_data.set_index("Date")

    st.line_chart(chart_data)

    st.caption(
        "Historical comparison of the TradeVision AI strategy "
        "against a Buy & Hold benchmark."
    )

except Exception as e:
    st.warning(f"Backtest chart unavailable: {e}")

st.subheader("Model Performance Comparison")

try:
    model_results = pd.read_csv(
        "data/processed/model_comparison.csv"
    )

    st.dataframe(
        model_results,
        width="stretch",
        hide_index=True
    )

except Exception as e:
    st.warning(f"Model comparison unavailable: {e}")

st.subheader("Strategy Risk & Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Strategy Return",
    "98.85%"
)

col2.metric(
    "Buy & Hold Return",
    "108.13%"
)

col3.metric(
    "Maximum Drawdown",
    "-25.06%"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Sharpe Ratio",
    "1.2237"
)

col5.metric(
    "Signal Win Rate",
    "55.79%"
)

col6.metric(
    "Position Change Rate",
    "4.04%"
)