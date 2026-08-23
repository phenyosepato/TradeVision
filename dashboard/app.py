import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.load_model import load_model
from src.models.prepare_data import prepare_dataset


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="TradeVision",
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

prediction = model.predict(latest_features)[0]

probability = model.predict_proba(latest_features)[0][1]


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("TradeVision")

st.subheader("Machine Learning Market Prediction Dashboard")

st.caption(
    "Gold Futures (GC=F) • Logistic Regression • "
    f"Latest Market Data: {latest_date.date()}"
)

st.divider()


# ---------------------------------------------------------
# KEY METRICS
# ---------------------------------------------------------

st.header("Market Snapshot")

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
    st.success("🟢 BULLISH — Model predicts upward movement.")
else:
    st.warning("🟡 NOT BULLISH — Model does not predict upward movement.")

st.write(
    f"Latest available market date: **{latest_date.date()}**"
)
if prediction == 1:
    st.info(
        "The model classifies the latest market observation as bullish, "
        "meaning it predicts a higher probability of upward price movement "
        "for the next trading period."
    )
else:
    st.info(
        "The model does not classify the latest market observation as bullish. "
        "This means the predicted probability of upward price movement "
        "does not meet the model's bullish classification threshold."
    )


# ---------------------------------------------------------
# PREDICTION CONFIDENCE
# ---------------------------------------------------------

st.subheader("Prediction Confidence")

st.progress(float(probability))

st.write(
    f"Probability of upward movement: **{probability:.2%}**"
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
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# GOLD PRICE HISTORY
# ---------------------------------------------------------

st.divider()

st.header("Gold Futures Price History")

price_data = X[["Close"]].copy()

st.line_chart(
    price_data,
    y="Close",
    width="stretch",
)


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.divider()

st.header("Model Performance Comparison")

comparison_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)

comparison_data = pd.read_csv(comparison_path)

comparison_data.columns = [
    column.strip()
    for column in comparison_data.columns
]

st.bar_chart(
    comparison_data,
    x="Model",
    y="Accuracy",
    width="stretch",
)

st.dataframe(
    comparison_data,
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# BACKTEST PERFORMANCE
# ---------------------------------------------------------

st.divider()

st.header("Historical Backtest Performance")

backtest_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "backtest_results.csv"
)

backtest_data = pd.read_csv(backtest_path)

backtest_data["Date"] = pd.to_datetime(
    backtest_data.iloc[:, 0]
)

chart_data = backtest_data[
    ["Date", "Strategy_Equity", "Buy_Hold_Equity"]
].copy()

chart_data = chart_data.set_index("Date")

st.line_chart(
    chart_data,
    width="stretch",
)
st.subheader("Strategy vs Buy & Hold")

st.caption(
    "Growth of $1 invested in the TradeVision AI strategy "
    "compared with a Buy & Hold strategy."
)

st.caption(
    "Historical comparison of the TradeVision AI strategy "
    "against a Buy & Hold benchmark."
)


# ---------------------------------------------------------
# BACKTEST SUMMARY
# ---------------------------------------------------------

st.subheader("Backtest Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Strategy Return",
        "98.85%",
    )

with col2:
    st.metric(
        "Buy & Hold Return",
        "108.13%",
    )

with col3:
    st.metric(
        "Signal Win Rate",
        "55.79%",
    )

st.caption(
    "Historical backtest period: "
    "2023-06-06 to 2026-07-31"
)


# ---------------------------------------------------------
# STRATEGY RISK & PERFORMANCE
# ---------------------------------------------------------

st.subheader("Strategy Risk & Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Maximum Drawdown",
        "-25.06%",
    )

with col2:
    st.metric(
        "Sharpe Ratio",
        "1.2237",
    )

with col3:
    st.metric(
        "Position Change Rate",
        "4.04%",
    )


# ---------------------------------------------------------
# RECENT BACKTEST OBSERVATIONS
# ---------------------------------------------------------

st.subheader("Recent Backtest Observations")

st.caption(
    "Showing the 20 most recent observations from the historical backtest."
)

st.dataframe(
    backtest_data.tail(20),
    width="stretch",
    hide_index=True,
)
st.subheader("Indicator Overview")

indicator_chart = indicator_data.set_index("Indicator")

st.bar_chart(
    indicator_chart,
    width="stretch",
)


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.divider()

st.subheader("About TradeVision AI")

st.write(
    "TradeVision AI is an experimental machine learning project "
    "designed to demonstrate financial data analysis, feature engineering, "
    "model evaluation, backtesting, and dashboard development."
)

st.caption(
    "Educational and research purposes only. Model predictions are not "
    "guaranteed and should not be treated as financial advice."
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • XGBoost • Streamlit"
)