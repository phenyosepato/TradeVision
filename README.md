TradeVision AI
Machine Learning Market Prediction & Trading Analytics

TradeVision AI is an end-to-end machine learning project that analyzes historical gold futures data (GC=F) and predicts the probability of upward price movement.

The project combines financial data processing, technical indicators, machine learning model comparison, probability analysis, historical backtesting, risk analysis, and an interactive Streamlit dashboard.

Important: TradeVision AI is an experimental machine learning project and is not financial advice or a guaranteed trading system.

Dashboard

TradeVision AI includes an interactive Streamlit dashboard for viewing:

Latest model prediction
Upward-movement probability
Technical indicators
Historical gold price data
Model performance comparison
Historical backtest performance
Strategy risk metrics
Prediction confidence
Run the Dashboard

Activate the project virtual environment and run:

streamlit run dashboard/app.py
Project Overview

The project uses historical gold futures data to build a machine learning pipeline for predicting whether the next trading period is likely to experience upward price movement.

The workflow includes:

Market data collection
Data validation and profiling
Feature engineering
Technical indicator generation
Train/test preparation
Multiple machine learning models
Hyperparameter tuning
Probability calibration
Statistical analysis
Historical backtesting
Performance and risk analysis
Interactive dashboard deployment
Technical Indicators

TradeVision AI uses several market-derived features, including:

SMA 20
SMA 50
SMA 200
EMA 20
RSI
MACD
MACD Signal
Bollinger Bands
ATR
Daily Return
Log Return
Volatility
Trading Volume
Day of Week
Month
Quarter
Machine Learning Models

Several models were evaluated:

Model	Accuracy
Logistic Regression	55.67%
Naive Baseline	55.54%
Reduced Feature Random Forest	48.49%
Decision Tree	46.47%
Tuned Random Forest	44.84%
XGBoost	44.21%
Random Forest	42.95%

Best-performing model: Logistic Regression

Accuracy: 55.67%

The model comparison demonstrates that more complex models did not necessarily outperform the simpler Logistic Regression model on this dataset.

Backtesting

The strategy was evaluated on historical data from:

2023-06-06 to 2026-07-31

Metric	TradeVision AI	Buy & Hold
Total Return	98.85%	108.13%
Annualized Return	25.69%	26.23%
Maximum Drawdown	-25.06%	-25.06%

Additional strategy metrics:

Sharpe Ratio: 1.2237
Signal Win Rate: 55.79%
Position Change Rate: 4.04%
Average Days Between Position Changes: 24.78

The backtest shows that the strategy produced a strong historical return, although it did not outperform the Buy & Hold benchmark over the tested period.

Statistical Analysis

The project also evaluates whether bullish signals produced meaningfully different subsequent returns.

Bullish signal mean return: 0.1022%
No-signal mean return: 0.0590%
Difference: 0.0432%
T-statistic: 0.2774
P-value: 0.7836

Result: Not statistically significant at the 5% level.

This is an important limitation of the model: the observed difference between bullish and non-bullish signal returns was not statistically significant in the tested data.

Project Structure
TradeVision-AI/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── permutation_importance.csv
├── saved_models/
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── tradevision_final.pkl
│   └── xgboost.pkl
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── .gitignore
├── PROJECT_PROGRESS.md
├── README.md
└── requirements.txt
Technologies
Python
Pandas
NumPy
Scikit-learn
XGBoost
yfinance
TA
Matplotlib
Streamlit
Git & GitHub
Running the Project

Clone the repository:

git clone https://github.com/phenyosepato/TradeVision-AI.git

Navigate into the project:

cd TradeVision-AI

Create a virtual environment:

python -m venv .venv

Install dependencies:

pip install -r requirements.txt

Run the dashboard:

streamlit run dashboard/app.py
Limitations

TradeVision AI should be treated as an experimental machine learning project.

The model's predictive accuracy is only modestly above the naive baseline, and statistical testing did not find a significant difference between bullish and non-bullish signal returns.

Historical backtesting also does not guarantee future performance.

The project is intended to demonstrate practical machine learning, data analysis, model evaluation, and financial-data engineering skills rather than provide guaranteed trading signals.

Author

Phenyo Sepato

GitHub: @phenyosepato