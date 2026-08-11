# TradeVision AI — Project Progress

## Project Goal

TradeVision AI is an end-to-end machine learning project focused on predicting upward price movement in gold futures (`GC=F`).

The project was developed to demonstrate practical skills in:

- Python development
- Data collection and processing
- Feature engineering
- Technical analysis
- Machine learning
- Model evaluation
- Statistical analysis
- Backtesting
- Risk analysis
- Streamlit dashboard development
- Git and GitHub

## Phase 1 — Project Setup

- Created the TradeVision AI project repository.
- Created the Python virtual environment.
- Configured the project structure.
- Created the `requirements.txt` file.
- Added Git version control.
- Connected the local repository to GitHub.

## Phase 2 — Data Collection & Processing

- Collected historical gold futures data using `yfinance`.
- Used the `GC=F` ticker.
- Stored raw market data in the project data directory.
- Processed and validated the dataset.
- Created a structured machine learning dataset.

## Phase 3 — Feature Engineering

Created market-derived features including:

- Daily Return
- Log Return
- SMA 20
- SMA 50
- SMA 200
- EMA 20
- RSI
- MACD
- MACD Signal
- Bollinger Bands
- ATR
- Volatility
- Trading Volume
- Day of Week
- Month
- Quarter

The target variable represents the direction of the next trading period.

## Phase 4 — Machine Learning

Evaluated multiple machine learning approaches:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest
- XGBoost
- Reduced Feature Random Forest
- Naive Baseline

### Best Model

Logistic Regression achieved the highest recorded accuracy:

**55.67%**

The results demonstrated that increased model complexity did not necessarily produce better predictive performance on this dataset.

## Phase 5 — Model Evaluation & Analysis

Completed:

- Train/test preparation
- Model comparison
- Hyperparameter tuning
- Feature importance analysis
- Prediction probability analysis
- Statistical testing
- Model evaluation

### Statistical Testing

The difference between bullish-signal returns and no-signal returns was tested.

The result was **not statistically significant at the 5% level**.

This provides an important limitation to the predictive interpretation of the model.

## Phase 6 — Backtesting

The trading strategy was evaluated using historical data from:

**2023-06-06 to 2026-07-31**

Recorded results:

| Metric | TradeVision AI | Buy & Hold |
|---|---:|---:|
| Total Return | 98.85% | 108.13% |
| Annualized Return | 25.69% | 26.23% |
| Maximum Drawdown | -25.06% | -25.06% |

Additional strategy metrics:

- Sharpe Ratio: **1.2237**
- Signal Win Rate: **55.79%**
- Position Change Rate: **4.04%**
- Average Days Between Position Changes: **24.78**

The strategy produced a strong historical return but did not outperform the Buy & Hold benchmark over the tested period.

## Phase 7 — Dashboard Development

Developed an interactive Streamlit dashboard displaying:

- Latest model prediction
- Upward-movement probability
- Market snapshot
- Technical indicators
- Gold futures price history
- Model performance comparison
- Model performance visualization
- Historical backtest performance
- Strategy versus Buy & Hold performance
- Backtest results
- Strategy risk metrics
- Prediction confidence
- Recent backtest observations

The dashboard was deployed using Streamlit Community Cloud.

### Live Dashboard

https://tradevision-ai-im5qbqcqullsxmfuar3wrr.streamlit.app/

## Phase 8 — GitHub & Portfolio Finalization

Completed:

- GitHub repository setup
- GitHub branch configuration
- Repository synchronization
- README documentation
- GitHub repository description
- Repository topics
- Streamlit deployment link
- `.gitignore` improvements
- Dashboard documentation
- Project progress documentation

## Current Project Status

**TradeVision AI is deployed and operational.**

The project currently includes:

- Machine learning pipeline
- Feature engineering pipeline
- Multiple trained models
- Model evaluation
- Statistical analysis
- Historical backtesting
- Risk analysis
- Interactive Streamlit dashboard
- GitHub repository documentation

## Important Limitations

TradeVision AI is an experimental machine learning project.

The model's predictive performance is modestly above the naive baseline, and statistical testing did not identify a statistically significant difference between bullish and non-bullish signal returns.

Historical backtesting does not guarantee future performance.

The project is intended to demonstrate practical machine learning, data analysis, financial-data engineering, and software development skills rather than provide guaranteed trading signals.