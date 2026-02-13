# Stock Price Prediction using Machine Learning and Alpaca API

## Overview
This project is an end-to-end stock price prediction system built using historical market data and machine learning.  
It fetches real stock market data using the Alpaca Market API, engineers technical indicators, trains a Random Forest regression model, and visualizes predictions through a Streamlit dashboard.

The system supports multiple stocks and provides insights such as predicted prices, feature importance, and model performance metrics.

---

## Features
- Fetches historical stock market data using Alpaca API (Paper Trading)
- Supports multiple stocks (NVDA, AAPL, TSLA)
- Technical indicators:
  - Volatility
  - Moving Averages (5, 10, 20)
  - Relative Strength Index (RSI)
- Machine Learning model using Random Forest Regressor
- Time-series safe train-test split
- Model evaluation using Mean Absolute Error (MAE)
- Interactive Streamlit dashboard:
  - Stock selection
  - Actual vs Predicted price visualization
  - Feature importance visualization
  - Model performance metrics
  - Raw data inspection

---

## Project Structure
```text
Stock_Price_Predictor/
│
├── data/
│ ├── raw_data_NVDA.csv
│ ├── raw_data_AAPL.csv
│ ├── raw_data_TSLA.csv
│ ├── predictions_NVDA.csv
│ ├── predictions_AAPL.csv
│ ├── predictions_TSLA.csv
│ ├── feature_importance_NVDA.csv
│ ├── feature_importance_AAPL.csv
│ └── feature_importance_TSLA.csv
│
├── src/
│ ├── fetch_data.py
│ ├── indicators.py
│ ├── train_model.py
│ └── visualize.py
│
├── app.py
├── .env
├── requirements.txt
├── .gitignore
└── README.md
```


---

## Tech Stack
- Python
- Alpaca Market API
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Stock_Price_Predictor
```

### 2. Create and activate virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a .env file in the project root:

```bash
ALPACA_API_KEY=your_paper_api_key
ALPACA_API_SECRET=your_paper_secret_key
```

## Running the Project

**Step 1: Fetch market data**
```bash
python src/fetch_data.py
```

**Step 2: Train models for all stocks**
```bash
python src/train_model.py
```

**Step 3: Launch Streamlit dashboard**
```bash
streamlit run app.py
```

## Model Details

Algorithm: Random Forest Regressor

Prediction target: Next-day closing price

Evaluation metric: Mean Absolute Error (MAE)

Data split: 80% training, 20% testing (no shuffling)

## Use Cases

- Educational machine learning project
- Finance and data science portfolio project
- Hackathon-ready stock analysis system
- Demonstration of ML + API + dashboard integration

## Disclaimer

This project is for educational purposes only and does not constitute financial or investment advice.