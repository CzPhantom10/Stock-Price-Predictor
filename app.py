import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide"
)

st.title("Stock Price Prediction Dashboard")

st.markdown(
    """
    This dashboard visualizes machine learning based stock price predictions
    using historical market data and technical indicators.
    """
)

stock = st.sidebar.selectbox(
    "Select Stock",
    ["NVDA", "AAPL", "TSLA"]
)

raw_path = f"data/raw_data_{stock}.csv"
pred_path = f"data/predictions_{stock}.csv"
feat_path = f"data/feature_importance_{stock}.csv"

if not (os.path.exists(raw_path) and os.path.exists(pred_path) and os.path.exists(feat_path)):
    st.error("Required data files not found. Please run fetch_data.py and train_model.py first.")
    st.stop()

raw_df = pd.read_csv(raw_path)
pred_df = pd.read_csv(pred_path)
feature_df = pd.read_csv(feat_path)

mae = abs(pred_df["target"] - pred_df["predicted"]).mean()
latest_actual = raw_df["close"].iloc[-1]
latest_predicted = pred_df["predicted"].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Selected Stock", stock)
col2.metric("Mean Absolute Error", f"${mae:.2f}")
col3.metric("Last Predicted Close", f"${latest_predicted:.2f}")
st.subheader("Actual vs Predicted Closing Price")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pred_df["target"].values, label="Actual Price")
ax.plot(pred_df["predicted"].values, label="Predicted Price")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)
st.subheader("Feature Importance")

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.barh(feature_df["feature"], feature_df["importance"])
ax2.invert_yaxis()
ax2.set_xlabel("Importance")

st.pyplot(fig2)
st.subheader("Next Trading Day Prediction")

st.markdown(
    """
    The value below represents the model's prediction for the **next trading day**
    based on the most recent available market data.
    """
)

st.metric(
    label="Predicted Next Day Closing Price",
    value=f"${latest_predicted:.2f}"
)

with st.expander("View Recent Raw Market Data"):
    st.dataframe(raw_df.tail(50))
