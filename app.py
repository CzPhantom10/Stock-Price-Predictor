import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("Stock Price Prediction Dashboard")

stock = st.sidebar.selectbox(
    "Select Stock",
    ["NVDA", "AAPL", "TSLA"]
)

pred_path = f"data/predictions_{stock}.csv"
feat_path = f"data/feature_importance_{stock}.csv"
raw_path = f"data/raw_data_{stock}.csv"

if not (os.path.exists(pred_path) and os.path.exists(feat_path)):
    st.error(f"Model outputs not found for {stock}. Train the model first.")
    st.stop()

pred_df = pd.read_csv(pred_path)
feature_df = pd.read_csv(feat_path)
raw_df = pd.read_csv(raw_path)

st.subheader(f"{stock} Price Prediction")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pred_df["target"], label="Actual Price")
ax.plot(pred_df["predicted"], label="Predicted Price")
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

st.subheader("Model Metrics")

mae = abs(pred_df["target"] - pred_df["predicted"]).mean()
st.metric("Mean Absolute Error", f"${mae:.2f}")

with st.expander("View Recent Raw Data"):
    st.dataframe(raw_df.tail(50))
