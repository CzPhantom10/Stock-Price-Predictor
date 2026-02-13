import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/predictions.csv")

plt.figure(figsize=(12,6))
plt.plot(df["target"].values, label="Actual Price")
plt.plot(df["predicted"].values, label="Predicted Price")
plt.title("NVDA Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
