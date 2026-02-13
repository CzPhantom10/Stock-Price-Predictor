import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from indicators import add_indicators
STOCKS = ["NVDA", "AAPL", "TSLA"]

FEATURES = [
    "close",
    "volume",
    "volatility",
    "ma_5",
    "ma_10",
    "ma_20",
    "rsi",
]

for stock in STOCKS:
    print(f"\nTraining model for {stock}")

    # Load raw data
    df = pd.read_csv(f"../data/raw_data_{stock}.csv")

    # Feature engineering
    df = add_indicators(df)

    X = df[FEATURES]
    y = df["target"]

    # Train-test split (time series safe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"{stock} MAE: ${mae:.2f}")

    # Save predictions
    df_test = df.iloc[-len(preds):].copy()
    df_test["predicted"] = preds
    df_test.to_csv(f"../data/predictions_{stock}.csv", index=False)

    # Save feature importance
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(f"../data/feature_importance_{stock}.csv", index=False)

print("Training completed for all stocks")
