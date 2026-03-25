from src.data_loader import load_data
from src.features import engineer_features
from src.train import train_model, FEATURE_COLS
from src.evaluate import classification_report
from src.backtest import run_backtest, financial_metrics
import numpy as np

if __name__ == "__main__":
    # --- Config ---
    TICKER    = "SPY"
    START = "2015-01-01"
    END   = "2023-01-01"
    HORIZON   = 5       # forward-return horizon h (days)
    THRESHOLD = 0.005    # τ for Buy/Sell labeling

    # --- Pipeline ---
    print("Loading data...")
    raw = load_data(TICKER, START, END)

    print("Engineering features...")
    feat_df = engineer_features(raw, horizon=HORIZON, threshold=THRESHOLD)

    print("Training model...")
    model, scaler, test_df, X_test = train_model(feat_df, epochs=100)

    print("\n--- Classification Metrics ---")
    y_true  = test_df["Label"].values.astype(int)
    y_pred  = classification_report(model, X_test, y_true)

    print("\n--- Financial Metrics ---")
    backtest_df = run_backtest(test_df, y_pred)
    financial_metrics(backtest_df)
