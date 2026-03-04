import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_model
from features import add_features, create_labels


df = pd.read_csv("data/price_data.csv")

df = add_features(df)
df = create_labels(df)

X = df[["returns","sma_10","sma_50","volatility"]]

y = df["signal"] + 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

model = build_model(X.shape[1])

model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)

model.save("models/trading_model.h5")
