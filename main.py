import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📊 Quant Trading Dashboard")

# ========================
# SIDEBAR
# ========================
asset = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL", "MSFT", "TSLA", "EURUSD=X", "GBPUSD=X"]
)

data = yf.download(asset, start="2020-01-01")

# ========================
# INDICATORS
# ========================
data["rsi"] = RSIIndicator(data["Close"]).rsi()
data["ema"] = EMAIndicator(data["Close"], window=20).ema_indicator()

# ========================
# ML MODEL (LIVE PREDICTION)
# ========================
data["target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
data = data.dropna()

features = ["rsi", "ema"]
X = data[features]
y = data["target"]

split = int(len(data) * 0.8)

model = RandomForestClassifier(n_estimators=200)
model.fit(X.iloc[:split], y.iloc[:split])

data["prediction"] = model.predict(X)

# ========================
# STRATEGY RETURNS
# ========================
data["strategy_returns"] = data["prediction"] * data["Close"].pct_change()
data["equity"] = (1 + data["strategy_returns"]).cumprod()

# ========================
# KPI
# ========================
col1, col2, col3 = st.columns(3)
col1.metric("Last Price", f"{data['Close'].iloc[-1]:.2f}")
col2.metric("RSI", f"{data['rsi'].iloc[-1]:.2f}")
col3.metric("Strategy Equity", f"{data['equity'].iloc[-1]:.2f}")

# ========================
# CHARTS
# ========================
st.subheader("Price & EMA")
st.line_chart(data[["Close", "ema"]])

st.subheader("ML Strategy Equity Curve")
st.line_chart(data["equity"])

st.subheader("Latest Predictions")
st.dataframe(data[["Close", "rsi", "prediction"]].tail(15))
