import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📊 Quant Trading Dashboard")

asset = st.sidebar.selectbox("Select Asset", ["AAPL", "MSFT", "TSLA", "EURUSD=X", "GBPUSD=X"])

# Download e pulizia immediata
df = yf.download(asset, start="2020-01-01")

# Se yfinance restituisce colonne doppie, le appiattiamo
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# TRUCCO FINALE: Estraiamo 'Close' come una serie 1D pura
close_data = df['Close'].iloc[:, 0] if len(df['Close'].shape) > 1 else df['Close']
close_data = pd.Series(close_data).astype(float)

# Calcolo Indicatori
df["rsi"] = RSIIndicator(close_data).rsi()
df["ema"] = EMAIndicator(close_data, window=20).ema_indicator()

# Modello ML
df["target"] = np.where(close_data.shift(-1) > close_data, 1, 0)
df = df.dropna()

X = df[["rsi", "ema"]]
y = df["target"]
split = int(len(df) * 0.8)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X.iloc[:split], y.iloc[:split])
df["prediction"] = model.predict(X)

# Grafici
st.metric("Price", f"{close_data.iloc[-1]:.2f}")
st.line_chart(df[["Close", "ema"]])
st.subheader("Latest Predictions")
st.dataframe(df.tail(10))