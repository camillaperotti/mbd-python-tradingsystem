import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import read_and_preprocess_data, load_model_and_scaler, preprocess_stock_data, make_prediction, get_trained_features

#Load stock data
data = read_and_preprocess_data()

#Define 5 available stocks
allowed_tickers = ["APPL", "ABT", "BRKR", "MSFT", "TSLA"]

#Sidebar: Stock Selection (Only Your 5 Tickers)
st.sidebar.header("Stock Market Selection")
tickers_in_data = sorted(set(data["Ticker"].unique()) & set(allowed_tickers))  # Ensure only available tickers are shown

ticker = st.sidebar.selectbox("Select a stock:", tickers_in_data)

# Filter data for the selected stock
data_ticker = preprocess_stock_data(data, ticker)

# COMPANY STOCKS
# Show Latest Stock Data
st.subheader(f"📊 Latest Data for {ticker}")
st.dataframe(data_ticker.tail())

# Stock Price Evolution Graph
fig = px.line(data_ticker, x="Date", y="Close", title=f"{ticker} Price Evolution", template="none")
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Closing Price")
st.plotly_chart(fig, use_container_width=True)


# PREDICTION
# Ensure "Price" column exists (rename if needed)
if "Adj. Close" in data_ticker.columns and "Price" not in data_ticker.columns:
    data_ticker.rename(columns={"Adj. Close": "Price"}, inplace=True)

# Generate lag features if they are missing
days = 4
for day in range(1, days + 1):
    column_name = f"Price d-{day}"
    if column_name not in data_ticker.columns:
        data_ticker[column_name] = data_ticker.groupby("Ticker")["Price"].shift(day)

#Drop NaN values (to match training)
data_ticker.dropna(inplace=True)

# Load Model & Scaler
model, scaler = load_model_and_scaler(ticker)

# Define the Prediction Function (Only in `1_Trading Prediction.py`)
def make_prediction(df_ticker, model, scaler):
    """
    Prepares the latest row, scales the features, and makes a stock price prediction.
    Returns the predicted value.
    """
    #Define the trained features
    trained_features = ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"]

    #Select only the trained features
    try:
        X_new = df_ticker.iloc[-1:][trained_features]
    except KeyError as e:
        st.error(f"Missing features: {e}")
        return None

    #Scale the features
    X_scaled = scaler.transform(X_new)

    #Make the prediction
    prediction = model.predict(X_scaled)[0]

    return prediction

#Make Prediction
if model and scaler:
    next_day_prediction = make_prediction(data_ticker, model, scaler)

    # Interpret the prediction (1 = Buy, 0 = Sell)
    recommendation = "📈 Buy" if next_day_prediction == 1 else "📉 Sell"

    # Display Prediction Results
    st.subheader("📈 Market Prediction")
    st.write(f"📊 **Predicted Movement for {ticker}: {'UP' if next_day_prediction == 1 else 'DOWN'}**")
    st.write(f"📢 **Recommendation: {recommendation}**")
else:
    st.error(f"❌ Model or Scaler files for {ticker} not found.")



# COMPANY DEEP DIVE

