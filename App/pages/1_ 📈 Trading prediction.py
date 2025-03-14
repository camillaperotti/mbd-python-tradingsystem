import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta

# Ensure "Scripts/" is in the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) ## mac: _init_, windows: __init__

# Import custom scripts
from Scripts.API import PySimFin
from Scripts.Exceptions import SimFinError, ResourceNotFoundError
from Scripts.predict import Prediction
from utils import read_and_preprocess_data, preprocess_stock_data

#Load stock data (Historical data for graph)
data = read_and_preprocess_data()

#Define 5 available stocks
allowed_tickers = ["AAPL", "ABT", "BRKR", "MSFT", "TSLA"]

##SIDEBAR: Stock Selection (Only Your 5 Tickers)
st.sidebar.header("Stock Market Selection")
tickers_in_data = sorted(set(data["Ticker"].unique()) & set(allowed_tickers))  # Ensure only available tickers are shown
ticker = st.sidebar.selectbox("Select a stock:", tickers_in_data)

# COMPANY STOCKS
# Show Latest Stock Data
st.subheader(f"📊 Latest Historical Data for {ticker}")
data_ticker = preprocess_stock_data(data, ticker)

#GRAPH, Historical data
# Stock Price Evolution Graph
fig = px.line(data_ticker, x="Date", y="Close", 
              title=f"{ticker} Price Evolution - Historical Data", template="none")
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Closing Price")
st.plotly_chart(fig, use_container_width=True)

##PREDICTION
# STREAMLIT APP
st.title("📊 Stock Price Movement Predictor")

if st.button("Predict Stock Movement"):
    st.write(f"### Predicting for {ticker}...")

    company = Prediction(ticker)
    company.load_api()
    company.transform_data()

    if company.load_model():
        prediction = company.predict_next_day()
        if prediction:
            st.subheader(f"📈 Prediction for {ticker}: {prediction}")
