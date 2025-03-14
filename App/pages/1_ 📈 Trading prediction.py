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

# Define API Keys
SIMFIN_API_KEY = "33cd76b1-b978-4165-8b91-5696ddea452a"

# Initialize API Clients
simfin_client = PySimFin(SIMFIN_API_KEY)

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

### 📑 FINANCIAL STATEMENTS
st.subheader("📑 Financial Statements")

statement_type = st.selectbox("Select Financial Statement:", ["Profit & Loss (PL)", "Balance Sheet (BS)", "Cash Flow (CF)", "Derived Ratios (DERIVED)"])
statement_code = {"Profit & Loss (PL)": "PL", "Balance Sheet (BS)": "BS", "Cash Flow (CF)": "CF", "Derived Ratios (DERIVED)": "DERIVED"}[statement_type]

if st.button("Fetch Financial Data"):
    st.write(f"Fetching {statement_type} for {ticker}...")

    # Define date range (last 2 years)
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    financial_data = simfin_client.get_financial_statement(ticker, statement_code, start_date, end_date)

    if not financial_data.empty:
        st.dataframe(financial_data.head(10))
    else:
        st.warning(f"⚠ No data available for {statement_type}.")

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


##
#MORE INFO
st.title("More Info")
if st.button("Predict Stock Movement"):
    st.write(f"### Predicting for {ticker}...")

    company_info = PySimFin("33cd76b1-b978-4165-8b91-5696ddea452a")
    company_info.get_financial_statement()

