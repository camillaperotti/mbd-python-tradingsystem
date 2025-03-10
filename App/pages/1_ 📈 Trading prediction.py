import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import pickle

from utils import read_and_preprocess_data, load_model_and_scaler, preprocess_stock_data, make_prediction

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
# 📊 Show Latest Stock Data
st.subheader(f"📊 Latest Data for {ticker}")
st.dataframe(data_ticker.tail())

# 📉 Stock Price Evolution Graph
fig = px.line(data_ticker, x="Date", y="Close", title=f"{ticker} Price Evolution", template="none")
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Closing Price")
st.plotly_chart(fig, use_container_width=True)


# PREDICTION



# COMPANY DEEP DIVE

