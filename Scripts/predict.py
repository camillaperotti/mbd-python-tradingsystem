#adapted Predictions.py to not be run on the terminal but in the streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from API import PySimFin
from datetime import datetime, timedelta
import logging

# Configure logging to show all messages and include timestamp
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Prediction:
    def __init__(self, ticker):
        self.ticker = ticker
        self.ticker_data = None
        self.model = None
        self.scaler = None
        self.model_path = f"models/model_{self.ticker}.pkl"
        self.scaler_path = f"models/scaler_{self.ticker}.pkl"

    def load_api(self):
        simfin = PySimFin("33cd76b1-b978-4165-8b91-5696ddea452a")

        end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")

        self.ticker_data = simfin.get_share_prices(self.ticker, start_date, end_date)
        logging.info(f"Successfully fetched data for {self.ticker} from {start_date} to {end_date}.")

    def transform_data(self):
        self.ticker_data = self.ticker_data[["ticker", "Date", "Adjusted Closing Price"]]
        self.ticker_data["Date"] = pd.to_datetime(self.ticker_data["Date"], format="%Y-%m-%d")
        self.ticker_data.rename(columns={"Adjusted Closing Price": "Price", "ticker": "Ticker"}, inplace=True)

        for day in range(1, 5):
            self.ticker_data[f"Price d-{day}"] = self.ticker_data.groupby("Ticker")["Price"].shift(day)

        self.ticker_data = self.ticker_data.dropna()
        self.ticker_data = self.ticker_data.drop(columns=["Ticker", "Date"])
        logging.info(f"Data transformation complete for {self.ticker} - ready for prediction.")

    def load_model(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            logging.error(f"Model or Scaler not found. Train the model first.")
            st.error("Model or Scaler not found.")
            return False

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logging.info(f"Model and scaler for {self.ticker} loaded successfully!\n")
        return True

    def predict_next_day(self):
        if self.model is None or self.scaler is None:
            logging.error(f"Model and scaler for {self.ticker} not loaded. Run load_model() first.")
            st.error(f"Model and scaler for {self.ticker} not loaded. Run load_model() first.")
            return None

        latest_features = self.ticker_data.iloc[-1:]
        latest_features_scaled = self.scaler.transform(latest_features)
        logging.debug(f"Features used for {self.ticker} prediction: {latest_features.values}")
        prediction = self.model.predict(latest_features_scaled)

        return "UP" if prediction[0] == 1 else "DOWN"


# STREAMLIT APP
st.title("📊 Stock Price Movement Predictor")

# User input for the stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", value="AAPL")

if st.button("Predict"):
    if ticker:
        st.write(f"### Predicting for {ticker}...")

        company = Prediction(ticker)
        company.load_api()
        company.transform_data()

        if company.load_model():
            prediction = company.predict_next_day()
            if prediction:
                st.subheader(f"Prediction for {ticker}: {prediction}")
                logging.info(f"Predicted movement for {ticker}: {prediction}\n")