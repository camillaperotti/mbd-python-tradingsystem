#Designed to be run in terminal

import argparse
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

class PredictionT:
    def __init__(self, ticker):
        self.ticker = ticker
        self.ticker_data = None #so we can refer to it easily
        self.model = None #so we can refer to it easily
        self.scaler = None #so we can refer to it easily
        self.model_path = f"models/model_{self.ticker}.pkl"
        self.scaler_path = f"models/scaler_{self.ticker}.pkl"

    def load_api(self):
        #Load api wrapper
        simfin = PySimFin("33cd76b1-b978-4165-8b91-5696ddea452a")
        
        # Get today's date and compute the date range
        end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # Yesterday's date
        start_date = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")  # 14 days ago, to have enough data to not have any null values

        # Fetch share prices for the past week
        self.ticker_data = simfin.get_share_prices(self.ticker, start_date, end_date)
        logging.info(f"Successfully fetched data for {self.ticker} from {start_date} to {end_date}.")

    def transform_data(self):
        # Drop irrelevant columns
        self.ticker_data = self.ticker_data[["ticker","Date","Adjusted Closing Price"]]

        # Data cleaning
        self.ticker_data["Date"] = pd.to_datetime(self.ticker_data["Date"], format="%Y-%m-%d")
        self.ticker_data.rename(columns={"Adjusted Closing Price": "Price"}, inplace=True)
        self.ticker_data.rename(columns={"ticker": "Ticker"}, inplace=True)

        # Add last 4 days' prices
        for day in range(1, 5):
            self.ticker_data[f"Price d-{day}"] = self.ticker_data.groupby("Ticker")["Price"].shift(day)

        # Drop missing values
        self.ticker_data = self.ticker_data.dropna()

        # Drop "Date" and "Ticker"
        self.ticker_data = self.ticker_data.drop(columns=["Ticker","Date"])
        logging.info(f"Data transformation complete for {self.ticker} - ready for prediction.")

    def load_model(self):
        # Load trained model and scaler for this ticker
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            logging.error(f"Model or Scaler not found. Train the model first.")
            raise FileNotFoundError("Model or Scaler not found.")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logging.info(f"Model and scaler for {self.ticker} loaded successfully!\n")

    def predict_next_day(self):
        # Make a prediction for the next day's price movement
        if self.model is None or self.scaler is None:
            logging.error(f"Model and scaler for {self.ticker} not loaded. Run load_model() first.")
            raise ValueError(f"Model and scaler for {self.ticker} not loaded.")

        # Use the last available row for prediction
        latest_features = self.ticker_data.iloc[-1:]
        latest_features_scaled = self.scaler.transform(latest_features)

        logging.info(f"Features used for {self.ticker} prediction: {latest_features.values}")

        prediction = self.model.predict(latest_features_scaled)
        return "UP" if prediction[0] == 1 else "DOWN"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next days's price for a specific stock ticker")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol to process (e.g., AAPL, TSLA)")
    args = parser.parse_args()

    logging.debug(f"========== Predicting for {args.ticker} ==========")
    company = PredictionT(args.ticker)

    # Load api
    company.load_api()

    # Do ETL
    company.transform_data()

    # Load and predict
    company.load_model()
    prediction = company.predict_next_day()
    logging.info(f"Predicted movement for {args.ticker}: {prediction}\n")