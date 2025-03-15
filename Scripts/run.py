import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) ## mac: _init_, windows: __init__

# Configure logging to show all messages and include timestamp
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Company:
    def __init__(self, ticker):
        
        self.ticker = ticker
        self.prices = None  # Data will be assigned later after ETL
        self.model = None
        self.scaler = None

        # Get the script's directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Default paths (relative to script location)
        self.model_path = os.path.join(base_dir, "Scripts", "models", f"model_{self.ticker}.pkl")
        self.scaler_path = os.path.join(base_dir, "Scripts", "models", f"scaler_{self.ticker}.pkl")

    def load_data(self, filepath):
        # Load full dataset and filter only relevant 5 companies
        prices_all = pd.read_csv(filepath, delimiter=';')
        prices = prices_all[prices_all["SimFinId"].isin([1253240, 111052, 63877, 56317, 59265])]
        logging.info(f"Successfully loaded dataframe prices for 5 selected companies.")
        return prices

    def process_data(self, prices):
        # Handle missing values, convert to correct data types
        prices["Dividend"] = prices["Dividend"].fillna(0)
        prices["Shares Outstanding"] = prices["Shares Outstanding"].astype(int)
        prices["Date"] = pd.to_datetime(prices["Date"], format="%Y-%m-%d")
        logging.info(f"Successfully cleaned dataframe prices for 5 selected companies.")
        return prices
    
    def save_data(self, prices, output_filepath):
        prices.to_csv(output_filepath, index=False)
        logging.info(f"Successfully saved processed data to {output_filepath}")

    def etl_pipeline(self, filepath, output_filepath):
        # Run the ETL pipeline (Extract, Transform, Load)
        logging.debug(f"Starting ETL for {self.ticker}...")
        self.prices = self.load_data(filepath)
        self.prices = self.process_data(self.prices)
        self.save_data(self.prices, output_filepath)
        logging.info(f"ETL Completed!\n")

    def prepare_data(self):
        logging.debug(f"Preparing data for {self.ticker} for ML")

        # Filter for the current ticker only
        prices = self.prices[self.prices["Ticker"] == self.ticker]
        prices = prices.copy()

        # Sort and clean
        prices.sort_values(by=["Ticker", "Date"], inplace=True) 
        prices.reset_index(drop=True, inplace=True) 
        prices.drop(columns=["SimFinId", "Open", "High", "Low", "Close", "Volume", "Dividend", "Shares Outstanding"], inplace=True)
        prices.rename(columns={"Adj. Close": "Price"}, inplace=True)

        # Add last 4 days' prices
        for day in range(1, 5):
            prices[f"Price d-{day}"] = prices.groupby("Ticker")["Price"].shift(day)
        
        # Drop missing values
        prices = prices.dropna()
        
        # Define target variable (if price goes up the next day)
        prices["Price_Up"] = (prices["Price"].shift(-1) > prices["Price"]).astype(int)
        
        # Drop last row per ticker
        prices = prices.drop(prices.groupby("Ticker").tail(1).index).reset_index(drop=True)
        
        # Save processed prices
        self.prices = prices  
        logging.info(f"Data preparation for ML completed!\n")

    def train_model(self):
        # Logistic Regression model
        logging.debug(f"Training model for {self.ticker}...")

        # Define features and target
        X = self.prices.drop(columns=["Ticker", "Date", "Price_Up"])
        y = self.prices["Price_Up"]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train Logistic Regression model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy for {self.ticker}: {accuracy:.4f}\n")

        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        logging.info(f"Model saved at {self.model_path}")
        logging.info(f"Scaler saved at {self.scaler_path}\n")

    def load_model(self):
        # Load trained model and scaler for this ticker
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            logging.error(f"Model or Scaler for {self.ticker} not found. Train the model first.")
            raise FileNotFoundError("Model or Scaler not found. Train the model first.")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logging.info(f"Model and scaler for {self.ticker} loaded successfully!\n")

    def predict_next_day(self):
        # Make a prediction for the next day's price movement
        if self.model is None or self.scaler is None:
            logging.error(f"Model or Scaler for {self.ticker} not loaded. Run load_model() first.")
            raise ValueError("Model and scaler not loaded. Run load_model() first.")

        # Use the last available row for prediction
        latest_features = self.prices.iloc[-1:].drop(columns=["Price_Up", "Date","Ticker"])
        latest_features_scaled = self.scaler.transform(latest_features)
        logging.info(f"Features used for {self.ticker} prediction: {latest_features.values}")
        prediction = self.model.predict(latest_features_scaled)
        return "UP" if prediction[0] == 1 else "DOWN"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL and ML for a specific stock ticker")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol to process (e.g., AAPL, TSLA)")
    args = parser.parse_args()

    # File paths
   # Get the script's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default paths (relative to script location)
    raw_data_path = os.path.join(base_dir, "ETL", "data", "us-shareprices-daily.csv")
    processed_data_path = os.path.join(base_dir, "ETL", "pricesbruker_output.csv")

    logging.debug(f"========== Running for {args.ticker} ==========")
    company = Company(args.ticker)

    # Run ETL
    company.etl_pipeline(raw_data_path, processed_data_path)

    # Prepare and train model
    company.prepare_data()
    company.train_model()

    # Load and predict
    company.load_model()
    prediction = company.predict_next_day()
    logging.info(f"Predicted movement for {args.ticker}: {prediction}\n")