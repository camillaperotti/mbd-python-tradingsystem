import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))) ## mac: _init_, windows: __init__

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data #it will be loaded once and then saved
def read_and_preprocess_data() -> pd.DataFrame:
    """
    Loads and preprocesses stock market data from the consolidated CSV file.
    Returns a Pandas DataFrame.
    """
    # Get the script's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default paths (relative to script location)
    csv_path = os.path.join(base_dir, "ETL", "prices_output.csv")

    return pd.read_csv(csv_path)

def load_model_and_scaler(ticker) -> tuple:
    """
    Loads the trained model and scaler for a selected stock ticker.
    Returns a tuple (model, scaler).
    """
    model_path = f"Scripts/models/model_{ticker}.pkl"
    scaler_path = f"Scripts/models/scaler_{ticker}.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def preprocess_stock_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Filters stock data for the selected ticker.
    Returns a DataFrame with only relevant data.
    """
    return data[data["Ticker"] == ticker].copy()

# Define the exact feature names used during training for each stock
TRAINED_FEATURES = {
    "APPL": ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"],
    "ABT": ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"],
    "BRKR": ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"],
    "MSFT": ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"],
    "TSLA": ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"]
}

def get_trained_features(ticker):
    """
    Returns the hardcoded feature names that were used for training the model.
    """
    return TRAINED_FEATURES.get(ticker, [])


def make_prediction(df_ticker: pd.DataFrame, model, scaler) -> float:
    """
    Uses the trained model and scaler to make a stock price prediction.
    Returns the predicted value.
    """
    #Get trained feature names
    trained_features = ["Price", "Price d-1", "Price d-2", "Price d-3", "Price d-4"]

    #Drop unnecessary columns to match training
    X_new = df_ticker.iloc[-1:].drop(columns=["Date", "Price_Up"], errors="ignore")

    #Ensure only trained features are used
    X_new = X_new[trained_features]

    #Scale the features
    X_scaled = scaler.transform(X_new)

    #Make the prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction

    ##return data, codes