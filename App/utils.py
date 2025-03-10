import pandas as pd
import streamlit as st
import pickle

@st.cache_data #it will be loaded once and then saved
def read_and_preprocess_data() -> pd.DataFrame:
    """
    Loads and preprocesses stock market data from the consolidated CSV file.
    Returns a Pandas DataFrame.
    """
    csv_path = "ETL/prices_output.csv"
    return pd.read_csv(csv_path)

def load_model_and_scaler(ticker) -> tuple:
    """
    Loads the trained model and scaler for a selected stock ticker.
    Returns a tuple (model, scaler).
    """
    model_path = f"Scripts/models/model_{ticker}.pkl"
    scaler_path = f"Scripts/models/scaler_{ticker}.pkl"

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    
    return model, scaler

def preprocess_stock_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Filters stock data for the selected ticker.
    Returns a DataFrame with only relevant data.
    """
    return data[data["Ticker"] == ticker].copy()

def make_prediction(df_ticker: pd.DataFrame, model, scaler) -> float:
    """
    Uses the trained model and scaler to make a stock price prediction.
    Returns the predicted value.
    """
    columns_to_use = [col for col in df_ticker.columns if col not in ["Date", "Close", "Ticker"]]  # ✅ Fixed
    X_new = df_ticker.iloc[-1:][columns_to_use]  # Get the latest row
    X_scaled = scaler.transform(X_new)
    prediction = model.predict(X_scaled)
    return prediction[0]

    ##return data, codes