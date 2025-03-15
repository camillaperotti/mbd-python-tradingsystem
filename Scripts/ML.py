import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib
import os
import logging

# Configure logging to show all messages and include timestamp
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get the script's directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define dynamic paths
filepath = os.path.join(base_dir, "ETL", "pricesbruker_output.csv")
model_dir = os.path.join(base_dir, "Scripts", "models")
model_path = os.path.join(model_dir, "model_BRKR.pkl")

def load_data(filepath):
    # Load clean file
    prices_bruker = pd.read_csv(filepath,index_col=0)
    
    # Return dataframe
    return prices_bruker

def prepare_data(prices_bruker):
    # Ensure data is sorted by date
    prices_bruker = prices_bruker.sort_values("Date").reset_index(drop=True)

    # Drop all unwanted columns, rename adjusted close column
    prices_bruker.drop(columns=["Ticker","SimFinId","Open","High", "Low", "Close", "Volume", "Dividend","Shares Outstanding"],inplace = True)
    prices_bruker.rename(columns={"Adj. Close": "Price"}, inplace= True)

    # Adding last 4 days' prices into the df
    days = 4
    for day in range(1, days + 1):
        prices_bruker[f"Price d-{day}"] = prices_bruker["Price"].shift(day)

    #dropping first rows with missing values
    prices_bruker = prices_bruker.dropna()

    # Adding prediction
    prices_bruker["Price_Up"] = (prices_bruker["Price"].shift(-1) > prices_bruker["Price"]).astype(int)
    prices_bruker = prices_bruker.drop(index=1238)

    # Splitting features from target
    x = prices_bruker.drop(columns=["Price_Up", "Date"])
    y = prices_bruker["Price_Up"]

    # Splitting data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Scaling the features (x):
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    # Returning all variables
    return prices_bruker, x_train_scaled, x_test_scaled, y_train, y_test, sc

def train_model(x_train_scaled, y_train, model_path, sc):
    # Training model
    classifier = LogisticRegression(random_state = 1)
    classifier.fit(x_train_scaled, y_train) 

    # Save trained model and scaler
    joblib.dump(classifier, model_path)
    joblib.dump(sc, model_path.replace("models/model_BRKR.pkl", "models/scaler_BRKR.pkl"))
    logging.info(f"Model trained and saved at {model_path}")

    # Return trained model
    return classifier, sc

def load_model(model_path): 
    # load the model
    classifier = joblib.load(model_path)
    sc = joblib.load(model_path.replace("models/model_BRKR.pkl", "models/scaler_BRKR.pkl"))
    logging.info("Model and scaler loaded successfully.")
    return classifier, sc

def predict_next_day(prices_bruker, classifier, sc):
    # Ensure "Price_Up" exists before dropping
    if "Price_Up" in prices_bruker.columns:
        latest_features = prices_bruker.iloc[-1:].drop(columns=["Price_Up", "Date"])
    else:
        latest_features = prices_bruker.iloc[-1:].drop(columns=["Date"]) 

    latest_features_scaled = sc.transform(latest_features)  # Scale it

    prediction = classifier.predict(latest_features_scaled)
    logging.info(f"Latest features used in ML.py:{latest_features}")
    return "UP" if prediction[0] == 1 else "DOWN"

def evaluate_model(classifier, x_test_scaled, y_test):
    # Evaluate trained model
    y_pred = classifier.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

def ml_pipeline(filepath, model_path):
    # Load data
    prices_bruker = load_data(filepath)

    # Prepare data
    prices_bruker, x_train_scaled, x_test_scaled, y_train, y_test, sc = prepare_data(prices_bruker)

    # Load model
    # Check if model already exists
    if os.path.exists(model_path):
        # Load the trained model
        classifier, sc = load_model(model_path)
        logging.debug("Using saved model for prediction.")
    else:
        # Train and save model if not found
        logging.debug("Model not found, training a new one.")
        classifier, sc = train_model(x_train_scaled, y_train, model_path, sc)

    # Predict next day's movement
    prediction = predict_next_day(prices_bruker, classifier, sc)
    logging.info(f"Next day's price movement: {prediction}")

if __name__ == "__main__":
    ml_pipeline(filepath, model_path)