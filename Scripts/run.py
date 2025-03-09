
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib
import os

class Company:

    def __init__(self, ticker) -> None:

        # attributes
        self.ticker = ticker
    
    # methods
    def load_data(self, filepath):
        # Load data
        prices_all = pd.read_csv(filepath,delimiter=';')

        # Select 5 companies' price data
        prices = prices_all[prices_all["SimFinId"].isin([1253240, 111052, 63877,56317, 59265])]

        # Return prices_bruker df
        return prices

    def process_data(self, prices):
         # Impute missing values
        prices["Dividend"] = prices["Dividend"].fillna(0)

        # Transform to correct data types
        prices["Shares Outstanding"] =  prices["Shares Outstanding"].astype(int)
        prices["Date"] = pd.to_datetime(prices.Date, format="%Y-%m-%d")

        # Return prices_bruker df
        return prices
    
    def save_data(self, prices, output_filepath):
        prices.to_csv(output_filepath)
        print(f"Processed data saved to {output_filepath}")

    def etl_pipeline(self, filepath, output_filepath):
        # Extract
        self.prices = self.load_data(filepath)
        print(f"Data loaded from {filepath}")
        
        # Transform
        self.prices = self.process_data(self.prices)
        print("Data processed successfully")
        
        # Load
        self.save_data(self.prices, output_filepath)

        return self.prices  # Return the processed DataFrame


    def prepare_data(self, prices):
        # Ensure data is sorted by date
        prices = prices.sort_values("Date").reset_index(drop=True)

        # Drop all unwanted columns, rename adjusted close column
        prices.drop(columns=["Ticker","SimFinId","Open","High", "Low", "Close", "Volume", "Dividend","Shares Outstanding"],inplace = True)
        prices.rename(columns={"Adj. Close": "Price"}, inplace= True)

        # Adding last 4 days' prices into the df
        days = 4
        for day in range(1, days + 1):
            prices[f"Price d-{day}"] = prices["Price"].shift(day)

        #dropping first rows with missing values
        prices = prices.dropna()

        # Adding prediction
        prices["Price_Up"] = (prices["Price"].shift(-1) > prices["Price"]).astype(int)
        prices = prices.drop(index=1238)

        # Splitting features from target
        x = prices.drop(columns=["Price_Up", "Date"])
        y = prices["Price_Up"]

        # Splitting data into training and testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        # Scaling the features (x):
        sc = StandardScaler()
        x_train_scaled = sc.fit_transform(x_train)
        x_test_scaled = sc.transform(x_test)

        # Returning all variables
        return prices, x_train_scaled, x_test_scaled, y_train, y_test, sc

    def train_model(self, x_train_scaled, y_train, model_path, sc):
        # Training model
        classifier = LogisticRegression(random_state = 1)
        classifier.fit(x_train_scaled, y_train) 

        # Save trained model and scaler
        joblib.dump(classifier, model_path)
        joblib.dump(sc, model_path.replace("model.pkl", "scaler.pkl"))
        print(f"Model trained and saved at {model_path}")

        # Return trained model
        return classifier, sc

    def load_model(self, model_path): 
        # load the model
        classifier = joblib.load(model_path)
        sc = joblib.load(model_path.replace("model.pkl", "scaler.pkl"))
        print("Model and scaler loaded successfully.")
        return classifier, sc

    def predict_next_day(self, prices, classifier, sc):
        # Ensure "Price_Up" exists before dropping
        if "Price_Up" in prices.columns:
            latest_features = prices.iloc[-1:].drop(columns=["Price_Up", "Date"])
        else:
            latest_features = prices.iloc[-1:].drop(columns=["Date"]) 

        latest_features_scaled = sc.transform(latest_features)  # Scale it

        prediction = classifier.predict(latest_features_scaled)
        print("Latest features used in ML.py:", latest_features)
        return "UP" if prediction[0] == 1 else "DOWN"

    def evaluate_model(self, classifier, x_test_scaled, y_test):
        # Evaluate trained model
        y_pred = classifier.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        return accuracy

    def ml_pipeline(self, model_path):
        # Prepare data
        prices, x_train_scaled, x_test_scaled, y_train, y_test, sc = self.prepare_data(self.prices)

        # Load model
        # Check if model already exists
        if os.path.exists(model_path):
            # Load the trained model
            classifier, sc = self.load_model(model_path)
            print("Using saved model for prediction.")
        else:
            # Train and save model if not found
            print("Model not found, training a new one.")
            classifier, sc = self.train_model(x_train_scaled, y_train, model_path, sc)

        # Predict next day's movement
        prediction = self.predict_next_day(prices, classifier, sc)
        print(f"Next day's price movement: {prediction}")

if __name__ == "__main__":
    # raw data file
    filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/data/us-shareprices-daily.csv"
    # clean data file
    output_filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/prices_output.csv"

     # model path as output
    model_path = "models/model.pkl"

    bruker = Company("1")
    bruker.etl_pipeline(filepath, output_filepath)
    bruker.ml_pipeline(model_path)
    

