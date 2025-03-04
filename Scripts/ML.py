import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib


def load_data(filepath):
    # Load clean file
    prices_bruker = pd.read_csv(filepath)
    
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
    return x_train_scaled, x_test_scaled, y_train, y_test, sc

def train_model(x_train_scaled, y_train, model_path, sc):
    # Training model
    classifier = LogisticRegression(random_state = 1)
    classifier.fit(x_train_scaled, y_train) 

    # Save trained model and scaler
    joblib.dump(classifier, model_path)
    joblib.dump(sc, model_path.replace("model.pkl", "scaler.pkl"))
    print(f"Model trained and saved at {model_path}")

    # Return trained model
    return classifier, sc # sc needed here ?

def load_model(model_path):
    # load the model
    model = joblib.load(model_path)
    sc = joblib.load(model_path.replace("model.pkl", "scaler.pkl"))
    print("Model and scaler loaded successfully.")
    return model, sc

def predict_next_day(x_test_scaled, model):
    # Uses trained model to predict if the next day's price will go UP or DOWN
    
    #prediction = model.predict(x_test_scaled[-1:])  
    #return "UP" if prediction[0] == 1 else "DOWN"
    
    # get last available row
    latest_features = x.iloc[-1:].values  # Last row
    latest_features_scaled = sc.transform(latest_features)  # Scale it

    # Predict
    next_day_prediction = classifier.predict(latest_features_scaled)
    return "UP" if next_day_prediction[0] == 1 else "DOWN"

def evaluate_model(model, x_test_scaled, y_test):
    # Evaluate trained model
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

def ml_pipeline(filepath, model_path):
    # Load data
    prices_bruker = load_data(filepath)

    # Prepare data
    x_train_scaled, x_test_scaled, y_train, y_test, sc = prepare_data(prices_bruker)

    # Train and save model
    model, sc = train_model(x_train_scaled, y_train, model_path, sc)

    # Load model for predictions
    model, sc = load_model(model_path)

    # Make prediction for next day
    prediction = predict_next_day(x_test_scaled, model)

    print(f"Next day's price movement: {prediction}")


if __name__ == "__main__":
    # clean data file as input
    filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/pricesbruker_output.csv"
    
    # model path as output
    model_path = "models/model.pkl"

    ml_pipeline(filepath, model_path)