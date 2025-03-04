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
    return x_train_scaled, y_train

def train_model(x_train_scaled, y_train):
    # Training model
    classifier = LogisticRegression(random_state = 1)
    classifier.fit(x_train_scaled, y_train) 

    # Save trained model
    

def save_model()





if __name__ == "__main__":
    # clean data file as input
    filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/pricesbruker_output.csv"
    # output ?? model pkl??
    output_filepath = 
