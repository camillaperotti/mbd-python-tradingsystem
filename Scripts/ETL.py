import pandas as pd

def load_data(filepath):
    # Load data
    prices = pd.read_csv(filepath,delimiter=';')

    # Select BRUKER company data
    prices_bruker = prices[prices["SimFinId"] == 1253240]

    # Return prices_bruker df
    return prices_bruker

def process_data(prices_bruker):
    # Impute missing values
    prices_bruker["Dividend"] = prices_bruker["Dividend"].fillna(0)

    # Transform to correct data types
    prices_bruker["Shares Outstanding"] =  prices_bruker["Shares Outstanding"].astype(int)
    prices_bruker["Date"] = pd.to_datetime(prices_bruker.Date, format="%Y-%m-%d")

    # Return prices_bruker df
    return prices_bruker
    
def save_data(prices_bruker, output_filepath):
    prices_bruker.to_csv(output_filepath)
    print(f"Processed data saved to {output_filepath}")

def etl_pipeline(filepath, output_filepath):
    # Extract
    prices_bruker = load_data(filepath)
    print(f"Data loaded from {filepath}")
    
    # Transform
    prices_bruker = process_data(prices_bruker)
    print("Data processed successfully")
    
    # Load
    save_data(prices_bruker, output_filepath)

    return prices_bruker  # Return the processed DataFrame


if __name__ == "__main__":
    # raw data file
    filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/data/us-shareprices-daily.csv"
    # clean data file
    output_filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/test.csv"

etl_pipeline(filepath, output_filepath)