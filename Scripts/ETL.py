import pandas as pd
import os
import argparse
import logging

# Configure logging to show all messages and include timestamp
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get the script's directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths (relative to script location)
default_raw_data_path = os.path.join(base_dir, "ETL", "data", "us-shareprices-daily.csv")
default_output_path = os.path.join(base_dir, "pricesbruker_output.csv")


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
    logging.info(f"Processed data saved to {output_filepath}")

def etl_pipeline(filepath, output_filepath):
    # Extract
    prices_bruker = load_data(filepath)
    logging.info(f"Data loaded from {filepath}")
    
    # Transform
    prices_bruker = process_data(prices_bruker)
    logging.info("Data processed successfully")
    
    # Load
    save_data(prices_bruker, output_filepath)

    return prices_bruker  # Return the processed DataFrame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL for Bruker stock data")
    parser.add_argument("--raw_data", type=str, default=default_raw_data_path, help="Path to raw data CSV file")
    parser.add_argument("--output_data", type=str, default=default_output_path, help="Path to save processed data")
    args = parser.parse_args()

    etl_pipeline(args.raw_data, args.output_data)