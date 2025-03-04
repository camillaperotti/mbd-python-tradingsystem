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
    
def save_data(prices_)


if __name__ == "__main__":
    # raw data file
    filepath = "/Users/camillaperotti/Desktop/IE/Courses MBD/Term 2/PDA II/00_GroupProject/mbd-python-tradingsystem/ETL/pricesbruker_output.csv"
    # clean data file
    output_file = 