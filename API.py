#Python API Wrapper for SimFin

#Import necessary libraries

import requests
import logging
import pandas as pd
from datetime import datetime

from Exceptions import SimFinError, ResourceNotFoundError

#Define class PySimFin - main client for the SimFin wrapper

class PySimFin:

    base_url = 'https://backend.simfin.com/api/v3/'

    #Initialize the API wrapper with the provided API key
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
        'Authorization': f'{api_key}'
    }

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)


    #Method to fetch data from the SimFin API
    def get_data(self, endpoint: str, params: dict):

        try:
            url = f'{self.base_url}{endpoint}'
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()  # Raise HTTP errors, if they exist
            data = response.json()
            return pd.DataFrame(data)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found. URL = {url}?{params}")
            else:
                raise SimFinError(f"HTTP Error: {e}")
            
        except requests.exceptions.RequestException as e:
            raise SimFinError(f"Request Error: {e}")
        
        except ValueError as e:
            raise SimFinError(f"Invalid JSON response: {e}")
        

    #This method will return DataFrame with all prices for the provided ticker in the provided time range.
    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:

        try:
            # Validate the date format (YYYY-MM-DD)
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")

        except ValueError:
            self.logger.error("Invalid date format. Please use YYYY-MM-DD.")
            return pd.DataFrame()  # Return an empty DataFrame on error

        self.logger.info(f"Fetching share prices for {ticker} from {start} to {end}")
        params = {"ticker": ticker, "start": start, "end": end}
        
        df = self.get_data("companies/prices/verbose", params)

        # Expand the 'data' column (data is a list of dictionaries)
        df_expanded = df.explode("data") 

        # Convert dictionary row into separate columns
        df_expanded = pd.concat([df_expanded.drop(columns=["data"]), df_expanded["data"].apply(pd.Series)], axis=1)

        return df_expanded
        
    
    #This method will return DataFrame with financial statements for the ticker provided in the provided time range.
    def get_financial_statement(self, ticker: str, statement: str, start: str, end: str) -> pd.DataFrame:

        try:
            # Validate the date format (YYYY-MM-DD)
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")

        except ValueError:
            self.logger.error("Invalid date format. Please use YYYY-MM-DD.")
            return pd.DataFrame()  # Return an empty DataFrame on error
        
        # Validate statement type (one of the inputs for the API request)
        valid_statements = ['PL', 'BS', 'CF', 'DERIVED']
        #PL = Profit & Loss; BS = Balance Sheet; CF = Cash Flow; DERIVED = Derived Ratios and Indicators

        if statement not in valid_statements:
            self.logger.error(f"Invalid statement type '{statement}'. Must be one of {valid_statements}.")
            return pd.DataFrame()

        self.logger.info(f"Fetching financial statements for {ticker} from {start} to {end}")
        params = {"ticker": ticker, 'statements': statement, "start": start, "end": end}
        
        df = self.get_data("companies/statements/verbose", params)

        return df
    

    #This method will return DataFrame with information data about the company
    def get_general_data(self, ticker: str) -> pd.DataFrame:

        self.logger.info(f"Fetching general data for {ticker}")
        params = {"ticker": ticker}
        
        df = self.get_data("companies/general/verbose", params)

        return df