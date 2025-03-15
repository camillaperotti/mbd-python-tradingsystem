#Python API Wrapper for SimFin

#Import necessary libraries

import requests
import logging
import pandas as pd
from datetime import datetime
from Scripts.Exceptions import SimFinError, ResourceNotFoundError

#Define class PySimFin - main client for the SimFin wrapper

class PySimFin:

    base_url = 'https://backend.simfin.com/api/v3/'

    #Initialize the API wrapper with the provided API key
    def __init__(self, api_key): ## mac: _init_, windows: __init__
        self.api_key = api_key
        self.headers = {
        'Authorization': f'{api_key}'
    }

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__) ## mac: _init_, windows: __init__


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

        self.logger.info(f"Fetching share prices for {ticker} from {start} to {end}.")
        params = {"ticker": ticker, "start": start, "end": end}
        
        df = self.get_data("companies/prices/verbose", params)

        # Check if DataFrame is empty
        if df.empty:
            self.logger.warning(f"No data available for {ticker} from {start} to {end}.")
            return pd.DataFrame() # Return an empty DataFrame on error

        # Expand the 'data' column (data is a list of dictionaries)
        df = df.explode("data") 

        # Convert dictionary row into separate columns
        df = pd.concat([df.drop(columns=["data"]), df["data"].apply(pd.Series)], axis=1)

        return df
        
    
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
            return pd.DataFrame() # Return an empty DataFrame on error

        self.logger.info(f"Fetching financial statements for {ticker} from {start} to {end}")
        params = {"ticker": ticker, 'statements': statement, "start": start, "end": end}
        
        df = self.get_data("companies/statements/verbose", params)

        # Check if DataFrame is empty
        if df.empty:
            self.logger.warning(f"No {statement} available for {ticker} from {start} to {end}.")
            return pd.DataFrame()
        
        # Ensure 'statements' is exploded properly
        df = df.explode("statements", ignore_index=True)

        # Extract 'statement' type and expand the 'data' column
        df["statement_type"] = df["statements"].apply(lambda x: x["statement"])
        df["data"] = df["statements"].apply(lambda x: x["data"])

        # Drop the original 'statements' column
        df = df.drop(columns=["statements"])

        # Explode the 'data' column to ensure each entry is in a separate row
        df = df.explode("data", ignore_index=True)

        # Convert each dictionary row into separate columns
        data_expanded = df["data"].apply(pd.Series)

        # Drop the original 'data' column and concatenate expanded data
        df = pd.concat([df.drop(columns=["data"]), data_expanded], axis=1)

        return df
    

    #This method will return DataFrame with information data about the company
    def get_general_data(self, ticker: str) -> pd.DataFrame:

        self.logger.info(f"Fetching general data for {ticker}")
        params = {"ticker": ticker}
        
        df = self.get_data("companies/general/verbose", params)

        # Check if DataFrame is empty
        if df.empty:
            self.logger.warning(f"No data available for {ticker}.")
            return pd.DataFrame()

        return df
    

#Define class NewsAPI - retrieve news from different companies

#These are the codes of the companies we are predicting to retrieve the news. In this API it is not the ticker, but the name
#AAPL = Apple
#ABT = Abbott
#BRKR = Bruker
#MSFT = Microsoft
#TSLA = Tesla

class NewsAPI:

    base_url = 'https://newsapi.org/v2/everything'

    #Initialize the API wrapper with the provided API key
    def _init_(self, api_key):
        self.api_key = api_key

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(_name_)

    #Method to fetch data from the SimFin API
    def get_data(self, params: dict):

        try:
            url = f'{self.base_url}'
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise HTTP errors, if they exist
            data = response.json()
            return data
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found. URL = {url}?{params}")
            else:
                raise SimFinError(f"HTTP Error: {e}")
            
        except requests.exceptions.RequestException as e:
            raise SimFinError(f"Request Error: {e}")
        
        except ValueError as e:
            raise SimFinError(f"Invalid JSON response: {e}")

    
    #This method will return DataFrame with all the news associated to a company from a specific date
    def get_news(self, company: str, from_date: str, sort_by: str = 'popularity') -> pd.DataFrame:

        self.logger.info(f"Fetching news for {company} from {from_date}.")
        params = {"q": company, "from": from_date, "sortBy": sort_by, "apiKey": self.api_key}
        
        df = self.get_data(params)

        # Check if DataFrame is empty and if 'articles' exist in the json response (data)
        if not df or 'articles' not in df or not df['articles']:
            self.logger.warning(f"No news available for {company} from {from_date}.")
            return pd.DataFrame()

        articles = df['articles']
        df = pd.DataFrame(articles)

        if 'source' in df.columns:
            source_df = pd.json_normalize(df['source'])
            df = pd.concat([df.drop('source', axis=1), source_df], axis=1)
        else:
            self.logger.warning("No 'source' key found in articles.")
        
        return df.head(2)