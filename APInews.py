#Python API Wrapper for NewsAPI

#Import necessary libraries

import requests
import logging
import pandas as pd

from Exceptions import SimFinError, ResourceNotFoundError

#Define class NewsAPI - retrieve news from different companies

class NewsAPI:

    base_url = 'https://newsapi.org/v2/everything'

    #Initialize the API wrapper with the provided API key
    def __init__(self, api_key):
        self.api_key = api_key

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

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