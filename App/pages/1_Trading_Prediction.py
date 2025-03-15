#Web-page configuration
import streamlit as st

# Set the page title and favicon (must be at the very top)
st.set_page_config(
    page_title="Trading Prediction - DataRock",
    page_icon="📈",
    layout="wide"
)

##Ensuring Python Uses the Project’s utils.py Instead of site-packages. As well as Scripts
import sys
import os

#Now import utils from App/
from utils import read_and_preprocess_data, preprocess_stock_data
##

import plotly.express as px
import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime, timedelta

# Go to 'MBD-Python-Trading System'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))) ## mac: _init_, windows: __init__

##


# Import custom scripts
from Scripts.API import PySimFin
#from Scripts.API import NewsAPI
from Scripts.Exceptions import SimFinError, ResourceNotFoundError
from Scripts.predict import Prediction
from utils import read_and_preprocess_data, preprocess_stock_data

# Define API Keys
SIMFIN_API_KEY = "33cd76b1-b978-4165-8b91-5696ddea452a"

# Initialize APIs
simfin_client = PySimFin(SIMFIN_API_KEY)

#Load stock data (Historical data for graph)
data = read_and_preprocess_data()

#Define 5 available stocks
allowed_tickers = ["AAPL", "ABT", "BRKR", "MSFT", "TSLA"]

##SIDEBAR: Stock Selection (Only Your 5 Tickers)
st.sidebar.header("📌 Stock Market Selection")
tickers_in_data = sorted(set(data["Ticker"].unique()) & set(allowed_tickers))  # Ensure only available tickers are shown
ticker = st.sidebar.selectbox("Select a stock:", tickers_in_data)

# Define mapping of tickers to company names
company_names = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corp",
    "TSLA": "Tesla Inc",
    "ABT": "Abbott Laboratories",
    "BRKR": "Bruker Corp"
}

# Get the full company name (default to ticker if not found)
company_name = company_names.get(ticker, ticker)

##COMPANY STOCKS##
# Display the company name as the main title
st.title(f"{company_name} Stock Analysis")

# Show Latest Stock Data
st.subheader(f"📊 Latest Historical Data for {ticker}")
data_ticker = preprocess_stock_data(data, ticker)

#GRAPH, Historical data
# Stock Price Evolution Graph
# Stock Price Evolution Graph with optimized dark theme
fig = px.line(data_ticker, x="Date", y="Close", 
              title=f"{ticker} Price Evolution - Historical Data", template="plotly_dark")
# Customize background to match dark UI
fig.update_layout(
    plot_bgcolor="rgba(30,30,30,0.9)",  # Dark gray background for the graph
    paper_bgcolor="rgba(30,30,30,0.9)",  # Dark gray background for the entire figure
    font=dict(color="white")  # Ensure text is visible on dark mode
)
# Update axes for clarity
fig.update_xaxes(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.2)")  # Subtle grid lines
fig.update_yaxes(title="Closing Price", showgrid=True, gridcolor="rgba(255,255,255,0.2)")
# Display the final graph
st.plotly_chart(fig)

###FINANCIAL STATEMENTS
st.subheader("📑 Financial Statements")

statement_type = st.selectbox("Select Financial Statement:", ["Profit & Loss (PL)", "Balance Sheet (BS)", "Cash Flow (CF)", "Derived Ratios (DERIVED)"])
statement_code = {"Profit & Loss (PL)": "PL", "Balance Sheet (BS)": "BS", "Cash Flow (CF)": "CF", "Derived Ratios (DERIVED)": "DERIVED"}[statement_type]

##**Date Range Selection**
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=730))  # Default: 2 years ago

with col2:
    end_date = st.date_input("End Date", datetime.today())  # Default: Today

# Ensure dates are formatted correctly
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")

# Fetch financial data only when the button is clicked
if "financial_data" not in st.session_state:
    st.session_state.financial_data = None  # Initialize session state

if st.button("Fetch Financial Data"):
    st.write(f"Fetching {statement_type} for {ticker} from {start_date} to {end_date}...")

    # Retrieve financial data and store it in session state
    financial_data = simfin_client.get_financial_statement(ticker, statement_code, start_date, end_date)

    if not financial_data.empty:
        st.session_state.financial_data = financial_data  # Store data persistently
    else:
        st.warning(f"⚠ No data available for {statement_type} in the selected date range.")

# **Ensure financial data persists after fetching**
if st.session_state.financial_data is not None:
    st.subheader("📊 Full Financial Data View")
    st.write("Displaying all columns for reference.")
    st.dataframe(st.session_state.financial_data)  # Show full dataset initially

    # **Step 2: Allow User to Select Columns**
    st.subheader("📌 Select Relevant Columns to Display")
    all_columns = st.session_state.financial_data.columns.tolist()  # Get all available columns

    # Ensure user can select columns (default selects first 6 columns)
    selected_columns = st.multiselect("Choose which columns to display:", all_columns, default=all_columns[:6])

    # **Step 3: Display Filtered Dataframe**
    if selected_columns:  # Ensure at least one column is selected
        financial_data_filtered = st.session_state.financial_data[selected_columns]  # Filter data

        st.subheader("📈 Filtered Financial Data")
        st.write("Showing selected columns:")
        st.dataframe(financial_data_filtered)  # Display updated dataframe
    else:
        st.warning("⚠ Please select at least one column to display.")

##PREDICTION
st.subheader("💡Stock Price Movement Predictor")

if st.button("Predict Stock Movement"):
    company = Prediction(ticker)
    company.load_api()
    company.transform_data()

    if company.load_model():
        prediction = company.predict_next_day()
        
        if prediction:  # Ensure a valid result
            # Define strategy based on prediction
            if prediction == "UP":
                emoji = "📈"
                action = "**BUY** ✅"
                message = f"**{emoji} {ticker} is predicted to go UP tomorrow!**"
                strategy = f"Recommended strategy: {action} (Expecting price increase)"
            else:  # If "DOWN"
                emoji = "📉"
                action = "**SELL** ❌"
                message = f"**{emoji} {ticker} is predicted to go DOWN tomorrow.**"
                strategy = f"Recommended strategy: {action} (Minimize potential losses)"

            # Display simplified prediction result
            st.markdown(message)
            st.markdown(strategy)


##NEWS
#NEWS_API_KEY = "9e47d5c4e7374f29a69f83554ed9c6b9"  # actual API key
#news_client = NewsAPI(NEWS_API_KEY)

# Map tickers to full company names for the NewsAPI
#ticker_to_company = {
 #   "AAPL": "Apple",
  #  "MSFT": "Microsoft",
   # "TSLA": "Tesla",
   # "ABT": "Abbott",
    #"BRKR": "Bruker"
#}

# 📰 STOCK MARKET NEWS SECTION
#st.subheader(f"📰 Latest News on {company_name}")

# Get the company name for the API request
#company_for_news = ticker_to_company.get(ticker, ticker)  # Default to ticker if not found

# Define date range (fetch last 7 days of news)
#news_from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

# Fetch news using NewsAPI
#news_df = news_client.get_news(company_for_news, news_from_date)

# Display the news articles
#if not news_df.empty:
#    for i, row in news_df.iterrows():
#        st.markdown(f"**[{row['title']}]({row['url']})** *(via {row['name']}, {row['publishedAt'][:10]})*")
#else:
#    st.warning(f"⚠ No recent news available for {company_name}.")