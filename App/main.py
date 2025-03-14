import numpy as np
import streamlit as st
import plotly.express as px

from utils import read_and_preprocess_data

def main():

    st.set_page_config(
        page_title="DataRock - Data-driven Financial Advisors",
        page_icon=":bar_chart:",
        layout="wide"
    )

    # Use columns to position the logo and text side-by-side
    col1, col2 = st.columns([3, 1])  # Adjust column widths as needed

    with col1:
        st.write("""
        Python Group Project: Trading system for selected US companies.
        
        ### Overview:
        This website contains several parts
        - **Homepage**: Contains information about the trading system app, instructions on how to use it.
        - **Trading Prediction and Company Deep Dive**: Trading prediction for selected company plus detailed information on the company.
        - **Group Organization**: Information about the DataRock team and their roles.
        """)

        st.markdown("---")

        st.write("""
        ### DataRock Trading System: 
        This web application provides insights into the daily stock price fluctuations of five major companies: 
        **Apple (AAPL), Abbott (ABT), Bruker (BRKR), Microsoft (MSFT), and Tesla (TSLA).** It is designed to help users make informed trading decisions.

        To provide a broader and more comprehensive view of the U.S. economy, the selection includes companies from different industries rather than focusing on a single sector: 
        - **Apple & Microsoft** (Tech industry) 
        - **Tesla** (Automotive & Energy sector) 
        - **Abbott** (Healthcare industry) 
        - **Bruker** (Biotech sector) 

        By incorporating companies from diverse industries, this application offers a well-rounded perspective on market movements, helping users navigate stock trading with greater confidence.
        
        **How to use DataRock:**
        - Use the sidebar to select the page to be viewed.
        - Within the trading subpage, select the company of interest to view predictions and information about the company.
        - Explore the 'Group Organization' page to learn about the DataRock team and their roles.
        """)
        
        st.markdown("---")
        
    with col2:
        # Add vertical spacing before the logo
        st.markdown("<br><br><br><br>", unsafe_allow_html=True) # Adjust number of <br> for desired spacing

        # Load and display the DataRock logo with increased size
        st.image("App/images/logo.png", width=250)
        
if __name__ == "__main__":
    main()