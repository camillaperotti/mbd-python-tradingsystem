import numpy as np
import streamlit as st
import plotly.express as px

from utils import read_and_preprocess_data

def main():

    st.set_page_config(
        page_title="Trading system for US companies",
        page_icon=":bar_chart:",
        layout="wide"
    )

    st.title("Trading system for US companies")
    st.write("""
    Python Group Project: Trading system for selected US companies.
    
    ### Overview:
    This website contains several parts
    - **Homepage**: Contains information about the trading system app, instructions on how to use it.
    - **Trading Prediction and Company Deep Dive**: Trading prediction for selected company plus detailed information on the company.
    - **Group Work Organization**: Information about the project team and work distribution. hello

             
    ### Trading system app: 
    This app provides information about XYZ... ADD INFORMATION. It is intended to give guidance to users wanting to trade stocks.
    How to use the web application:
    - Use the sidebar to select the page to be viewed.
    - Within the trading subpage, select the company of interest to view predictions and information about the company.
    """)
    
    st.markdown("---")
    
    #sources = sorted(data.src_neigh_name.unique())
    #destinations = sorted(data.dst_neigh_name.unique())
    
    #source = st.sidebar.selectbox('Select the source', sources)
    #destination = st.sidebar.selectbox('Select the destination', destinations)
    
    #aux = data[(data.src_neigh_name == source) & (data.dst_neigh_name == destination)]
    #aux = aux.sort_values("date")
    
if __name__ == "__main__":
    main()
