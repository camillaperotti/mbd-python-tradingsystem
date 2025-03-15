import streamlit as st
import os

def main():
    st.set_page_config(
        page_title="DataRock - Data-driven Financial Advisors",
        page_icon=":bar_chart:",
        layout="wide"
    )

    # 🔹 Correct the path to the 'images' folder (move up one level)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    IMAGE_DIR = os.path.join(BASE_DIR, "images")  # Ensure it finds 'App/images/' correctly

    # Debugging - Check if the images folder exists
    if not os.path.exists(IMAGE_DIR):
        st.error(f"🚨 Folder not found: {IMAGE_DIR}")
    else:
        st.write("✅ Found 'images/' folder:", IMAGE_DIR)
        st.write("📂 Files in 'images' folder:", os.listdir(IMAGE_DIR))

    # Logo Section
    col_logo, col_text = st.columns([1, 3])
    with col_logo:
        logo_path = os.path.join(IMAGE_DIR, "logo_w.png")  # 🔹 Use absolute path

        if os.path.exists(logo_path):  # 🔹 Ensure the logo exists before displaying
            st.image(logo_path, width=300)
        else:
            st.error(f"❌ Logo not found: {logo_path}")

    with col_text:
        st.markdown("<br>", unsafe_allow_html=True)  

    # Main Content
    st.markdown("""
    ### Data-driven Financial Advisors 💡
    Welcome to **DataRock**, a platform for **data-driven financial advisory**.  
    We provide **advanced market insights and predictive analytics** for **brokers and individual investors**,  
    helping you make informed trading decisions using **machine learning and financial modeling**.
    """)

    st.markdown("---")  # Separator for clarity

    st.write("""
    ### 📊 DataRock Trading System: 
    This web application provides insights into the daily stock price fluctuations of five major companies: 
    **Apple (AAPL), Abbott (ABT), Bruker (BRKR), Microsoft (MSFT), and Tesla (TSLA).** It is designed to help you make informed trading decisions.

    To provide a broader and more comprehensive view of the U.S. economy, the selection includes companies from different industries rather than focusing on a single sector: 
    - **Apple & Microsoft** (💻 Tech industry) 
    - **Tesla** (🚗 Automotive & Energy sector) 
    - **Abbott** (🏥 Healthcare industry) 
    - **Bruker** (🔬 Biotech sector) 

    By incorporating companies from diverse industries, this application offers a well-rounded perspective on market movements, helping users navigate stock trading with greater confidence.
    """)

    st.write("""
    **Group 2 Python Project:** Trading system for selected US companies.
    
    ### Overview:
    This website contains several parts:
    - **Homepage**:🏠 Contains information about the trading system app, instructions on how to use it.
    - **Trading Prediction and Company Deep Dive**: 📈 Trading prediction for selected company plus detailed information on the company.
    - **Group Organization**: Information about the DataRock team and their roles.
    """)

    st.markdown("---")

    st.write("""
    **How to use DataRock:**
    - 📌 Use the sidebar to select the page to be viewed.
    - 📊 Within the trading subpage, select the company of interest to view predictions and information about the company.
    - 📢 Explore the 'Group Organization' page to learn about the DataRock team and their roles.
    """)

    st.markdown("---")

if __name__ == "__main__":
    main()
