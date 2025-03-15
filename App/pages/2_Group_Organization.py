import streamlit as st
import os

# Team Organization Page
st.title("Team Organization 👨‍💼👩‍💼")

# Project Introduction
st.markdown("""
Welcome to our **team organization page**!  
We are a team of data-driven financial advisors leveraging **Machine Learning** to build an **automated trading prediction system**.  
Each member contributed their expertise in different areas, ensuring a seamless and effective system.
""")

# Team Section
st.subheader("Meet Our Team")

# Team Members Data (with correct image paths)
team_members = [
    {"name": "Camilla Perotti", "role": "Project Manager & ML Engineer", "image": "images/camilla.jpg",
     "description": "Oversaw the project, ensuring collaboration and value creation. Led the ETL process and designed scalable data pipelines."},
    
    {"name": "Lucía Sarobe", "role": "ML Engineer & Model Integration", "image": "images/lucia.jpg",
     "description": "Developed and optimized the logistic regression model, ensuring smooth integration into the trading app."},
    
    {"name": "Tomás Silva", "role": "Data Engineer & API Development", "image": "images/tomas.jpg",
     "description": "Designed and implemented the API layer to fetch real-time stock and financial news data for precise model predictions."},
    
    {"name": "Héctor Marmol", "role": "Web Developer & Deployment Lead", "image": "images/hector.jpg",
     "description": "Developed the interactive Streamlit interface and managed app deployment for an intuitive user experience."},
]

# Display team members in a 4-column layout
cols = st.columns(4)

#for i, member in enumerate(team_members):
   # with cols[i]:
        #st.image(member["image"], caption=member["name"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

for i, member in enumerate(team_members):
    image_path = os.path.join(IMAGE_DIR, os.path.basename(member['image']))
    
    if not os.path.exists(image_path):
        st.warning(f"⚠️ Image not found: {image_path}")  # Debugging message
    
    with cols[i]:
        st.image(image_path, caption=member["name"])
        
# Footer
st.markdown("---")
st.markdown("📢 *Thank you for visiting our team page!*")

