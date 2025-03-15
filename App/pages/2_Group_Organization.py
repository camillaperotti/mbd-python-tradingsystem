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

# 🔹 Get absolute path to the 'images' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "images"))  # Move up one level to 'images/'

# 🔹 Debugging - Check if the images folder exists
if not os.path.exists(IMAGE_DIR):
    st.error(f"🚨 Folder not found: {IMAGE_DIR}")
else:
    st.write("✅ Found 'images/' folder:", IMAGE_DIR)
    st.write("📂 Files in 'images' folder:", os.listdir(IMAGE_DIR))

# Team Section
st.subheader("Meet Our Team")

# Team Members Data (with correct image paths)
team_members = [
    {"name": "Camilla Perotti", "role": "Project Manager & ML Engineer", "image": "camilla.jpg",
     "description": "Oversaw the project, ensuring collaboration and value creation. Led the ETL process and designed scalable data pipelines."},

    {"name": "Lucía Sarobe", "role": "ML Engineer & Model Integration", "image": "lucia.jpg",
     "description": "Developed and optimized the logistic regression model, ensuring smooth integration into the trading app."},

    {"name": "Tomás Silva", "role": "Data Engineer & API Development", "image": "tomas.jpg",
     "description": "Designed and implemented the API layer to fetch real-time stock and financial news data for precise model predictions."},

    {"name": "Héctor Marmol", "role": "Web Developer & Deployment Lead", "image": "hector.jpg",
     "description": "Developed the interactive Streamlit interface and managed app deployment for an intuitive user experience."},
]

# 🔹 Debugging file paths
st.write("📂 Current Working Directory:", os.getcwd())
st.write("📂 Files in Root:", os.listdir("."))
st.write("📂 Files in 'images' folder:", os.listdir(IMAGE_DIR) if os.path.exists(IMAGE_DIR) else "🚨 Folder not found!")

# 🔹 Check image file existence before displaying
for member in team_members:
    image_path = os.path.join(IMAGE_DIR, os.path.basename(member["image"]))  # 🔹 Use absolute path

    if not os.path.exists(image_path):
        st.warning(f"⚠️ Image not found: {image_path}")

# 🔹 Display team members in a 4-column layout
cols = st.columns(4)

for i, member in enumerate(team_members):
    img_path = os.path.join(IMAGE_DIR, os.path.basename(member["image"]))  # 🔹 Use absolute path

    with cols[i]:
        if os.path.exists(img_path):
            st.image(img_path, caption=member["name"])
        else:
            st.error(f"❌ Missing image: {img_path}")

# Footer
st.markdown("---")
st.markdown("📢 *Thank you for visiting our team page!*")
