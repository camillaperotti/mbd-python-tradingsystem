import streamlit as st
import os

st.set_page_config(
    page_title="Team Organization - DataRock",
    page_icon="👥",
    layout="wide"
)

# 🔹 Correct the path to the 'images' folder (move up one level)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up from 'App/pages/' to 'App/'
IMAGE_DIR = os.path.join(BASE_DIR, "images")  # Now correctly points to 'App/images/'

# Debugging - Check if the images folder exists
if not os.path.exists(IMAGE_DIR):
    st.error(f"🚨 Folder not found: {IMAGE_DIR}")
else:
    st.write("✅ Found 'images/' folder:", IMAGE_DIR)
    st.write("📂 Files in 'images' folder:", os.listdir(IMAGE_DIR))

# Team Organization Page
st.title("Team Organization 👨‍💼👩‍💼")

# Team Members Data (Now correctly referencing 'App/images/')
team_members = [
    {"name": "Camilla Perotti", "role": "Project Manager & ML Engineer", "image": "camilla.jpg",
     "description": "Led the project to ensure seamless collaboration. Managed ETL, ensuring data integrity and scalable processing with OOP."},

    {"name": "Lucía Sarobe", "role": "ML Engineer & Web Developer", "image": "lucia.jpg",
     "description": "Developed and fine-tuned the logistic regression model while ensuring smooth model integration with the app."},

    {"name": "Tomás Silva", "role": "Data Engineer", "image": "tomas.jpg",
     "description": "Designed the API layer to fetch real-time stock data as well as news data, ensuring accurate financial inputs for predictions."},

    {"name": "Héctor Marmol", "role": "Web Developer", "image": "hector.jpg",
     "description": "Developed the interactive Streamlit interface and managed application deployment."},
]

# Display team members in a 4-column layout
cols = st.columns(4)

for i, member in enumerate(team_members):
    with cols[i]:
        img_path = os.path.join(IMAGE_DIR, member["image"])  # 🔹 Now correctly points to 'App/images/'

        if os.path.exists(img_path):  # 🔹 Ensure image exists before displaying
            st.image(img_path)
        else:
            st.error(f"❌ Image not found: {img_path}")

        # Display name
        st.markdown(f"""  
        <div style="text-align: center; font-weight: bold; font-size: 16px;">
            {member["name"]}
        </div>  
        """, unsafe_allow_html=True)

        # Display role
        st.markdown(f"""  
        <div style="text-align: center; font-style: italic; font-size: 14px;">
            {member["role"]}
            <br><br>
        </div>  
        """, unsafe_allow_html=True)

        # Display description
        st.markdown(f"""  
        <div style="text-align: center; font-size: 13px;">
            {member["description"]}
        </div>  
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("📢 *Thank you for visiting our team page!*")
