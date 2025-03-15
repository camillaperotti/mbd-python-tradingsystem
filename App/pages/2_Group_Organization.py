import os
import streamlit as st

st.set_page_config(
    page_title="Team Organization - DataRock",
    page_icon="👥",
    layout="wide"
)

# Get the absolute path to the images folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
IMAGE_DIR = os.path.join(BASE_DIR, "App", "images")  # Ensure it finds the images folder

# Debugging - Print working directory
st.write("📂 Current Working Directory:", os.getcwd())
st.write("📂 Expected Image Path:", IMAGE_DIR)
st.write("📂 Files in 'images' folder:", os.listdir(IMAGE_DIR) if os.path.exists(IMAGE_DIR) else "🚨 Folder not found!")

# Team Organization Page
st.title("Team Organization 👨‍💼👩‍💼")

# Team Members Data
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
        img_path = os.path.join(IMAGE_DIR, member["image"])  # Get the correct absolute path
        
        if os.path.exists(img_path):  # Ensure the image exists before displaying
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
