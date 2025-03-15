import streamlit as st

st.set_page_config(
    page_title="Team Organization - DataRock",
    page_icon="👥",
    layout="wide"
)

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
    {"name": "Camilla Perotti", "role": "Project Manager & ML Engineer", "image": "App/images/camilla.jpg",
     "description": "Led the project to ensure seamless collaboration. Managed ETL, ensuring data integrity and scalable processing with OOP."},
    
    {"name": "Lucía Sarobe", "role": "ML Engineer & Web Developer", "image": "App/images/lucia.jpg",
     "description": " Developed and fine-tuned the logistic regression model while ensuring smooth model integration with the app."},
    
    {"name": "Tomás Silva", "role": "Data Engineer               ", "image": "App/images/tomas.jpg",
     "description": "Designed the API layer to fetch real-time stock data as well as news data, ensuring accurate financial inputs for predictions."},
    
    {"name": "Héctor Marmol", "role": "Web Developer              ", "image": "App/images/hector.jpg",
     "description": "Developed the interactive Streamlit interface and managed application deployment."},
]

# Display team members in a 4-column layout
cols = st.columns(4)

for i, member in enumerate(team_members):
    with cols[i]:
        st.image(member["image"], use_container_width=True)  # Display profile picture
                
        # Display name (Bold & Centered)
        st.markdown(f"""  
        <div style="text-align: center; font-weight: bold; font-size: 16px;">
            {member["name"]}
        </div>  
        """, unsafe_allow_html=True)

        # Display role (Italic, Centered, extra spacing)
        st.markdown(f"""  
        <div style="text-align: center; font-style: italic; font-size: 14px;">
            {member["role"]}
            <br><br>  <!-- Adds extra spacing before the description -->
        </div>  
        """, unsafe_allow_html=True)

        # Display description (Centered, Sans-Serif)
        st.markdown(f"""  
        <div style="text-align: center; font-size: 13px;">
            {member["description"]}
        </div>  
        """, unsafe_allow_html=True)   

# Footer
st.markdown("---")
st.markdown("📢 *Thank you for visiting our team page!*")