import streamlit as st

# 🎯 Team Organization Page
st.title("📌 Team Organization")

# 🏆 Project Introduction
st.markdown("""
Welcome to our **team organization page**!  
We are a group of four members working on an **automated trading prediction system** using **Machine Learning** and **Streamlit**.

Our approach followed these key steps:
1. 📊 **Data exploration and cleaning**  
2. 📈 **Time series analysis**  
3. 🤖 **Model training and optimization**  
4. 🛠 **Streamlit app development**  
5. 🚀 **Final implementation and testing**
""")

# 📌 Process Overview (Replace with an actual diagram)
#st.subheader("📌 Project Workflow")
#st.image("images/path_to_diagram.png", caption="Workflow Diagram", use_column_width=True)  

# 👥 Team Section
st.subheader("👥 Meet Our Team")

# 📌 Team Members Data (with correct image paths)
team_members = [
    {"name": "Camilla Perotti", "image": "App/images/camilla.jpg"},
    {"name": "Héctor Marmol", "image": "App/images/hector.jpg"},
    {"name": "Tomás Silva", "image": "App/images/tomas.jpg"},
    {"name": "Lucía Sarobe", "image": "App/images/lucia.jpg"},
]

# Display team members in a 4-column layout
cols = st.columns(4)

for i, member in enumerate(team_members):
    with cols[i]:
        st.image(member["image"], caption=member["name"], use_container_width=True)
        
# Footer
st.markdown("---")
st.markdown("📢 *Thank you for visiting our team page!*")

