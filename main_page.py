import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image



# Title and subtitle
st.title("Omnia Ahmed Elsaei")
st.subheader("Data Scientist based in Riyadh")

# Display profile picture
image = Image.open("C:\Users\maxia\Downloads\IMG1.png")  # Replace 'profile_pic.jpg' with your image file path
st.image(image, caption='Omnia Ahmed Elsaei', width=200)
st.title("Omnia Ahmed Elaeis")
st.subheader("Data Scientist based in Riyadh")

# Contact information
st.write("Email: omnia@example.com")

# About section
st.header("About")
st.write("""
I am an enthusiastic data scientist with a strong foundation in statistical analysis, machine learning, and data visualization. My passion lies in extracting insights from complex datasets and translating them into actionable strategies. I have experience working on various projects, ranging from predictive modeling to exploratory data analysis, and I am eager to contribute to innovative solutions that drive data-driven decisions.
""")



# Sample data
text = """
Python SQL Data Analysis Visualization Pandas NumPy Matplotlib Seaborn Power bi Tableau Machine Learning Scikit-learn TensorFlow Statistical Analysis Hypothesis testing regression analysis JupyterNotebooks Git AWS Communication teamwork problem-solving adaptability
"""

# Skills section
st.header("Skills")
skills = ["Python", "NumPy", "SQL", "Data Analysis & Visualization", "Pandas", "Scikit-learn", "Machine Learning", "TensorFlow", "Statistical Analysis", "Hypothesis Testing", "Regression Analysis", "Jupyter Notebooks", "Git", "AWS"]
st.write(", ".join(skills))

# Soft Skills section
st.header("Soft Skills")
soft_skills = ["Communication", "Teamwork", "Problem-solving", "Adaptability"]
st.write(", ".join(soft_skills))