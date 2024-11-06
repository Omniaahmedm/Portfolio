import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu


# Function to load image and convert to base64

with st.sidebar:
    selected = option_menu(
        menu_title="Explore",
        options=['Overview', 'Projects'],
        menu_icon="cast"
    )

if selected == "Overview": 
    # Title and subtitle
   st.title("OMNIA AHMED ELAEIS")
   st.write("Data Scientist based in Riyadh")
   # URLs for your profiles
   github_url = "https://github.com/Omniaahmedm/DataSciencePortfolio"
   linkedin_url = "https://www.linkedin.com/in/omnia-elaeis"
   kaggle_url = "https://www.kaggle.com/omniaahmedmahmoud" 
# HTML code for clickable images
   html_code = f"""
   <a href="{github_url}" target="_blank"><img src="https://img.icons8.com/?size=100&id=ARy6tFUfwclb&format=png&color=000000" width="30" height="30"></a>
   <a href="{linkedin_url}" target="_blank"><img src="https://img.icons8.com/?size=100&id=85141&format=png&color=000000" width="30" height="30"></a>
   <a href="{kaggle_url}" target="_blank"><img src="https://img.icons8.com/?size=100&id=1iP83OYM1FL-&format=png&color=000000" width="30" height="30"></a>
   """
# Display the clickable images
   st.markdown(html_code, unsafe_allow_html=True)
   st.subheader("About:")
   st.write(''' I am an enthusiastic data scientist with a strong foundation in statistical analysis, machine learning, and data visualization. My passion lies in extracting insights from complex datasets and translating them into actionable strategies. I have experience working on various projects, ranging from predictive modeling to exploratory data analysis, and I am eager to contribute to innovative solutions that drive data-driven decisions. ''')
   st.subheader('Skills:')
   st.write('Python, SQL ,Data Analysis & Visualization, Pandas, GeoPandas, NumPy, Matplotlib, Seaborn, Power bi, Tableau, Machine Learning, Scikit-learn, TensorFlow, Statistical Analysis, Hypothesis testing, regression analysis, Geospatial Analysis, JupyterNotebooks, Git, AWS')
   st.subheader('Soft Skills:')
   st.write('Communication, teamwork, problem-solving, adaptability')
   st.subheader('Education:')
   st.write('Bachelor of Computer Science')
   st.write('University Of Khartoum,Khartoum,Sudan')
   st.subheader('Certifications:')
   st.write('IBM Data Science Professional Cerificate – Coursera | 2023')
elif selected == "Projects":
  col1, col2, col3 = st.columns([1, 1, 1])
     with col1:
        st.image("https://miro.medium.com/max/737/1*Xap6OxaZvD7C7eMQKkaHYQ.jpeg", caption="BProject1")
        st.markdown("[Link]()")
     with col2:
        st.image("https://miro.medium.com/max/737/1*Xap6OxaZvD7C7eMQKkaHYQ.jpeg", caption="Project2") 
        st.markdown("[Link]()")
    
