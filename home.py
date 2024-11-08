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
  col1, col2, col3 = st.columns(3)
 
  st.image("https://miro.medium.com/v2/resize:fit:828/format:webp/0*4oNUgPOA69JyeVwL.jpg", caption="Suspicious Web Threat Interactions")
  st.write('')
  st.write('''This Python-based pipeline is designed for web traffic analysis and threat detection, aimed at identifying suspicious or malicious interactions within web application traffic. The goal is to flag anomalous behaviors such as data exfiltration, brute-force attacks, DoS attacks, and geographic anomalies that may indicate potential cybersecurity threats. The pipeline performs several stages of data processing, feature engineering, and machine learning model training to detect malicious activity.

Key Steps in the Pipeline:
Data Preprocessing:

Datetime Conversion: Converts timestamps (e.g., creation_time, end_time) to datetime format to calculate session durations.
Missing Value Handling: Fills missing values in the dataset with zeros or other appropriate strategies.
Feature Engineering:

Aggregates web traffic data by source IP address (src_ip) and source IP country code (src_ip_country_code).
Flags suspicious IP addresses with high outgoing traffic (bytes_out) or multiple failed requests (HTTP 403 and 404).
Identifies potential exfiltration activities by comparing outbound to inbound traffic on common ports (HTTP/HTTPS).
Detects brute-force or DoS attempts based on a high number of requests and failed login attempts (403 errors).
Threat Detection:

Flags suspicious countries based on high traffic volume or failed requests.
Flags suspicious IP addresses or countries based on quantile thresholds (top 5% of traffic volume or failed requests).
Exfiltration detection based on the disproportionate amount of data sent compared to received data.
Brute-force and DoS detection based on abnormal request rates, particularly from specific source IP addresses.
Machine Learning Model:

XGBoost is used to classify whether a session is suspicious or not, using aggregated features (e.g., total bytes transferred, failed requests, request count).
The model is trained on the feature set, and performance is evaluated using accuracy, classification reports, and confusion matrices.
Evaluation:

Performance metrics like accuracy, precision, recall, and F1-score are used to evaluate the effectiveness of the model.
The SHAP library is employed for model interpretability, explaining the importance of features in predicting suspicious activities.''')
    st.write('Skills:')
    st.write('''- Data Preprocessing & Cleaning  
                - Feature Engineering      
                - Machine Learning & Model Training   
                - Evaluation & Interpretability ''') 
    st.write('Tools:')
    st.write('Pandas , NumPy,  Matplotlib / Seaborn, Scikit-learn, XGBoost ')
    st.markdown("[Link](https://portfolio-project1.streamlit.app/)")
 
    
