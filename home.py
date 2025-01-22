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
    tab1, tab2 = st.tabs(["Project", "Power BI report"])
  #col1, col2, col3 = st.columns(3)
    with tab1:
        st.subheader('Supply Chain Analytics & Modeling')
        st.image("https://blog.cedarmanagement.co.uk/wp-content/uploads/2020/04/Supply-chain-blog-cover-desktop-size-15-04.png", caption="Supply Chain Analytics & Modeling")
  
        st.write('Skills: Data Analysis | Machine Learning & Model Training   ')
        st.write('Tools: Pandas | NumPy| Matplotlib / Seaborn | Scikit-learn  ')
        
        st.write(''' 

         #### Situation
         Global market competition demands efficient supply chain management.

         #### Task
         Improve supply chain operations through advanced analytics and modeling.

         #### Action
         - Phase 1: Data Analysis
         Data cleaning and preprocessing.
         Exploratory Data Analysis (EDA).
   
         - Phase 2: Modeling
         Cost optimization (Linear Regression).
   
         #### Tools and Techniques
         Python (Pandas, NumPy, Scikit-learn).
         Data visualization (Power BI).

         #### Result
         - Optimal production volume: 104
         -  Minimized manufacturing cost: $44.93
         - Model performance: Moderate (MSE=858.72, R2=-0.07)
''')
   
 
        st.write('''
           #### EDA Key Findings
           - **Defect Rates**: Haircare products have the highest defect rate (2.48%), followed by skincare (2.33%) and cosmetics (1.92%).
           - **Supply Chain Risk**: Top 10 highest-risk SKUs are primarily due to high lead times and low stock levels.
           - **Inventory Optimization**: EOQ analysis suggests optimal order quantities vary significantly across SKUs.
           - **Customer Segmentation**: Female customers generate highest average revenue ($6,095) for skincare products.
           - **Lead Time Optimization**: Sea transportation mode has the shortest average lead time (12.18 days).
           - **Best Route**: Route A has the shortest average lead time (14.7 days).

           #### Recommendations
            - Improve quality control measures for haircare products.
            - Implement risk mitigation strategies for high-risk SKUs.
            - Optimize inventory levels based on EOQ analysis.
            - Target female customers with personalized marketing campaigns.
            - Prioritize sea transportation mode for faster delivery.
            - Utilize Route A for efficient logistics.
           #### Future Analysis

            - Conduct root cause analysis for high defect rates.
            - Develop predictive models for demand forecasting.
            - Analyze cost savings from optimized inventory levels.
            - Investigate customer demographics' impact on purchasing behavior.
            - Evaluate sustainability implications of transportation modes.
        ''')
        st.write('''
           #### Modeling Key Findings
          - Linear Regression cost optimization model identifies optimal production volume (104) with minimized manufacturing cost (44.93).
          - Cost optimization model performance is moderate (MSE=858.72, R2=-0.07).
          - Further hyperparameter tuning and feature engineering may improve model performance.
          ''')
     with tab2:
         st.subheader('Supply Chain Analytics Power bi Report')
         st.write('''
        ### Introduction
          This report analyzes the DataCo Global supply chain dataset, providing insights into consumer behavior, sales trends and market geography. The report aims to identify areas for optimization and inform strategic business decisions.
        ### Objective
          Expand payment options and prioritize timely delivery. Focus on Western Europe and Estados Unidos markets. Regularly review sales trends and customer demographics.
          ''')
         st.image("https://github.com/Omniaahmedm/Portfolio/blob/main/Screenshot%202025-01-22%20135543.png?raw=true", caption="Supply Chain Analytics Dashboard")
