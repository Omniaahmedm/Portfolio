import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

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
    #st.subheader('Projects :')
    # Cache the data loading function
    @st.cache_data
    def load_data():
        dataset_url = "Bank Customer Churn Prediction.csv"
        data = pd.read_csv(dataset_url)
        return data
    # Project 1
    st.header("Project 1: Bank Customer Churn Prediction")
    Introduction,DASHBOARD,Prediction,Documentation=st.tabs(["Introduction","DASHBOARD","Prediction",'Documentation'])
    with Introduction: 
      st.image("https://miro.medium.com/max/737/1*Xap6OxaZvD7C7eMQKkaHYQ.jpeg", caption="Bank Customer Churn Prediction")
      st.subheader('Description:')   
      st.write("Predicting bank churn involves identifying customers who are likely to leave the bank.")
   
      st.markdown("[Link]()")
    
    with DASHBOARD: 
       st.subheader('Bank Customer Chrun Prediction Dashborad')
       st.image("D:\Myportfoilo\dashborad.png")
    
        
    with Prediction: 
        
        
        filename = 'Churn_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        
        df = load_data()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Set Streamlit options
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
        st.title("Customer Churn Prediction")
        categorical_features = df.select_dtypes(include='O')
        categorical_dummies=pd.get_dummies(categorical_features,dtype=float) 
        num_cols= df.select_dtypes(include=np.number)
        df=pd.concat([categorical_dummies ,num_cols],axis=1)
        with st.form("my_form"):
            country = st.selectbox('Country', ['France', 'Germany', 'Spain'])
            gender = st.selectbox('Gender', ['Male', 'Female'])
            balance=st.number_input(label='Balance ',step=0.001,format="%.6f")
            age = st.number_input('Age', min_value=18, max_value=100)
            credit_score = st.number_input('Credit Score', min_value=250, max_value=850)
            tenure = st.number_input('Tenure', min_value=0, max_value=10)
            credit_card = st.selectbox('Has Credit Card', [0, 1])
            active_member = st.selectbox('Active Member', [0, 1])
            products_number = st.number_input('Number of Products', min_value=1, max_value=4)
            estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=200000)

            submitted = st.form_submit_button("Submit")

        if submitted:
             user_data = {
                'country_France': 1 if country == 'France' else 0,
                'country_Germany': 1 if country == 'Germany' else 0,
                'country_Spain': 1 if country == 'Spain' else 0,
                'gender_Female': 1 if gender == 'Female' else 0,
                'gender_Male': 1 if gender == 'Male' else 0,
                'balance': balance,
                'age': age,
                'credit_score': credit_score,
                'tenure': tenure,
                'credit_card': credit_card,
                'active_member': active_member,
                'products_number': products_number,
                'estimated_salary': estimated_salary
                  }
             input_df = pd.DataFrame(user_data, index=[0])
             scaler=StandardScaler()
             scaler.fit(input_df)
             input_scaled = scaler.transform(input_df)
                 # Make prediction
             prediction = loaded_model.predict(input_scaled)
             prediction_proba = loaded_model.predict_proba(input_scaled)[:, 1]

             # Display result
        
             if prediction[0] == 1:
                 st.write("Prediction: The customer is likely to churn.")
             else:
                 st.write("Prediction: The customer is unlikely to churn.")

    with Documentation:
        st.subheader(' Project Overview')
        st.write('''This project demonstrates how to predict bank customer churn using a machine learning model.
                  The project is divided into the following sections:''')

        st.write('1- Introduction: Overview of the project and its objectives.')
        st.write('2- DASHBOARD: Visual representation of the customer churn prediction analysis.')
        st.write('3- Prediction: Detailed explanation of the prediction process using a pre-trained model.')
        st.write('4- Documentation: Comprehensive documentation of the code and its functionality.')
        st.subheader('How to Run the Project')
        st.write('1- Clone the Repository: Clone the GitHub repository to your local machine.')
       
        st.write('2- Install Dependencies: Install the required dependencies using pip install -r requirements.txt.')
       
        st.write('3- Run the Streamlit App: Use the command streamlit run app.py to start the Streamlit app.')
        st.subheader('Dependencies')
        st.write('pandas')
        st.write('numpy')
        st.write('scikit-learn')        
        st.write('streamlit')
        st.subheader('Conclusion')
        st.write('''This project provides a comprehensive guide to predicting bank customer churn using a machine learning model. By following the steps outlined in this documentation,
                  you can implement similar churn prediction models for your own data.''')

                 
        


    st.header("Project 2 : Customer Segmentation using RFM ")
    Introduction,segments,Documentation=st.tabs(["Introduction","segments",'Documentation'])
    with Introduction: 
      st.image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*p9t-VYFneDs6Bt00F1Z8Fw.png", caption="Customer Segmentation")
      st.subheader('Description:')   
      st.write("Customer Segmentaion Using RFM analysis that analyzing customer behavior based on three key factors: recency, frequency, and monetary value.")
   
      st.markdown("[Link]()")
    with segments: 
        st.write("Customer Segmentaion Using RFM analysis")
        @st.cache_data
        def load_data():
            data= pd.read_csv('marketing_campaign.csv', sep="\t")
            return data
        data=load_data()
        data = data.dropna()
        if 'Dt_Customer' in data.columns:
           data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', dayfirst=True)
        else:
          st.error("The column 'Dt_Customer' is missing from the data.")
        # Preprocess data
        data['Recency'] = (pd.to_datetime('2024-01-01') - data['Dt_Customer']).dt.days
        data['TotalPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'] + data["NumDealsPurchases"]
        data['Monetary'] = (data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] +
                    data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'])
        # Ensure numeric values
        data['Recency'] = pd.to_numeric(data['Recency'], errors='coerce')
        data['TotalPurchases'] = pd.to_numeric(data['TotalPurchases'], errors='coerce')
        data['Monetary'] = pd.to_numeric(data['Monetary'], errors='coerce')

       # Drop rows with NaN values
        data.dropna(subset=['Recency', 'TotalPurchases', 'Monetary'], inplace=True)
        # Calculate RFM metrics
        rfm = data[['ID', 'Recency', 'TotalPurchases', 'Monetary']]

       # Assign RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['TotalPurchases'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

       # Combine RFM scores
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

      # Segment customers
        def segment_customer(df):
          if df['RFM_Score'] == '555':
              return 'Best Customers'
          elif df['RFM_Score'] == '111':
            return 'Lost Customers'
          elif df['RFM_Score'][0] == '5':
            return 'Loyal Customers'
          elif df['RFM_Score'][1] == '5':
            return 'Frequent Customers'
          elif df['RFM_Score'][2] == '5':
             return 'Big Spenders'
          else:
           return 'Others'

        rfm['Segment'] = rfm.apply(segment_customer, axis=1)

       # Analyze segments
        segment_analysis = rfm.groupby('Segment').agg({
          'Recency': 'mean',
          'TotalPurchases': 'mean',
          'Monetary': ['mean', 'count']
         }).reset_index()
        st.title('RFM Segmentation Analysis')

        st.write('## Segment Analysis')
        st.dataframe(segment_analysis)

        #st.write('## RFM Table')
        #st.dataframe(rfm)

        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

# Example data for training the model
# Replace this with your actual data
       
   # Splitting the data into features and target
        X = rfm[['Recency', 'TotalPurchases', 'Monetary']]
        y = rfm['Segment']
        st.write(print(y))
# Training the model
        rfm = RandomForestClassifier()
        rfm.fit(X, y)

# Streamlit form for new customer prediction
        st.write('## Predict New Customer Segment')
        with st.form(key='predict_form'):
             recency = st.number_input('Recency (days)', min_value=0)
             total_purchases = st.number_input('Total Purchases', min_value=0)
             monetary = st.number_input('Monetary Value', min_value=0)
             submit_button = st.form_submit_button(label='Predict')
     
        if submit_button:
            new_data = pd.DataFrame({
            'Recency': [recency],
            'TotalPurchases': [total_purchases],
           'Monetary': [monetary]
         })
            prediction = rfm.predict(new_data)
            st.write(f'The predicted segment for the new customer is: {prediction[0]}')
    with Documentation:
       st.subheader(' Project Overview')
       st.write('''This project demonstrates how to use RFM analysis for customer segmentation and predict customer segments using machine learning. 
                The project is divided into the following sections:''')

       st.write('Introduction: Overview of the project and its objectives.')
       st.write('  Segments: Detailed explanation of the RFM analysis process and customer segmentation.')
       st.write(' Documentation: Comprehensive documentation of the code and its functionality.')
       st.subheader('How to Run the Project')
       st.write('1- Clone the Repository: Clone the GitHub repository to your local machine.')
       
       st.write('2- Install Dependencies: Install the required dependencies using pip install -r requirements.txt.')
       
       st.write('3- Run the Streamlit App: Use the command streamlit run app.py to start the Streamlit app.')
       st.subheader('Dependencies')
       st.write('pandas')
       st.write('numpy')
       st.write('scikit-learn')        
       st.write('streamlit')
       st.subheader('Conclusion')
       st.write('''This project provides a comprehensive guide to customer segmentation using RFM analysis and demonstrates how to predict customer segments using a machine learning model. 
                By following the steps outlined in this documentation, you can implement similar customer segmentation and prediction models for your own data.''')

