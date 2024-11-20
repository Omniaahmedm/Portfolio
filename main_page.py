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
        options=['Overview', 'EDA'],
        menu_icon="cast"
    )

if selected == "Overview": 
    # Title and subtitle
   st.title("OMNIA AHMED ELAEIS")
   st.write("Data Scientist based in Riyadh")
  
elif selected == "EDA":
