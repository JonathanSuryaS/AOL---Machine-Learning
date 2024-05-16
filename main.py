# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from streamlit_pandas_prodiling import st_profile_report

# Data
data = pd.read_csv('trainPlane.csv')


# Main Section
st.markdown("""
    <h1 style='text-align: center; color: white; font-family: Arial;'>Airline Satisfaction Prediction</h1>
    """, unsafe_allow_html=True)
# Short Description for Apps



# Navigations
st.sidebar.header("Program Menu")
### Homepage
if st.sidebar.button('Homepage'):
   st.title('Homepage')

### Exploratory Data analysis
if st.sidebar.button("Exploratory Data Analysis"):
  # Display dashboard content
  st.markdown("""
    <br><h2 style='text-align: left; color: white; font-family: Arial;'>Exploratory Data Analysis (EDA)</h2>
    """, unsafe_allow_html=True)
  
  # Data
  st.markdown("""
    <h3 style='text-align: left; color: white; font-family: Arial;'>Dataset</h3>
    """, unsafe_allow_html=True)
  data

  # Overciew
  st.markdown("""
    <h4 style='text-align: left; color: white; font-family: Arial;'>Overview</h4>
    """, unsafe_allow_html=True)
  pr = ProfileReport(data, explorative = True)
  st_profile_report(pr)

  # Data distribution
  fig, ax = plt.subplots(1, 2, figsize=(18, 8))
  data['satisfaction'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1])
  ax[0].set_ylabel('')
  ax[0].set_title('Satisfaction Distribution (Pie chart)')
  ax[0].set_title('Satisfaction Distribution (Count plot)')
  sns.countplot(data = data, x = 'satisfaction', hue = 'satisfaction')
  st.pyplot(fig)



if st.sidebar.button("Prediction"):
  st.title('Set')






  # Functions
def set_background(color):
    hex_color = f"#{color}"
    html = f"""
    <style>
    .stApp {{
        background-color: {hex_color};
    }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)

# Set the background color
set_background("7091E6")