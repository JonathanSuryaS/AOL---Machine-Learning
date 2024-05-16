# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


if st.sidebar.button("Exploratory Data Analysis"):
  # Display dashboard content
  st.markdown("""
    <h2 style='text-align: left; color: white; font-family: Arial;'>Exploratory Data Analysis</h2>
    """, unsafe_allow_html=True)



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