#Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# simport pandas_profiling as pp
import pickle
from streamlit_option_menu import option_menu













def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Exploratory Data Analysis", "Input Data", "Choose Model", "Result and Evaluation"], 
            icons=['house', 'bar-chart', 'upload', 'robot', 'check-circle'], 
            menu_icon="cast", 
            default_index=0
        )
        
        st.write(f"Section: {selected}")
    if selected == "Home":
        st.title("Home")
        st.write("Welcome to the Home page.")
    elif selected == "Exploratory Data Analysis":
        EDA()
    elif selected == "Input Data":
        prediction()
    elif selected == "Choose Model":
        ChooseModel()
    elif selected == "Result and Evaluation":
        st.title("Result and Evaluation")
        evaluation()
        

main()