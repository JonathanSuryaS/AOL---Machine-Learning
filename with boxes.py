# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling as pp


def EDA(data):
     # Display dashboard content
    st.markdown("""
      <br><h1 style='text-align: left; color: white; font-family: Arial;'>Exploratory Data Analysis (EDA)</h1>
      """, unsafe_allow_html=True)

    # Data
    st.markdown("""
      <h2 style='text-align: left; color: white; font-family: Arial;'>Dataset</h2>
      """, unsafe_allow_html=True)
    st.markdown("""
      <p style='text-align: left; color: white; font-family: Arial;'> The dataset that will be utilized for the Machine Learning Project's implementation.</p>
      """, unsafe_allow_html=True)
    data

    ###Overview
    st.markdown("""
                <br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Overview</h3>
      """, unsafe_allow_html=True)
    st.markdown("""
        <h4 style='text-align: left; color: white; font-family: Arial;'>Statistical summary of data.</h4>
      """, unsafe_allow_html=True)

    #### Statistical Summary
    st.markdown("""
      <p style='text-align: left; color: white; font-family: Arial;'>Statistical summary for the dataset.</p>
      """, unsafe_allow_html=True)
    data.describe().T


    ### TBD
    # pr = ProfileReport(data, explorative=True)
    # st.write(data)
    # st_profile_report(pr)

    # Relationship between variables
    st.markdown("""
      <br><h2 style='text-align: left; color: white; font-family: Arial;'>Relationship Betwween Variables</h2>
      """, unsafe_allow_html=True)
    # Target Distribution
    st.markdown("""
      <h3 style='text-align: left; color: white; font-family: Arial;'>Distribution of Target (Satisfaction)</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['satisfaction'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Satisfaction Distribution (Pie chart)')
    ax[1].set_title('Satisfaction Distribution (Count plot)')
    sns.countplot(data = data, x = 'satisfaction', hue = 'satisfaction')
    st.pyplot(fig)
    data.loc[data['satisfaction']=='satisfied', 'satisfaction'] = 1
    data.loc[data['satisfaction']=='neutral or dissatisfied', 'satisfaction'] = 0
    data['satisfaction'] = data['satisfaction'].astype(int)
    spaces()

    #Unamed 0
    st.markdown(container_css, unsafe_allow_html=True)
    with st.container():
      count_sum = data['Unnamed: 0'].value_counts().sum()
      st.markdown(f"""
      <div class="box">
          <h3 style='text-align: left;'>Unnamed: 0</h3>
          <p style='text-align: justify;'>This features does not represent anything else beside rows (each customer). Thus, the columns / features need to be dropped</p>
          <p style='font-size: 20px; font-weight: bold;'>{count_sum}</p>
      </div>
      """, unsafe_allow_html=True)
      spaces()

    # ID
    st.markdown(container_css, unsafe_allow_html=True)
    with st.container():
      count_sum = data['id'].value_counts().sum()
      st.markdown(f"""
      <div class="box">
          <h3 style='text-align: left;'>ID</h3>
          <p style='text-align: justify;'>This features does not represent anything else beside rows (each customer). Thus, the columns / features need to be dropped</p>
          <p style='font-size: 20px; font-weight: bold;'>{count_sum}</p>
      </div>
      """, unsafe_allow_html=True)
      spaces()
    

    # Gender
    st.markdown("""<br>
      <h3 style='text-align: justify; color: white; font-family: Arial;'>Gender</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Gender'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Gender Distribution (Pie chart)')
    ax[1].set_title('Pointplot Gender vs Satisfaction')
    sns.pointplot(data = data, x = 'Gender', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown(container_css, unsafe_allow_html=True)
    with st.container():
      count_sum = data['Unnamed: 0'].value_counts().sum()
      st.markdown(f"""
      <div class="box">
          <p style='text-align: justify; color: white; font-family: Arial;'>Male customer is easier to satisfy as it have higher 
                chance for satisfied rather than Female, having average of satsifaction around 0.44, with female around 0.42 </p>
      </div>
      """, unsafe_allow_html=True)
      spaces()
      
    # Customer Type
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Customer Type</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Customer Type'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Customer Type Distribution (Pie chart)')
    ax[1].set_title('Pointplot Customer Type vs Satisfaction')
    sns.pointplot(data = data, x = 'Customer Type', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>it's easier to satisfy loyal customer than disloyal customer, as loyal customer usually means 
                that this customer is used to buy airline ticket from a single company many times (loyal), having average of satisfaction above 0.45, while disloyal customer 
                having average below 0.25</p>
      """, unsafe_allow_html=True)

    # Age
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Customer Age</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.histplot(data = data, x = 'Age', ax = ax[0], bins = 30, kde = True)
    ax[0].set_ylabel('')
    ax[0].set_title('Customer Age Distribution (Histogram)')
    ax[1].set_title('Customer Age Distribution (Boxplot)')
    sns.boxplot(data = data, x = 'Age')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The distribution of an age is mostly unimodal, where the data does not have an outlier as seen in the boxplot.
                The central 50% of customer ages range from 30 to 50 years.
                The median age is approximately 40 years.
                The overall age range of customers spans from about 10 to 80 years, with no outliers.</p>
      """, unsafe_allow_html=True)

    fig = plt.figure(figsize = (18, 8))
    sns.histplot(data = data, x = 'Age', bins = 30, hue = 'satisfaction', kde = True)
    plt.title('Customer Age vs Satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Customer with age range around 40 - 60 are easier to satisfy, as the average of satisfaction 
                increase for this specific range.</p>
      """, unsafe_allow_html=True)

    # Type of Travel
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Type of Travel</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Type of Travel'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Type of Travel Distribution (Pie chart)')
    ax[1].set_title('Type of Travel vs Satisfaction')
    sns.pointplot(data = data, x = 'Type of Travel', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>For type of travel, customer with Business travel is easier to satisfy as the chance to be satisfied is
                significantly higher, having average of satisfaction 0.6 rather than Personal travel with average of satisfaction around 0.1.</p>
      """, unsafe_allow_html=True)
    
    # Class
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Class</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Class'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Class Distribution (Pie chart)')
    ax[1].set_title('Class vs Satisfaction')
    sns.pointplot(data = data, x = 'Class', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The class shows that, the higher the class of Airline seat, starting with
                Business class, Eco Plus, and Eco, the higher the chance of passenger being satisfied, with Business Class
                having average around 0.7, followed by eco plus above 0.2 and the least is eco with average below 0.2</p>
      """, unsafe_allow_html=True)
    
    # Flight Distance
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Flight Distance</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.histplot(data = data, x = 'Flight Distance', ax = ax[0], bins = 30, kde = True)
    ax[0].set_ylabel('')
    ax[1].set_title('Flight Distance Distribution (Boxplot)')
    ax[0].set_title('Flight Distance Distribution (Histogram)')
    sns.boxplot(data = data, x = 'Flight Distance')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The histogram appears to perform a right - skewed distribution,
                which means most of data points are clustered on the left side of the graph, with a 'tail' extending towards the right side.
                In context of flight distance, this means that most flights are relatively short distance.
                </p>
      """, unsafe_allow_html=True)

def main():
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
    EDA(data)




  if st.sidebar.button("Prediction"):
    st.title('Set')



  # Functions
def spaces():
   st.markdown("""
      <h3 style='text-align: left; color: white; font-family: Arial;'><br></h3>
      """, unsafe_allow_html=True)

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

container_css = """
<style>
.box {
    background-color: #191923; /* Box background color */
    padding: 20px;           /* Box padding */
    border-radius: 10px;     /* Box rounded corners */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Box shadow */
    margin-bottom: 20px;     /* Bottom margin for spacing */
}
</style>
"""


main()