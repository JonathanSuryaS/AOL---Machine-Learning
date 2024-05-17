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

    # Overview
    st.markdown("""
                <br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Overview</h3>
      """, unsafe_allow_html=True)
    st.markdown("""
        <h4 style='text-align: left; color: white; font-family: Arial;'>Statistical summary of data.</h4>
      """, unsafe_allow_html=True)

    # Statistical Summary
    st.markdown("""
      <p style='text-align: left; color: white; font-family: Arial;'>Statistical summary for the dataset.</p>
      """, unsafe_allow_html=True)
    data.describe().T

    # TBD
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
    data['satisfaction'].value_counts().plot(
        kind='pie', ax=ax[0], autopct='%1.1f%%', explode=[0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Satisfaction Distribution (Pie chart)')
    ax[1].set_title('Satisfaction Distribution (Count plot)')
    sns.countplot(data=data, x='satisfaction', hue='satisfaction')
    st.pyplot(fig)
    data.loc[data['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    data.loc[data['satisfaction'] ==
             'neutral or dissatisfied', 'satisfaction'] = 0
    data['satisfaction'] = data['satisfaction'].astype(int)

    # Unamed 0
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Unnamed: 0</h3>
      """, unsafe_allow_html=True)
    st.write(data['id'].value_counts().sum())
    st.markdown("""
      <p style='text-align: left; color: white; font-family: Arial;'>This features does not represent anything else beside rows. Thus, the columns / features
                need to be dropped</p>
      """, unsafe_allow_html=True)

    # ID
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>ID</h3>
      """, unsafe_allow_html=True)
    st.write(data['id'].value_counts().sum())
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>This features does not represent anything else beside rows (each customer). Thus, the columns / features
                need to be dropped</p>
      """, unsafe_allow_html=True)

    # Gender
    st.markdown("""<br>
      <h3 style='text-align: justify; color: white; font-family: Arial;'>Gender</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Gender'].value_counts().plot(
        kind='pie', ax=ax[0], autopct='%1.1f%%', explode=[0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Gender Distribution (Pie chart)')
    ax[1].set_title('Pointplot Gender vs Satisfaction')
    sns.pointplot(data=data, x='Gender', y='satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Male customer is easier to satisfy as it have higher 
                chance for satisfied rather than Female, having average of satsifaction around 0.44, with female around 0.42 </p>
      """, unsafe_allow_html=True)

    # Customer Type
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Customer Type</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Customer Type'].value_counts().plot(
        kind='pie', ax=ax[0], autopct='%1.1f%%', explode=[0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Customer Type Distribution (Pie chart)')
    ax[1].set_title('Pointplot Customer Type vs Satisfaction')
    sns.pointplot(data=data, x='Customer Type', y='satisfaction')
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
    sns.histplot(data=data, x='Age', ax=ax[0], bins=30, kde=True)
    ax[0].set_ylabel('')
    ax[0].set_title('Customer Age Distribution (Histogram)')
    ax[1].set_title('Customer Age Distribution (Boxplot)')
    sns.boxplot(data=data, x='Age')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The distribution of an age is mostly unimodal, where the data does not have an outlier as seen in the boxplot.
                The central 50% of customer ages range from 30 to 50 years.
                The median age is approximately 40 years.
                The overall age range of customers spans from about 10 to 80 years, with no outliers.</p>
      """, unsafe_allow_html=True)

    fig = plt.figure(figsize=(18, 8))
    sns.histplot(data=data, x='Age', bins=30, hue='satisfaction', kde=True)
    plt.title('Customer Age vs Satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Customer frequency that are satisfied increased with age range around 40 - 60, having the 
      highest frequency among satisfied customer for this specific range.</p>
      """, unsafe_allow_html=True)

    # Type of Travel
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Type of Travel</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Type of Travel'].value_counts().plot(
        kind='pie', ax=ax[0], autopct='%1.1f%%', explode=[0, 0.1])
    ax[0].set_ylabel('')
    ax[0].set_title('Type of Travel Distribution (Pie chart)')
    ax[1].set_title('Type of Travel vs Satisfaction')
    sns.pointplot(data=data, x='Type of Travel', y='satisfaction')
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
    data['Class'].value_counts().plot(kind='pie', ax=ax[0], autopct='%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Class Distribution (Pie chart)')
    ax[1].set_title('Class vs Satisfaction')
    sns.pointplot(data=data, x='Class', y='satisfaction')
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
    sns.histplot(data=data, x='Flight Distance', ax=ax[0], bins=30, kde=True)
    ax[0].set_ylabel('')
    ax[1].set_title('Flight Distance Distribution (Boxplot)')
    ax[0].set_title('Flight Distance Distribution (Histogram)')
    sns.boxplot(data=data, x='Flight Distance')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The histogram appears to perform a right - skewed distribution,
                which means most of data points are clustered on the left side of the graph, with a 'tail' extending towards the right side.
                In context of flight distance, this means that most flights are relatively short distance.<br>
                On the other hands, the boxplot shows that the median flight distance is approximately 2,500 miles, with 50% of flights
                fall between 1,500 to 3,500 miles.
                </p>
      """, unsafe_allow_html=True)
    fig = plt.figure(figsize=(18, 8))
    sns.histplot(data=data, x='Flight Distance',
                 bins=30, hue='satisfaction', kde=True)
    plt.title('Flight Distance vs Satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The frequency of customer that satisfied about the flight decrease as the flight distance increase, however it can be noted aswell
      that the dissasitified customer frequency also decrease as the flight distance increase.
                </p>
      """, unsafe_allow_html=True)

    # Inflight Wifi Service
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Inflight Wifi Service</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Inflight wifi service'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Inflight wifi service Distribution (Pie chart)')
    ax[1].set_title('Pointplot Inflight wifi service vs Satisfaction')
    sns.pointplot(data = data, x = 'Inflight wifi service', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The pointplot shows that the Customer Satisfaction at its peak at 0 and 5, however since 0 only represented by only 3%
      of customer, we can safely assume that as the inflight wifi service increase (starting from 3), so is the average customer satisfaction.
                </p>
    """, unsafe_allow_html=True)
    
    
    # Departure / Arrival time convenient
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Departure / Arrival Time Convenient</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Departure/Arrival time convenient'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Departure/Arrival time convenient Distribution (Pie chart)')
    ax[1].set_title('Pointplot Departure/Arrival time convenient vs Satisfaction')
    sns.pointplot(data = data, x = 'Departure/Arrival time convenient', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Departure / Arrival Time Convenient shows that the higher the category, the lower 
      average of customer satisfaction</p>
    """, unsafe_allow_html=True)
    
    #Ease of online booking
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Ease of Online Booking</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Ease of Online booking'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Ease of Online booking Distribution (Pie chart)')
    ax[1].set_title('Pointplot Ease of Online booking vs Satisfaction')
    sns.pointplot(data = data, x = 'Ease of Online booking', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>The pointplot of Ease of Online Booking shows the higher the category, the higher the average of customer satisfaction.
      We can see that although zero is higher than one, zero only represented b 4.3%, thus we can assume the higher the category, the higher the average satisfaction.</p>
    """, unsafe_allow_html=True)
    
    
    #Gate Location
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Gate Location</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Gate location'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Gate Location Distribution (Pie chart)')
    ax[1].set_title('Pointplot Gate Location vs Satisfaction')
    sns.pointplot(data = data, x = 'Gate location', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>For Gate Location,  category three is the lowest and the average of satisfaction decreased towards three, and then rise again.
      For category zero, it is irrelevant since it
      represented by 0%, meaning a very small portion of data.</p>
    """, unsafe_allow_html=True)

  # Food and Drink
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Food and Drink</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Food and drink'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Food and drink booking Distribution (Pie chart)')
    ax[1].set_title('Pointplot Food and drink vs Satisfaction')
    sns.pointplot(data = data, x = 'Food and drink', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Food and drinks also represent the same as majority, where as the category rise, so is the average of
      customer satisfaction, zero is also irrelevant since it only represented by 0.1%</p>
    """, unsafe_allow_html=True)

  # Online Boarding
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Online Boarding</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Online boarding'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Online boarding Distribution (Pie chart)')
    ax[1].set_title('Pointplot Online boarding vs Satisfaction')
    sns.pointplot(data = data, x = 'Online boarding', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Starting from category 3, the higher Online Boarding Category, the higher average of customer satisfaction, whive category below 3 
      does not show much difference of average satisfaction.</p>
    """, unsafe_allow_html=True)
    
    
    # Seat Comfort
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Seat Comfort</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Seat comfort'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Seat Comfort Distribution (Pie chart)')
    ax[1].set_title('Point plot Seat Comfort boarding vs Satisfaction')
    sns.pointplot(data = data, x = 'Seat comfort', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Seat comfort also represent the same thing, while the category below 3 does not show much difference between
      average of satisfaction, higher category than 3 shows higher average of satisfaction.</p>
    """, unsafe_allow_html=True)
    
    # Inflight Entertainment
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Inflight Entertainment</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Inflight entertainment'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Inflight entertainment Distribution (Pie chart)')
    ax[1].set_title('Point plot Inflight entertainment vs Satisfaction')
    sns.pointplot(data = data, x = 'Inflight entertainment', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Inflight Entertainment shows that the higher the category of Inflight Entertainment, the higher the average
      of customer satisfaction</p>
    """, unsafe_allow_html=True)

    # On - Board Service
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>On - Board Service</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['On-board service'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('On-board service  Distribution (Pie chart)')
    ax[1].set_title('Point plot On-board service  vs Satisfaction')
    sns.pointplot(data = data, x = 'On-board service', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>On - Board Service shows athat the higher the category of On - Board Service, the higher the average
      of customer satisfaction</p>
    """, unsafe_allow_html=True)
  

    # Leg Room Service
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Leg Room Service</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Leg room service'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Leg room service  Distribution (Pie chart)')
    ax[1].set_title('Point plot Leg room service  vs Satisfaction')
    sns.pointplot(data = data, x = 'Leg room service', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Leg Room Service also shows the same, the higher, the greater average of customer satisfaction.
      However, category 2 and 3 is not showing much difference</p>
    """, unsafe_allow_html=True)
    
    # Baggage Handling
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Seat Comfort</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Baggage handling'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Baggage handling  Distribution (Pie chart)')
    ax[1].set_title('Point plot Baggage handling   vs Satisfaction')
    sns.pointplot(data = data, x = 'Baggage handling', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Starting from category 3, the higher Online Boarding Category, the higher average of customer satisfaction</p>
    """, unsafe_allow_html=True)
    
    # Inflight Service
    st.markdown("""<br>
      <h3 style='text-align: left; color: white; font-family: Arial;'>Seat Comfort</h3>
      """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Inflight service'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%')
    ax[0].set_ylabel('')
    ax[0].set_title('Inflight service  Distribution (Pie chart)')
    ax[1].set_title('Point plot Inflight service vs Satisfaction')
    sns.pointplot(data = data, x = 'Inflight service', y = 'satisfaction')
    st.pyplot(fig)
    st.markdown("""
      <p style='text-align: justify; color: white; font-family: Arial;'>Starting from category 3, the higher Online Boarding Category, the higher average of customer satisfaction</p>
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
    # Homepage
    if st.sidebar.button('Homepage'):
        st.title('Homepage')

# Exploratory Data analysis
    if st.sidebar.button("Exploratory Data Analysis"):
        EDA(data)

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
set_background("508CA4")

container_css = """
<style>
.box {
    background-color: #3f4c6b; /* Box background color */
    padding: 20px;           /* Box padding */
    border-radius: 10px;     /* Box rounded corners */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Box shadow */
    margin-bottom: 20px;     /* Bottom margin for spacing */
}
</style>
"""

data = pd.read_csv('trainPlane.csv')
EDA(data)
