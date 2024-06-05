# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# simport pandas_profiling as pp
import pickle
# from pandas_profiling import ProfileReport
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import plotly.express as px
from mlxtend.plotting import plot_decision_regions
from PIL import Image

# Global Variables
data = pd.read_csv('trainPlane.csv')
if 'to_predict' not in st.session_state:
    st.session_state.to_predict = None
if 'sc' not in st.session_state:
    from sklearn.preprocessing import StandardScaler
    st.session_state.sc = StandardScaler()
if 'ct' not in st.session_state:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    columns_to_encode = [0, 1, 3, 4]
    st.session_state.ct = ColumnTransformer(
        transformers=[
            ('encode', OneHotEncoder(), columns_to_encode)
        ],
        remainder='passthrough'  # Keep the rest of the columns as is
    )
if 'Model' not in st.session_state:
    st.session_state.Model = None

def plot_distribution(column):
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        data[column].value_counts().plot(
            kind='pie', ax=ax[0], autopct='%1.1f%%')
        ax[0].set_ylabel('')
        ax[0].set_title(f'{column} Distribution (Pie chart)')
        sns.pointplot(data=data, x=column, y='satisfaction', ax=ax[1])
        ax[1].set_title(f'Pointplot {column} vs Satisfaction')
        st.pyplot(fig)


def homepage():
    set_background('EEEEEE')
    st.title('Customer Airline Satisfaction')
    st.image('Airline-satisfaction-cover-1.png')
    st.write("""
        The Airlines Customer Satisfaction contains data about customer feedback and flight details for passenger who have flown with specific airlines service. These datas is structured and analyzed to predict customer satisfaction based on multiple parameters such as the customer data itself, and the airlines.
    """)

    space()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Customer data')
        st.image('customer.jpeg')
    with col2:
        space()
        st.write("""
            Customer data contains various type of information about customer that can be used for analysis, prediction, and improving business strategies such as airlines service. Customer data contains both categorical and numerical which can be seen below. These various feature can be analyzed to gain useful insight which can lead to improved marketing strategies and customer experience itself during flight.
        """)
    space()
    custdata = create_radio_input('Choose category', ['Category', 'Numerical'])

    if custdata == 'Category':
        custcat = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        catdata = st.selectbox('Select data', custcat)
        if catdata == 'Gender':
            coltype1, coltype2 = st.columns(2)
            with coltype1:
                st.header('Customer Gender')
                st.image('malefemale.jpeg')
            with coltype2:
                space()
                st.write('\n')
                st.write('''Customer Gender contain data about the gender of customer. This data can be used to analyze the pattern of customer and their preference, specifically between male and female customer as airlines passenger. By analyzing it alongside other relevant data points, businesses can gain insights that lead to improved marketing strategies, targeted product development, and a more personalized customer experience.''')
            space()
        elif catdata == 'Customer Type':
            coltype1, coltype2 = st.columns(2)
            with coltype1:
                st.header('Customer Type')
                st.image('loyal.jpeg')
            with coltype2:
                space()
                st.write('\n')
                st.write('''Customer type is crucial for business, understanding various customer type helps optimize marketing strategies, improve customer satisfaction, and could potentially increase loyalty. Customer type is divided by two categories, namely Loyal Customer and Disloyal Customer''')
            space()
            typ1,typ2 = st.columns(2)
            with typ2:
                disimage = Image.open('disloyal.png')
                disimage = disimage.resize((310, 200))
                st.subheader('Disloyal Customer')
                st.image(disimage)
                st.write('''Disloyal customer are those who switch airlines service / brands frequently and do not exhibit strong brand preference.''')
            
            with typ1:
                loyalimage = Image.open('loyal2.jpeg')
                loyalimage = loyalimage.resize((320, 200))
                st.subheader('Loyal Customer')
                st.image(loyalimage)
                st.write('''Loyal customer are those who repeatedly choose specific airlines service / brands over competitor due to strong preference for brand / service.''')
        elif catdata == 'Type of Travel':
            coltype1, coltype2 = st.columns(2)
            with coltype1:
                st.header('Type of Travel')
                st.image('travelheader.png')
            with coltype2:
                space()
                st.write('\n')
                st.write('''This data is about the type of travel customer had during flight. Type of travel is categorized into two categorical, Business Travel and Personal Travel. Type of Travel can be used to analyze which customer is easier to satisfy based on type of travel they had.''')
            space()
            typ1,typ2 = st.columns(2)
            with typ2:
                disimage = Image.open('personal.webp')
                disimage = disimage.resize((310, 200))
                st.subheader('Personal Travel')
                st.image(disimage)
                st.write('''In contrast, personal travel refers to trips embarked upon for non-business-related reasons, such as vacations, family visits, leisure outings, or special occasions like weddings or holidays. Personal travelers seek experiences tailored to their recreational, cultural, or social interests, with preferences ranging from budget-friendly options to luxury accommodations.''')
            
            with typ1:
                loyalimage = Image.open('business.webp')
                loyalimage = loyalimage.resize((320, 200))
                st.subheader('Business Travel')
                st.image(loyalimage)
                st.write('''This category comprises journeys undertaken for professional purposes, including corporate meetings, conferences, client visits, and work-related training or events. Business travelers prioritize factors such as schedule flexibility, proximity to business destinations, and amenities conducive to productivity during transit.''')

        elif catdata == 'Class':
            coltype1, coltype2 = st.columns(2)
            with coltype1:
                st.header('Airlines Classes')
                st.image('cabin.jpeg')
            with coltype2:
                space()
                st.write('\n')
                st.write('''Airlines Cabin (Classes) holds information about cabin in airlines service. These cabin aree categorized into three classes namely Business Class, Economy Plus, and Economy. With this data, we can analyze which customer is easier to satisfied given their classes.''')
            space()
            typ1,typ2,typ3 = st.columns(3)
            with typ1:
                business = Image.open('businessclass.jpeg')
                business = business.resize((310, 200))
                st.subheader('Business Class')
                st.image(business)
                st.write('''This category represents passengers who flew in business class. Business class typically offers more spacious seating, enhanced amenities, and a higher level of service compared to both economy class.''')
            
            with typ2:
                ecoplus = Image.open('ecoplus.png')
                ecoplus = ecoplus.resize((320, 200))
                st.subheader('Economy+ Class')
                st.image(ecoplus)
                st.write('''This category represents passengers who flew in economy plus class. Economy Plus is a class of service offered by some airlines that provides some additional benefits over economy class, such as extra legroom or wider seats.''')
            
            with typ3:
                eco = Image.open('economy.webp')
                eco = eco.resize((320, 200))
                st.subheader('Economy Class')
                st.image(eco)
                st.write('''This category represents passengers who flew in economy class. Economy class is the most basic and affordable class of service offered by airlines.''')

    else:
        custnum = ['Age', 'Flight Distance']
        custnum = st.selectbox('Select data', custnum)

    


def EDA():
    set_background("8EA7E9")
    # Display dashboard content
    st.markdown("""
    <br><h1 style='text-align: left; color: black; font-family: Arial;'>Exploratory Data Analysis (EDA)</h1>
    """, unsafe_allow_html=True)
    # Data
    st.markdown("""
    <h2 style='text-align: left; color: black; font-family: Arial;'>Dataset</h2>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: left; color: black; font-family: Arial;'> The dataset that will be utilized for the Machine Learning Project's implementation.</p>
    """, unsafe_allow_html=True)
    st.write(data.head())

    # Statistical Summary
    summary = st.checkbox('Statistical summary data', value=False)
    if summary:
        st.markdown("""
          <h4 style='text-align: left; color: white; font-family: Arial;'>Statistical summary of data.</h4>
        """, unsafe_allow_html=True)
    
        # Statistical Summary
        st.markdown("""
        <p style='text-align: left; color: white; font-family: Arial;'>Statistical summary for the dataset.</p>
        """, unsafe_allow_html=True)
        st.table(data.describe().T.round(2))
    
    #Data Entry
    entry = st.checkbox('Data Entry', value = False)
    if entry:
        st.table(data.isnull().sum())
        
    #Target distribution
    space()
    st.markdown("""
    <h2 style='text-align: left; color: black; font-family: Arial;'>Satisfaction distribution</h2>
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
    
        
    #Relationship Between Variables
    space()
    st.markdown("""
    <h2 style='text-align: left; color: black; font-family: Arial;'>Relationship Between Variables</h2>
    """, unsafe_allow_html=True)
    st.subheader('Customer data')
    data_options = ['Categorical data','Numerical data']
    customer_options = st.radio('Select Customer data types', data_options)
    if customer_options == 'Numerical data':
        cust_num = ['Age', 'Flight Distance']
        customer_numerical = st.selectbox('Select data', cust_num)
        if customer_numerical == 'Age':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            sns.histplot(data=data, x='Age', ax=ax[0], bins=30, kde=True)
            ax[0].set_ylabel('')
            ax[0].set_title('Customer Age Distribution (Histogram)')
            ax[1].set_title('Customer Age Distribution (Boxplot)')
            sns.boxplot(data=data, x='Age', ax = ax[1])
            st.pyplot(fig)
            fig = plt.figure(figsize=(18, 8))
            sns.histplot(data=data, x='Age', bins=30, hue='satisfaction', kde=True)
            plt.title('Customer Age vs Satisfaction')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            sns.histplot(data=data, x='Flight Distance', ax=ax[0], bins=30, kde=True)
            ax[0].set_ylabel('')
            ax[1].set_title('Flight Distance Distribution (Boxplot)')
            ax[0].set_title('Flight Distance Distribution (Histogram)')
            sns.boxplot(data=data, x='Flight Distance', ax = ax[1])
            st.pyplot(fig)
            fig = plt.figure(figsize=(18, 8))
            sns.histplot(data=data, x='Flight Distance',bins=30, hue='satisfaction', kde=True)
            plt.title('Flight Distance vs Satisfaction')
            st.pyplot(fig)

    else:
        cust_cat = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        customer_categorical = st.selectbox('Select data', cust_cat)
        if customer_categorical == 'Gender':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            data['Gender'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1], shadow = True)
            ax[0].set_ylabel('')
            ax[0].set_title('Gender Distribution (Pie chart)')
            ax[1].set_title('Pointplot Gender vs Satisfaction')
            sns.pointplot(data = data, x = 'Gender', y = 'satisfaction')
            st.pyplot(fig)
        elif customer_categorical == 'Customer Type':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            data['Customer Type'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1], shadow = True)
            ax[0].set_ylabel('')
            ax[0].set_title('Customer Type Distribution (Pie chart)')
            ax[1].set_title('Pointplot Customer Type vs Satisfaction')
            sns.pointplot(data = data, x = 'Customer Type', y = 'satisfaction')
            st.pyplot(fig)
        elif  customer_categorical == 'Type of Travel':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            data['Type of Travel'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0, 0.1], shadow = True)
            ax[0].set_ylabel('')
            ax[0].set_title('Type of Travel Distribution (Pie chart)')
            ax[1].set_title('Type of Travel vs Satisfaction')
            sns.pointplot(data = data, x = 'Type of Travel', y = 'satisfaction')
            st.pyplot(fig)
        elif customer_categorical == 'Class':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            data['Class'].value_counts().plot(kind = 'pie', ax = ax[0], autopct = '%1.1f%%', explode = [0.1, 0, 0], shadow = True)
            ax[0].set_ylabel('')
            ax[0].set_title('Class Distribution (Pie chart)')
            ax[1].set_title('Class vs Satisfaction')
            sns.pointplot(data = data, x = 'Class', y = 'satisfaction')
            st.pyplot(fig)

    space()
    st.subheader('Plane data')
    plane_data = ['Categorical data','Numerical data']
    plane_option = st.radio('Select Plane data types', plane_data)
    if plane_option == 'Categorical data':
        plane_cat = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                     'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                     'On-board service', 'Leg room service', 'Baggage handling', 'Checking service', 'Inflight service',
                     'Cleanliness']
        categorical_plane = st.selectbox('Select data', plane_cat)
        if categorical_plane == 'Inflight wifi service':
            plot_distribution('Inflight wifi service')
        elif categorical_plane == 'Departure/Arrival time convenient':
            plot_distribution('Departure/Arrival time convenient')
        elif categorical_plane == 'Ease of Online booking':
            plot_distribution('Ease of Online booking')
        elif categorical_plane == 'Gate location':
            plot_distribution('Gate location')
        elif categorical_plane == 'Food and drink':
            plot_distribution('Food and drink')
        elif categorical_plane == 'Online boarding':
            plot_distribution('Online boarding')
        elif categorical_plane == 'Seat comfort':
            plot_distribution('Seat comfort')
        elif categorical_plane == 'Inflight entertainment':
            plot_distribution('Inflight entertainment')
        elif categorical_plane == 'On-board service':
            plot_distribution('On-board service')
        elif categorical_plane == 'Leg room service':
            plot_distribution('Leg room service')
        elif categorical_plane == 'Baggage handling':
            plot_distribution('Baggage handling')
        elif categorical_plane == 'Checking service':
            plot_distribution('Checking service')
        elif categorical_plane == 'Inflight service':
            plot_distribution('Inflight service')
        elif categorical_plane == 'Cleanliness':
            plot_distribution('Cleanliness')
    elif plane_option == 'Numerical data':
        plane_num = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
        numerical_plane = st.selectbox('Select data', plane_num)
        if numerical_plane == 'Departure Delay in Minutes':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            sns.histplot(data = data, x = 'Departure Delay in Minutes', ax = ax[0], bins = 30, kde = True)
            ax[0].set_ylabel('')
            ax[1].set_title('Departure Delay in Minutes Distribution (Boxplot)')
            ax[0].set_title('Departure Delay in Minutes Distribution (Histogram)')
            sns.boxplot(data = data, x = 'Departure Delay in Minutes')
            st.pyplot(fig)
            fig = plt.figure(figsize = (18, 8))
            sns.scatterplot(data = data, x = 'Arrival Delay in Minutes', y = 'Departure Delay in Minutes', hue = 'satisfaction')
            plt.title('Arrival Delay in Minutes vs Departure Delay in Minutes')
            st.pyplot(fig)
        elif numerical_plane == 'Arrival Delay in Minutes':
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))
            sns.histplot(data = data, x = 'Departure Delay in Minutes', ax = ax[0], bins = 30, kde = True)
            ax[0].set_ylabel('')
            ax[1].set_title('Arrival Delay in Minutes Distribution (Boxplot)')
            ax[0].set_title('Arrival Delay in Minutes Distribution (Histogram)')
            sns.boxplot(data = data, x = 'Arrival Delay in Minutes')
            st.pyplot(fig)
            fig = plt.figure(figsize = (18, 8))
            sns.scatterplot(data = data, x = 'Arrival Delay in Minutes', y = 'Departure Delay in Minutes', hue = 'satisfaction')
            plt.title('Arrival Delay in Minutes vs Departure Delay in Minutes')
            st.pyplot(fig)
    
    space()
    st.subheader('Feature Correlation')
    numerical = data.select_dtypes(include='number')
    fig = plt.figure(figsize=(18, 8))
    plt.title('Correlation between features')
    sns.heatmap(data = numerical.corr(), annot= True, linewidths=0.2, cmap='Blues')
    st.pyplot(fig)
    
def EDA2():
    profile = ProfileReport(data, title = 'Report')
    st_profile_report(profile)
    
def create_radio_input(label, options):
    result = st.radio(label=label, options=options)
    st.markdown(
        """<style>
      div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
          font-size: 20px;
      }
      </style>
      """, unsafe_allow_html=True)
    return result

def create_number_input(label, min_value, max_value, value, step):
    number = st.number_input(
        label, min_value=min_value, max_value=max_value, value=value, step=step)
    st.markdown(
        """<style>
      div[class*="stNumberInput"] > label > div[data-testid="stMarkdownContainer"] > p {
          font-size: 20px;
      }
      </style>
      """, unsafe_allow_html=True)
    return number

def create_slider(label, min_value, max_value):
    slider = st.slider(label, min_value, max_value)
    st.markdown(
        """<style>
      div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
          font-size: 20px;
      }
      </style>
      """, unsafe_allow_html=True)
    return slider


def prediction():
    set_background('FBA834')
    st.markdown("""
    <br><h1 style='text-align: left; color: black; font-family: Arial;'>Does the customer satisfied with airline service?</h1>
    """, unsafe_allow_html=True)
    st.image('satisfied.jpg')

    # Insert Gender
    TempCustomerGender = create_radio_input(
        'Insert Customer Gender', ["Male", "Female"])

    # Insert Customer Type
    TempCustomerType = create_radio_input(
        'Insert Customer Type', ["Loyal Customer", "Disloyal Customer"])

    # Age
    Tempage = create_number_input(
        "Enter your age:", min_value=0, max_value=100, value=25, step=1)

    # Insert Type of Travel
    TempTypeTravel = create_radio_input('Insert Type of Travel', [
                                    'Personal Travel', 'Business travel'])

    # Insert Class of Airlines
    TempClass = create_radio_input('Insert Travel Class', [
                               'Business', 'Eco Plus', 'Eco'])

    # Inser Flight Distance
    TempDistance = create_number_input(
        "Flight Distance", min_value=30, max_value=5000, value=30, step=1)

    # Airlines Service
    Tempinflight_wifi = create_slider('Inflight wifi service', 0, 5)
    Tempdeparture_arrival_time = create_slider('Departure/Arrival time convenient', 0, 5)
    Temponline_booking = create_slider('Ease of Online booking', 0, 5)
    Tempgate_location = create_slider('Gate location', 0, 5)
    Tempfood_drink = create_slider('Food and drink', 0, 5)
    Temponline_boarding = create_slider('Online boarding', 0, 5)
    Tempseat_comfort = create_slider('Seat comfort', 0, 5)
    Tempinflight_entertainment = create_slider('Inflight entertainment', 0, 5)
    Temponboard_service = create_slider('On-board service', 0, 5)
    Templegroom_service = create_slider('Leg room service', 0, 5)
    Tempbaggage_handling = create_slider('Baggage handling', 0, 5)
    Tempcheckin_service = create_slider('Checkin service', 0, 5)
    Tempinflight_service = create_slider('Inflight service', 0, 5)
    Tempcleanliness = create_slider('Cleanliness', 0, 5)

    # Departure & Arrival Late
    TempDeparture = create_number_input(
        "Departure Delay", min_value=0, max_value=1600, value=10, step=1)
    TempArrival = create_number_input(
        "Arrival Delay", min_value=0, max_value=1600, value=10, step=1)

    # Inputted data
    input_data = {
        'Gender': [TempCustomerGender],
        'Customer Type': [TempCustomerType],
        'Age': [Tempage],
        'Type of Travel': [TempTypeTravel],
        'Class': [TempClass],
        'Flight Distance': [TempDistance],
        'Inflight wifi service': [Tempinflight_wifi],
        'Departure/Arrival time convenient': [Tempdeparture_arrival_time],
        'Ease of Online booking': [Temponline_booking],
        'Gate location': [Tempgate_location],
        'Food and drink': [Tempfood_drink],
        'Online boarding': [Temponline_boarding],
        'Seat comfort': [Tempseat_comfort],
        'Inflight entertainment': [Tempinflight_entertainment],
        'On-board service': [Temponboard_service],
        'Leg room service': [Templegroom_service],
        'Baggage handling': [Tempbaggage_handling],
        'Checkin service': [Tempcheckin_service],
        'Inflight service': [Tempinflight_service],
        'Cleanliness': [Tempcleanliness],
        'Departure Delay in Minutes': [TempDeparture],
        'Arrival Delay in Minutes': [TempArrival]
    }

    space()
    st.markdown("""
    <br><h3 style='text-align: left; color: white; font-family: Arial;'>Your Data</h3>
    """, unsafe_allow_html=True)
    input_data = pd.DataFrame(input_data)
    st.write(input_data)

    st.write(
        '<style>div.Widget.row-widget.stButton>div{display:flex;justify-content:center;}</style>', unsafe_allow_html=True)
    # Create a centered button
    PredictButton = st.button("Submit Data")
    
    space()
    if PredictButton:
        input_data.drop(columns=['Cleanliness', 'Departure Delay in Minutes', 'Inflight wifi service'], inplace= True)
        st.session_state.to_predict = input_data
        st.success('Data Submitted, Create your model to predict!')

def getFeatures():
    data.drop(columns=['id', 'Unnamed: 0'], inplace=True)
    data.loc[data['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    data.loc[data['satisfaction'] == 'neutral or dissatisfied', 'satisfaction'] = 0
    data['satisfaction'] = data['satisfaction'].astype(int)
    data.dropna(inplace=True)

    X = data.drop(columns=['satisfaction', 'Cleanliness',
                      'Departure Delay in Minutes', 'Inflight wifi service']).values
    y = data['satisfaction'].values
    X = np.array(st.session_state.ct.fit_transform(X))
    return X,y 

def training(selected, test_size):
    from sklearn.model_selection import train_test_split
    X, y = getFeatures()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True)
    columns_to_scale = [9, 10, 23]
    X_train[:, columns_to_scale] = st.session_state.sc.fit_transform(X_train[:, columns_to_scale])
    X_test[:, columns_to_scale] = st.session_state.sc.transform(X_test[:, columns_to_scale])
    st.session_state.train = X_train
    st.session_state.validation = y_train
    st.session_state.test = X_test
    st.session_state.result = y_test
    
    st.markdown("""
    <style>
    .training {
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    #model_options = ['Logistic Regression', 'NaiveBayes',
               #'SVM','DecisionTree','RandomForest', 'XGBoost']
    space()
    if selected == 'Logistic Regression':
        Logistic()
    elif selected == 'NaiveBayes':
        NaiveBayes()
    elif selected == 'DecisionTree':
        DecisionTree()
    elif selected == 'RandomForest':
        RandomForest()
    elif selected == 'XGBoost':
        XGBoost()
        
# Model Model untuk Klasifikasi
def Logistic():
    from sklearn.linear_model import LogisticRegression
    import streamlit as st

    st.title("Logistic Regression Parameter Tuning")

    penalty_options = ['None (default)', 'l2', 'l1', 'elasticnet']
    penalty_map = {'None (default)': None, 'l2': 'l2', 'l1': 'l1', 'elasticnet': 'elasticnet'}
    penalty = penalty_map[st.selectbox('Penalty', penalty_options)]

    C = st.slider('C (Inverse of regularization strength, default=1.0)', 0.01, 10.0, 1.0)

    solver_options = ['lbfgs (default)', 'newton-cg', 'liblinear', 'sag', 'saga']
    if penalty == 'none':
        solver_options = ['lbfgs (default)', 'newton-cg', 'sag', 'saga']
    elif penalty == 'l1':
        solver_options = ['liblinear', 'saga']
    elif penalty == 'elasticnet':
        solver_options = ['saga']
    solver_map = {'lbfgs (default)': 'lbfgs', 'newton-cg': 'newton-cg', 'liblinear': 'liblinear', 'sag': 'sag', 'saga': 'saga'}
    solver = solver_map[st.selectbox('Solver', solver_options)]

    if penalty == 'elasticnet':
        l1_ratio = st.slider('l1_ratio (for elasticnet, default=0.5)', 0.0, 1.0, 0.5)
    else:
        l1_ratio = None

    max_iter = st.slider('Max iterations (default=100)', 100, 1000, 100)
    tol = st.slider('Tolerance for stopping criteria (default=1e-4)', 1e-5, 1e-1, 1e-4)
    class_weight_options = ['None (default)', 'balanced']
    class_weight_map = {'None (default)': None, 'balanced': 'balanced'}
    class_weight = class_weight_map[st.selectbox('Class weight', class_weight_options)]
    verbose = st.checkbox('Verbose (default=False)', value=False)
    n_jobs = st.slider('Number of jobs (parallelization, default=1)', -1, 4, 1)
    random_state = st.slider('Random state (default=None)', 0, 100, 42)

    predict = st.button('Deploy Model')
    if predict:
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state
        )
        # Ensure the following session state variables are correctly set before calling model.fit
        # st.session_state.train, st.session_state.validation, st.session_state.test
        model.fit(st.session_state.train, st.session_state.validation)
        st.session_state.Model = model
        pred = model.predict(st.session_state.test)
        st.session_state.Pred = pred
        st.success('Model Created, please check Result and Evaluation')


def DecisionTree():
    from sklearn.tree import DecisionTreeClassifier
    st.title("Decision Tree Parameter Tuning")

    criterion_options = ['gini (default)', 'entropy', 'log_loss']
    criterion_map = {'gini (default)': 'gini', 'entropy': 'entropy', 'log_loss': 'log_loss'}
    criterion = criterion_map[st.selectbox('Criterion', criterion_options)]

    splitter_options = ['best (default)', 'random']
    splitter = st.selectbox('Splitter', splitter_options, index=0)

    max_depth = st.slider('Max depth (default=None)', 1, 100, 10)
    min_samples_split = st.slider('Min samples split (default=2)', 2, 20, 2)
    min_samples_leaf = st.slider('Min samples leaf (default=1)', 1, 20, 1)
    max_features = st.slider('Max features (default=None)', 1, st.session_state.train.shape[1], st.session_state.train.shape[1])
    random_state = st.slider('Random state (default=None)', 0, 100, 42)

    predict = st.button('Deploy Model')
    if predict:
        model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        model.fit(st.session_state.train, st.session_state.validation)
        pred = model.predict(st.session_state.test)
        st.session_state.Pred = pred
        st.session_state.Model = model
        pred = model.predict(st.session_state.test)
        st.session_state.Pred = pred
        st.success('Model Created, please check Result and Evaluation')

def RandomForest():
    from sklearn.ensemble import RandomForestClassifier
    st.title("Random Forest Parameter Tuning")

    n_estimators = st.slider('Number of estimators (default=100)', 10, 500, 100)
    criterion_options = ['gini (default)', 'entropy', 'log_loss']
    criterion_map = {'gini (default)': 'gini', 'entropy': 'entropy', 'log_loss': 'log_loss'}
    criterion = criterion_map[st.selectbox('Criterion', criterion_options)]

    max_depth = st.slider('Max depth (default=None)', 1, 100, 10)
    min_samples_split = st.slider('Min samples split (default=2)', 2, 20, 2)
    min_samples_leaf = st.slider('Min samples leaf (default=1)', 1, 20, 1)
    max_features = st.slider('Max features (default=None)', 1, st.session_state.train.shape[1], st.session_state.train.shape[1])
    random_state = st.slider('Random state (default=None)', 0, 100, 42)
    n_jobs = st.slider('Number of jobs (parallelization, default=1)', -1, 4, 1)
    verbose = st.checkbox('Verbose (default=False)', value=False)

    predict = st.button('Deploy Model')
    if predict:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        model.fit(st.session_state.train, st.session_state.validation)
        st.session_state.Model = model
        pred = model.predict(st.session_state.test)
        st.session_state.Pred = pred
        st.success('Model Created, please check Result and Evaluation')

def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    st.title("Naive Bayes Parameter Tuning")

    st.write("Gaussian Naive Bayes does not have hyperparameters to tune in the same way other models do. Click 'Deploy Model' to use the default Gaussian Naive Bayes model.")

    predict = st.button('Deploy Model')
    if predict:
        model = GaussianNB()
        model.fit(st.session_state.train, st.session_state.validation)
        st.session_state.Model = model
        st.success('Model Created, please check Result and Evaluation')

def XGBoost():
    from xgboost import XGBClassifier
    st.title("XGBoost Parameter Tuning")

    n_estimators = st.slider('Number of estimators (default=100)', 10, 500, 100)
    max_depth = st.slider('Max depth (default=6)', 1, 15, 6)
    learning_rate = st.slider('Learning rate (default=0.3)', 0.01, 1.0, 0.3)
    subsample = st.slider('Subsample (default=1.0)', 0.1, 1.0, 1.0)
    colsample_bytree = st.slider('Colsample by tree (default=1.0)', 0.1, 1.0, 1.0)
    gamma = st.slider('Gamma (default=0)', 0.0, 10.0, 0.0)
    random_state = st.slider('Random state (default=0)', 0, 100, 0)
    n_jobs = st.slider('Number of jobs (parallelization, default=1)', -1, 4, 1)
    verbosity = st.slider('Verbosity (default=1)', 0, 3, 1)

    predict = st.button('Deploy Model')
    if predict:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=verbosity
        )
        model.fit(st.session_state.train, st.session_state.validation)
        st.session_state.Model = model
        pred = model.predict(st.session_state.test)
        st.session_state.Pred = pred
        st.success('Model Created, please check Result and Evaluation')


# def svm():
#     from sklearn.svm import SVC
#     st.title("SVC Parameter Tuning")
    
#     # Define the parameter options
#     kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
#     kernel = st.selectbox('Kernel', kernel_options)
    
#     C = st.slider('C (Regularization parameter)', 0.01, 10.0, 1.0)
#     degree = st.slider('Degree (for poly kernel)', 1, 5, 3)
#     gamma_options = ['scale', 'auto']
#     gamma = st.selectbox('Gamma', gamma_options)
#     coef0 = ''
#     if kernel in ['poly', 'sigmoid']:
#         coef0 = st.slider('Coef0 (for poly/sigmoid kernels)', 0.0, 1.0, 0.0)
#     else:
#         coef0 = 0.0  # Default value for other kernels
#     shrinking = st.checkbox('Shrinking', value=True)
#     probability = st.checkbox('Probability', value=False)
#     tol = st.slider('Tolerance for stopping criterion', 1e-5, 1e-1, 1e-3)
#     cache_size = st.slider('Cache size (MB)', 100, 1000, 200)
#     class_weight_options = ['None', 'balanced']
#     class_weight = st.selectbox('Class weight', class_weight_options)
#     class_weight = None if class_weight == 'None' else 'balanced'
#     verbose = st.checkbox('Verbose', value=False)
#     max_iter = st.slider('Max iterations', -1, 1000, -1)
#     decision_function_shape_options = ['ovr', 'ovo']
#     decision_function_shape = st.selectbox('Decision function shape', decision_function_shape_options)
#     break_ties = st.checkbox('Break ties', value=False)
#     random_state = st.slider('Random state', 0, 100, 42)
    
#     predict = st.button('Deploy Model')
#     if predict:
#         svc = SVC(
#             C=C,
#             kernel=kernel,
#             degree=degree,
#             gamma=gamma,
#             coef0=coef0,
#             shrinking=shrinking,
#             probability=probability,
#             tol=tol,
#             cache_size=cache_size,
#             class_weight=class_weight,
#             verbose=verbose,
#             max_iter=max_iter,
#             decision_function_shape=decision_function_shape,
#             break_ties=break_ties,
#             random_state=random_state
#         )   
#         svc.fit(st.session_state.train, st.session_state.validation)
#         st.session_state.Model = svc
        

def ChooseModel():
    set_background('FAF2D3')
    st.markdown("""
    <br><h1 style='text-align: left; color: black; font-family: Arial;'>Model for Classification</h1>
    """, unsafe_allow_html=True)
    
    
    def create_slider(label, min_value, max_value):
        slider = st.slider(label, min_value, max_value)
        st.markdown(
            """<style>
          div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
              font-size: 20px;
          }
          </style>
          """, unsafe_allow_html=True)
        return slider
    test_size = create_slider('Insert Test Size', 0.1, 0.9)
    
    st.markdown("""
    <style>
    .custom-label {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    # Add a label with the custom CSS class
    st.markdown('<div class="custom-label">Choose Models</div>', unsafe_allow_html=True)
    
    model_options = ['Logistic Regression', 'NaiveBayes',
                'DecisionTree','RandomForest', 'XGBoost']
    selected_models = st.selectbox('', model_options)
    training(selected_models, test_size)
    


    

def evaluation():
    if st.session_state.Model == None:
        set_background('ff0f0f')
        st.subheader('No model has been trained')
        st.subheader('Please choose the given models in Model page')
        return
    
    
    set_background('B1AFFF')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Model evaluation criteria')
    st.image('eval.png')
    space()
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
    from sklearn.model_selection import cross_val_score
    pred = st.session_state.Model.predict(st.session_state.test)
    st.session_state.Pred = pred
    tries = st.session_state.Model.predict(st.session_state.train)

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    
    if isinstance(st.session_state.Model, LogisticRegression):
        st.subheader('Model Evaluation: Logistic Regression')
    elif isinstance(st.session_state.Model, GaussianNB):
        st.subheader('Model Evaluation: Naive Bayes Classifier')
    elif isinstance(st.session_state.Model, RandomForestClassifier):
        st.subheader('Model Evaluation: Random Forest Classifier')
    elif isinstance(st.session_state.Model, DecisionTreeClassifier):
        st.subheader('Model Evaluation: Decision Tree Classifier')
    elif isinstance(st.session_state.Model, XGBClassifier):
        st.subheader('Model Evaluation: XGBoost Classifier')
    
    #rep = classification_report(pred, st.session_state.result)
    st.write(f'Accuracy score: {accuracy_score(pred, st.session_state.result)*100:.2f}%')
    st.write(f'F1 score: {f1_score(pred, st.session_state.result)*100:.2f}%')
    st.write(f'Recall score: {recall_score(pred, st.session_state.result)*100:.2f}%')
    st.write(f'Precision score: {precision_score(pred, st.session_state.result)*100:.2f}%')
    st.write(f'Cross Validation Score(10): {(cross_val_score(st.session_state.Model, st.session_state.test, st.session_state.result, cv=10).mean())*100:.2f}%')
    space()
    st.header('Model Confusion Matrix')

    col1, col2 = st.columns(2)
    # Training
    #st.subheader('Training Set')
    labels = ['neutral or dissatisfied', 'satisfied']
    cm_train = confusion_matrix(tries, st.session_state.validation)
    sns.heatmap(cm_train, cmap='Blues', annot=True, fmt='.0f', linecolor='white',xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix of Training Set', size=15)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    st.pyplot()
    space()
    # Test set confusion matrix
    #st.subheader('Test Set')
    cm_test = confusion_matrix(pred, st.session_state.result)
    sns.heatmap(cm_test, cmap='Blues', annot=True, fmt='.0f', linecolor='white',xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix of Test Set', size=15)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    st.pyplot()
    space()

    

def result():
    if st.session_state.Model == None:
        set_background('ff0f0f')
        st.subheader('No model has been trained')
        st.subheader('Please choose the given models in Model page')
        return
    if st.session_state.to_predict is not None:

        
        #PCA
        from sklearn.decomposition import PCA
        st.subheader('Decision Boundary')
        pca1 = PCA(n_components=3)
        pca2 = PCA(n_components=2)
        X_pca1 = pca1.fit_transform(st.session_state.test)
        X_pca2 = pca2.fit_transform(st.session_state.test)
        y = st.session_state.Pred
        plt.figure(figsize=(8, 6))
        color_map = {0: 'red', 1: 'blue'}
        
        if isinstance(st.session_state.Model, LogisticRegression):
            clf = LogisticRegression()
        elif isinstance(st.session_state.Model, GaussianNB):
            clf = GaussianNB()
        elif isinstance(st.session_state.Model, RandomForestClassifier):
            clf = RandomForestClassifier()
        elif isinstance(st.session_state.Model, DecisionTreeClassifier):
            clf = DecisionTreeClassifier()
        elif isinstance(st.session_state.Model, XGBClassifier):
            clf = XGBClassifier()

        clf.fit(X_pca2, y)
        plots = plot_decision_regions(X_pca2, y, clf=clf)
        fig = plt.gcf()
        fig
        space()


        # Define custom labels for the legend
        labels = {0: 'neutral or dissatisfied', 1: 'satisfied'}

        # Create the 3D scatter plot
        fig = px.scatter_3d(x=X_pca1[:, 0], y=X_pca1[:, 1], z=X_pca1[:, 2], color=y,
                            color_discrete_map=color_map, labels={'color': 'Labels'}, color_continuous_scale=list(color_map.values()))

        # Update legend title
        fig.update_layout(coloraxis_colorbar=dict(title='Labels'))

    
        
        # Plot the data points
        fig.update_traces(marker=dict(size=5))
        
        # Add title and labels
        st.subheader('Decision Boundary in 3D')
        fig.update_layout( 
                          scene=dict(xaxis_title='Principal Component 1',
                                     yaxis_title='Principal Component 2',
                                     zaxis_title='Principal Component 3'))
        
        # Show the figure
        st.plotly_chart(fig)

        to_predict = st.session_state.to_predict.values
        to_predict = st.session_state.ct.transform(to_predict)
        columns_to_scale = [9, 10, 23]
        to_predict[:, columns_to_scale] = st.session_state.sc.fit_transform(to_predict[:, columns_to_scale])
        if st.session_state.Model.predict(to_predict):
            st.success('Customer Satisfied')
        else:
            st.error('Customer is not satisfied')
        #predict.iloc[:, [0, 1, 14]] = st.session_state.sc.transform(predict.iloc[:, [0, 1, 14]])
    else:
        set_background('ff0f0f')
        st.subheader('No data has been submitted')
        st.subheader('Please insert data in Input Data page')


def main():
    # Get the session state
    session_state = st.session_state
    
    with st.sidebar:
        # Initialize selected section if it doesn't exist in the session state
        if "selected_section" not in session_state:
            session_state.selected_section = "Home"
        
        selected = option_menu(
            "Main Menu", 
            ["Home", "Exploratory Data Analysis", "Input Data", "Model", "Evaluation", "Result"], 
            icons = ['house', 'bar-chart', 'upload', 'robot', 'check-circle', 'trophy'], 
            menu_icon="cast", 
            default_index=0
        )
        
        st.write(f"Section: {selected}")
        
        # Update the selected section in the session state
        session_state.selected_section = selected

    # Get the selected section from session state
    selected = session_state.selected_section

    if selected == "Home":
        homepage()
    elif selected == "Exploratory Data Analysis":
        EDA()
    elif selected == "Input Data":
        prediction()
    elif selected == "Model":
        ChooseModel()
    elif selected == "Evaluation":
        st.title("Evaluation")
        evaluation()
    elif selected == 'Result':
        st.title('Result')
        result()


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


def space():
    st.markdown("""
              <br><br>""", unsafe_allow_html=True)


data = pd.read_csv('trainPlane.csv')
main()
