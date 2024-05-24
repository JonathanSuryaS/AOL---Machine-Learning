# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# simport pandas_profiling as pp
import pickle
from streamlit_option_menu import option_menu


# Global Variables
data = pd.read_csv('trainPlane.csv')
CustomerGende= None
CustomerType = None
age = None
TypeTravel = None 
Class = None
Distance = None
inflight_wifi = None
departure_arrival_time = None
online_booking = None
gate_location = None
food_drink = None
online_boarding = None
seat_comfort = None
inflight_entertainment = None
onboard_service = None
legroom_service = None
baggage_handling = None
checkin_service = None
inflight_service = None
cleanliness = None

#Model
Model = None

test = None
result = None
train = None
validation = None


def EDA():
    def plot_distribution(data, column, title):
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        data[column].value_counts().plot(
            kind='pie', ax=ax[0], autopct='%1.1f%%')
        ax[0].set_ylabel('')
        ax[0].set_title(f'{title} Distribution (Pie chart)')
        sns.pointplot(data=data, x=column, y='satisfaction', ax=ax[1])
        ax[1].set_title(f'Pointplot {title} vs Satisfaction')
        st.pyplot(fig)

    def add_section(title, data, column, text):
        st.markdown(f"""<br>
            <h3 style='text-align: left; color: white; font-family: Arial;'>{title}</h3>
            """, unsafe_allow_html=True)
        plot_distribution(data, column, title)
        st.markdown(f"""
            <p style='text-align: justify; color: white; font-family: Arial;'>{text}</p>
            """, unsafe_allow_html=True)

    set_background("508CA4")
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
    <br><h2 style='text-align: left; color: white; font-family: Arial;'>Relationship Between Variables</h2>
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
    GenderText = 'Male customer is easier to satisfy as it have higher chance for satisfied rather than Female, having average of satsifaction around 0.44, with female around 0.42 '
    add_section("Gender", data, 'Gender', GenderText)

    # Customer Type
    CustomerTypeText = """it's easier to satisfy loyal customer than disloyal customer, as loyal customer usually means 
              that this customer is used to buy airline ticket from a single company many times (loyal), having average of satisfaction above 0.45, while disloyal customer 
              having average below 0.25
  """
    add_section('Customer Type', data, 'Customer Type', CustomerTypeText)

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
    TypeTravelTxt = """
    For type of travel, customer with Business travel is easier to satisfy as the chance to be satisfied is
              significantly higher, having average of satisfaction 0.6 rather than Personal travel with average of satisfaction around 0.1.
    """
    add_section('Type of Travel', data, 'Type of Travel', TypeTravelTxt)

    # Class
    ClassTxt = """
    The class shows that, the higher the class of Airline seat, starting with
              Business class, Eco Plus, and Eco, the higher the chance of passenger being satisfied, with Business Class
              having average around 0.7, followed by eco plus above 0.2 and the least is eco with average below 0.2
    """
    add_section('Class', data, 'Class', ClassTxt)

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
    InflightWifiTxt = """
    The pointplot shows that the Customer Satisfaction at its peak at 0 and 5, however since 0 only represented by only 3%
    of customer, we can safely assume that as the inflight wifi service increase (starting from 3), so is the average customer satisfaction.
    """
    add_section('Inflight Wifi Service', data,
                'Inflight wifi service', InflightWifiTxt)

    # Departure / Arrival time convenient
    DepartureTxt = """
    Departure / Arrival Time Convenient shows that the higher the category, the lower 
    average of customer satisfaction
    """
    add_section('Departure / Arrival time convenient', data,
                'Departure/Arrival time convenient', DepartureTxt)

    # Ease of online booking
    EaseTxt = """
    The pointplot of Ease of Online Booking shows the higher the category, the higher the average of customer satisfaction.
    We can see that although zero is higher than one, zero only represented b 4.3%, thus we can assume the higher the category, the higher the average satisfaction.
    """
    add_section('Ease of online booking', data,
                'Ease of Online booking', EaseTxt)

    # Gate Location
    GateTxt = """
    For Gate Location,  category three is the lowest and the average of satisfaction decreased towards three, and then rise again.
    For category zero, it is irrelevant since it
    represented by 0%, meaning a very small portion of data.
    """
    add_section('Gate Location', data, 'Gate location', GateTxt)

    # Food and Drink
    FoodTxt = """
    Food and drinks also represent the same as majority, where as the category rise, so is the average of
    customer satisfaction, zero is also irrelevant since it only represented by 0.1%
    """
    add_section('Food and Drink', data, 'Food and drink', FoodTxt)

    # Online Boarding
    BoardingTxt = """
    Starting from category 3, the higher Online Boarding Category, the higher average of customer satisfaction, whive category below 3 
    does not show much difference of average satisfaction
    """
    add_section('Online Boarding', data, 'Online boarding', BoardingTxt)

    # Seat Comfort
    SeatTxt = """
    Seat comfort also represent the same thing, while the category below 3 does not show much difference between
    average of satisfaction, higher category than 3 shows higher average of satisfaction.
    """
    add_section('Seat Comfort', data, 'Seat comfort', SeatTxt)

    # Inflight Entertainment
    EntTxt = """
    Inflight Entertainment shows that the higher the category of Inflight Entertainment, the higher the average
    of customer satisfaction
    """
    add_section('Inflight Entertainment', data,
                'Inflight entertainment', EntTxt)

    # On - Board Service
    BoardTxt = """
      On - Board Service shows athat the higher the category of On - Board Service, the higher the average
      of customer satisfaction
      """
    add_section('On - Board Service', data, 'On-board service', BoardTxt)

    # Leg Room Service
    LegTxt = """
    Leg Room Service also shows the same, the higher, the greater average of customer satisfaction.
    However, category 2 and 3 is not showing much difference
    """
    add_section('Leg room service', data, 'Leg room service', LegTxt)

    # Baggage Handling
    BaggageTxt = """
    Baggage handling shows the higher the category, the higher average of customer satisfaction. But it can be seen 
    that the average of satisfaction slightly decrease for category 3.
    """
    add_section('Baggage Handling', data, 'Baggage handling', BaggageTxt)

    # Inflight Service
    ServiceTxt = """
    Inflight Service shows also shows the higher the category, the higher average satisfaction of customer. However, it can be seen
    that the average of satisfaction slightly decrease for category 3.
    """
    add_section('Inflight Service', data, 'Inflight service', ServiceTxt)

    # Cleanliness
    CleanTxt = """
    For Cleanliness, it shows that as the higher category, so is the average of customer satisfaction.
    """
    add_section('Cleanliness', data, 'Cleanliness', CleanTxt)

    # Departure  & Arrival Delay in Minutes
    st.markdown("""<br>
    <h3 style='text-align: left; color: white; font-family: Arial;'>Departure  & Arrival Delay in Minutes</h3>
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.histplot(data=data, x='Departure Delay in Minutes',
                 ax=ax[0], bins=30, kde=True)
    ax[0].set_ylabel('')
    ax[1].set_title('Departure Delay in Minutes Distribution (Boxplot)')
    ax[0].set_title('Departure Delay in Minutes Distribution (Histogram)')
    sns.boxplot(data=data, x='Departure Delay in Minutes')
    st.pyplot(fig)
    st.markdown("""
    <p style='text-align: justify; color: white; font-family: Arial;'>For Cleanliness, it shows that as the higher category, so is the average of customer satisfaction.</p>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.histplot(data=data, x='Departure Delay in Minutes',
                 ax=ax[0], bins=30, kde=True)
    ax[0].set_ylabel('')
    ax[1].set_title('Departure Delay in Minutes Distribution (Boxplot)')
    ax[0].set_title('Departure Delay in Minutes Distribution (Histogram)')
    sns.boxplot(data=data, x='Departure Delay in Minutes')
    st.pyplot(fig)
    st.markdown("""
    <p style='text-align: justify; color: white; font-family: Arial;'>For Cleanliness, it shows that as the higher category, so is the average of customer satisfaction.</p>
    """, unsafe_allow_html=True)

    fig = plt.figure(figsize=(18, 8))
    sns.scatterplot(data=data, x='Arrival Delay in Minutes',
                    y='Departure Delay in Minutes')
    plt.title('Arrival Delay in Minutes vs Departure Delay in Minutes')
    plt.show()
    st.pyplot(fig)
    st.markdown("""
    <p style='text-align: justify; color: white; font-family: Arial;'>Arrival Delay in Minutes and Departure Delay in Minutes, performs a linear pattern, where the higher Arrival Delay in Minutes
    the higher Departure delay in Minutes.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <br><h2 style='text-align: left; color: white; font-family: Arial;'>Relationship Between Variables</h2>
    """, unsafe_allow_html=True)
    numerical = data.select_dtypes(include='number')
    fig = plt.figure(figsize=(18, 8))
    plt.title('Correlation between features')
    sns.heatmap(data=numerical.corr(), annot=True, linewidths=0.2)
    st.pyplot(fig)
    st.markdown("""
    <p style='text-align: justify; color: white; font-family: Arial;'>The graph shows us that some feature has a high colinearity to each other, such as Departure Delay in Minutes
    with Arrival Delay in Minutes, and many more.</p>
    """, unsafe_allow_html=True)


def prediction():
    set_background('f1e0C5')
    st.markdown("""
    <br><h1 style='text-align: left; color: white; font-family: Arial;'>Does the customer satisfied with airline service?</h1>
    """, unsafe_allow_html=True)
    st.image('satisfied.jpg')

    # Gender & Type
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

    # Insert Gender
    TempCustomerGender = create_radio_input(
        'Insert Customer Gender', ["Male", "Female"])

    # Insert Customer Type
    TempCustomerType = create_radio_input(
        'Insert Customer Type', ["Loyal", "Disloyal"])

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
        CustomerGender = TempCustomerGender
        CustomerType = TempCustomerType
        age = Tempage
        TypeTravel = TempTypeTravel
        Class = TempClass
        Distance = TempDistance
        inflight_wifi = Tempinflight_wifi
        departure_arrival_time = Tempdeparture_arrival_time
        online_booking = Temponline_booking
        gate_location = Tempgate_location
        food_drink = Tempfood_drink
        online_boarding = Temponline_boarding
        seat_comfort = Tempseat_comfort
        inflight_entertainment = Tempinflight_entertainment
        onboard_service = Temponboard_service
        legroom_service = Templegroom_service
        baggage_handling = Tempbaggage_handling
        checkin_service = Tempcheckin_service
        inflight_service = Tempinflight_service
        cleanliness = Tempcleanliness
        st.success('Data Submitted, Predict your data!')


def training(selected, test_size):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X, y = getFeatures()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    X_train.iloc[:, [0, 1, 14]] = sc.fit_transform(X_train.iloc[:, [0, 1, 14]])
    X_test.iloc[:, [0, 1, 14]] = sc.transform(X_test.iloc[:, [0, 1, 14]])
    global train, validation, test, result
    train = X_train
    validation = y_train
    test = X_test
    result = y_test
    
    st.markdown("""
    <style>
    .training {
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    #model_options = ['Logistic Regression', 'NaiveBayes',
               #'SVM','DecisionTree','RandomForest', 'XGBoost']
    if selected == 'Logistic Regression':
        Logistic()
    elif selected == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        Model = GaussianNB()
        Model.fit(X_train, y_train)
    elif selected == 'SVM':
        svm()

def Logistic():
    from sklearn.linear_model import LogisticRegression
    st.title("Logistic Regression Parameter Tuning")

    penalty_options = ['None (default)', 'l2', 'l1', 'elasticnet']
    penalty_map = {'None (default)': None, 'l2': 'l2', 'l1': 'l1', 'elasticnet': 'elasticnet'}
    penalty = penalty_map[st.selectbox('Penalty', penalty_options)]

    C = st.slider('C (Inverse of regularization strength, default=1.0)', 0.01, 10.0, 1.0)

    solver_options = ['lbfgs (default)', 'newton-cg', 'liblinear', 'sag', 'saga']
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
        model.fit(train, validation)
        global Model
        Model = model
            
    
def svm():
    from sklearn.svm import SVC
    st.title("SVC Parameter Tuning")
    
    # Define the parameter options
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = st.selectbox('Kernel', kernel_options)
    
    C = st.slider('C (Regularization parameter)', 0.01, 10.0, 1.0)
    degree = st.slider('Degree (for poly kernel)', 1, 5, 3)
    gamma_options = ['scale', 'auto']
    gamma = st.selectbox('Gamma', gamma_options)
    coef0 = ''
    if kernel in ['poly', 'sigmoid']:
        coef0 = st.slider('Coef0 (for poly/sigmoid kernels)', 0.0, 1.0, 0.0)
    else:
        coef0 = 0.0  # Default value for other kernels
    shrinking = st.checkbox('Shrinking', value=True)
    probability = st.checkbox('Probability', value=False)
    tol = st.slider('Tolerance for stopping criterion', 1e-5, 1e-1, 1e-3)
    cache_size = st.slider('Cache size (MB)', 100, 1000, 200)
    class_weight_options = ['None', 'balanced']
    class_weight = st.selectbox('Class weight', class_weight_options)
    class_weight = None if class_weight == 'None' else 'balanced'
    verbose = st.checkbox('Verbose', value=False)
    max_iter = st.slider('Max iterations', -1, 1000, -1)
    decision_function_shape_options = ['ovr', 'ovo']
    decision_function_shape = st.selectbox('Decision function shape', decision_function_shape_options)
    break_ties = st.checkbox('Break ties', value=False)
    random_state = st.slider('Random state', 0, 100, 42)
    
    svc = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state
    )   
    global Model
    Model = svc.fit(train, validation)
        

def ChooseModel():
    set_background('135D66')
    st.markdown("""
    <br><h1 style='text-align: left; color: white; font-family: Arial;'>Model for Classification</h1>
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
    space()
    test_size = create_slider('Insert Test Size', 10, 90)
    
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
               'SVM','DecisionTree','RandomForest', 'XGBoost']
    selected_models = st.selectbox('', model_options)
    training(selected_models, test_size)


def evaluation():
    train
    validation
    test
    result




def main():
    # Get the session state
    session_state = st.session_state
    
    with st.sidebar:
        # Initialize selected section if it doesn't exist in the session state
        if "selected_section" not in session_state:
            session_state.selected_section = "Home"
        
        selected = option_menu(
            "Main Menu", 
            ["Home", "Exploratory Data Analysis", "Input Data", "Choose Model", "Result and Evaluation"], 
            icons=['house', 'bar-chart', 'upload', 'robot', 'check-circle'], 
            menu_icon="cast", 
            default_index=0
        )
        
        st.write(f"Section: {selected}")
        
        # Update the selected section in the session state
        session_state.selected_section = selected

    # Get the selected section from session state
    selected = session_state.selected_section

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


def getFeatures():
    data.drop(columns=['id', 'Unnamed: 0'], inplace=True)
    data.loc[data['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    data.loc[data['satisfaction'] == 'neutral or dissatisfied', 'satisfaction'] = 0
    data['satisfaction'] = data['satisfaction'].astype(int)
    data.dropna(inplace=True)

    X = data.drop(columns=['satisfaction', 'Cleanliness',
                      'Departure Delay in Minutes', 'Inflight wifi service'])
    y = data['satisfaction']

    X = pd.get_dummies(X, drop_first=True)
    return X,y 


data = pd.read_csv('trainPlane.csv')
main()
