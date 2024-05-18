# Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
#simport pandas_profiling as pp
import pickle


def EDA(data):
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


def prediction(data):
    set_background('f1e0C5')
    with open('random_forest.pkl', 'rb') as file:
        model = pickle.load(file)
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
    CustomerGender = create_radio_input(
        'Insert Customer Gender', ["Male", "Female"])

    # Insert Customer Type
    CustomerType = create_radio_input(
        'Insert Customer Type', ["Loyal", "Disloyal"])

    # Age
    age = create_number_input("Enter your age:", min_value=0, max_value=100, value=25, step=1)

    # Insert Type of Travel
    TypeTravel = create_radio_input('Insert Type of Travel', [
                                    'Personal Travel', 'Business travel'])

    # Insert Class of Airlines
    Class = create_radio_input('Insert Travel Class', [
                               'Business', 'Eco Plus', 'Eco'])
    
    #Inser Flight Distance
    Distance = create_number_input("Flight Distance", min_value=30, max_value=5000, value=30, step=1)
    
    #Airlines Service
    inflight_wifi = create_slider('Inflight wifi service', 0, 5)
    departure_arrival_time = create_slider('Departure/Arrival time convenient', 0, 5)
    online_booking = create_slider('Ease of Online booking', 0, 5)
    gate_location = create_slider('Gate location', 0, 5)
    food_drink = create_slider('Food and drink', 0, 5)
    online_boarding = create_slider('Online boarding', 0, 5)
    seat_comfort = create_slider('Seat comfort', 0, 5)
    inflight_entertainment = create_slider('Inflight entertainment', 0, 5)
    onboard_service = create_slider('On-board service', 0, 5)
    legroom_service = create_slider('Leg room service', 0, 5)
    baggage_handling = create_slider('Baggage handling', 0, 5)
    checkin_service = create_slider('Checkin service', 0, 5)
    inflight_service = create_slider('Inflight service', 0, 5)
    cleanliness = create_slider('Cleanliness', 0, 5)
    
    #Departure & Arrival Late
    Departure = create_number_input("Departure Delay", min_value=0, max_value=1600, value=10, step=1)
    Arrival = create_number_input("Arrival Delay", min_value=0, max_value=1600, value=10, step=1)

    
    #Inputted data
    input_data = {
        'Gender': [CustomerGender],
        'Customer Type': [CustomerType],
        'Age': [age],
        'Type of Travel': [TypeTravel],
        'Class': [Class],
        'Flight Distance': [Distance],
        'Inflight wifi service': [inflight_wifi],
        'Departure/Arrival time convenient': [departure_arrival_time],
        'Ease of Online booking': [online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [inflight_entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [legroom_service],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [Departure],
        'Arrival Delay in Minutes': [Arrival]
    }

    df = pd.DataFrame(input_data)

    # Convert categorical variables to dummy/one-hot encoded variables
    df = pd.get_dummies(df, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

    # Define the desired columns order
    desired_columns = [
        'Age', 'Flight Distance', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
        'Arrival Delay in Minutes', 'Gender_Male', 'Customer Type_Disloyal', 'Type of Travel_Personal Travel',
        'Class_Eco', 'Class_Eco Plus'
    ]

    # Ensure all desired columns are present in the DataFrame, adding missing columns with default values
    for col in desired_columns:
        if col not in df.columns:
            df[col] = 0
    
    space()
    st.markdown("""
    <br><h3 style='text-align: left; color: white; font-family: Arial;'>Your Data</h3>
    """, unsafe_allow_html=True)
    input_data = pd.DataFrame(input_data)
    st.write(input_data)
    
    new_row = pd.DataFrame(input_data, index=[len(data)])
    new_data = pd.concat([data, new_row], ignore_index=True)
    ##Modeling
    ###Dropping undeeded columns
    new_data = data.drop(columns=['Unnamed: 0', 'id', 'Cleanliness', 'Departure Delay in Minutes', 'Inflight wifi service', 'satisfaction'])

    ###Encoding
    new_data = pd.get_dummies(new_data, drop_first=True)
    
    ###Processing
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    new_data.iloc[:, [0, 1, 14]] = sc.fit_transform(new_data.iloc[:, [0, 1, 14]])
    
    last_row = new_data.iloc[[-1]]
    result = model.predict(last_row)
    
    if(result == 1):
      set_background('77DD77')
      st.markdown("""
    <br><h3 style='text-align: center; color: white; font-family: Arial;'>Customer is Satisfied</h3>
    """, unsafe_allow_html=True)
    elif (result == 0):
      set_background('D2372F')
      st.markdown("""
    <br><h3 style='text-align: center; color: white; font-family: Arial;'>Customer is not Satisfied</h3>
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


def space():
    st.markdown("""
              <br><br>""", unsafe_allow_html=True)

# Set the background color


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
main()
