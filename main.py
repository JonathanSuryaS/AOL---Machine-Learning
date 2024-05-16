# Libraries
import streamlit as st


# Main Section
st.title('Airline Satisfaction Prediction')
# Short Description for Apps



# Navigations
st.sidebar.header("Program Menu")
### Homepage
if st.sidebar.button('Homepage'):
   st.title('Homepage')


if st.sidebar.button("Exploratory Data Analysis"):
  # Display dashboard content
  st.markdown('''
  # **Exploratory Data Analysis**   

    ''')



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
set_background("f9bfbf")