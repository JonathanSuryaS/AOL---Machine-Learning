import streamlit as st
from hydralit_components import hydralit_navbar

app = hydralit_navbar(
    header="My Streamlit App",  # Customize the header text
    links=[
        ("Home", "#"),  # Link text and corresponding URL (or anchor for internal navigation)
        ("About", "#about"),
        ("Contact", "#contact"),
        # Add more links as needed
    ],
)

st.write(app)