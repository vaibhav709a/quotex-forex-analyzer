import streamlit as st
import os

def get_api_key():
    # Try to get from Streamlit secrets first (for deployment)
    try:
        return st.secrets["TWELVEDATA_API_KEY"]
    except:
        # Fallback to environment variable or user input
        api_key = os.getenv("TWELVEDATA_API_KEY")
        if not api_key:
            # If no API key found, ask user to input
            api_key = st.sidebar.text_input(
                "TwelveData API Key",
                type="password",
                help="Enter your TwelveData API key"
            )
        return api_key