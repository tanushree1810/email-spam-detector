import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

# Streamlit page configuration
st.set_page_config(page_title="Spam Detection App", page_icon="📧", layout="centered")

# Add logo (make sure to put your logo in the same directory or update the path)
try:
    logo = Image.open("logo.png")  # Update path if needed
    st.image(logo, use_column_width=True, width=50)  # Adjust the size of the logo
except FileNotFoundError:
    st.warning("Logo not found. Please check the file path.")

# Title and description
st.title("Spam Detection App")
st.write("**Enter a message below to check if it's spam or not.**")

# User input
user_input = st.text_area("Message:")
if st.button("Predict"):
    if user_input.strip():
        # Transform the input using the saved vectorizer
        vect_input = vectorizer.transform([user_input])
        
        # Predict using the loaded model
        prediction = model.predict(vect_input)
        
        # Display result
        result = "Spam" if prediction[0] == 1 else "Ham"
        st.success(f"The message is: **{result}**")
    else:
        st.warning("Please enter a valid message.")
