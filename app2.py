import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('credit_fraud.pkl')

# Set up the Streamlit app title and description
st.title("Credit Card Fraud Detection App")
st.write("""
This application uses a machine learning model to predict whether a credit card transaction is fraudulent based on features such as amount, transaction type, and balances.
""")

# Add fields for user input based on the model's expected features
step = st.number_input("Transaction Step (hours)", min_value=0, step=1)
types = st.number_input("Transaction Type (encoded as an integer)", min_value=0, step=1)
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceorig = st.number_input("Old Balance (origin)", min_value=0.0)
newbalanceorig = st.number_input("New Balance (origin)", min_value=0.0)
oldbalancedest = st.number_input("Old Balance (destination)", min_value=0.0)
newbalancedest = st.number_input("New Balance (destination)", min_value=0.0)
isflaggedfraud = st.number_input("Is Flagged as Fraud (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1)

# Prediction function
def predict(step, types, amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud):
    # Arrange the input data for the model
    features = np.array([[step, types, amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud]])
    
    # Make prediction
    prediction = model.predict(features)
    return prediction

# Button to trigger prediction
if st.button("Predict"):
    # Call the prediction function
    prediction = predict(step, types, amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud)
    
    # Display result
    if prediction == 1:
        st.write("The transaction is **fraudulent**.")
    else:
        st.write("The transaction is **not fraudulent**.")
