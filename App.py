import streamlit as st
import joblib
import pandas as pd

# Load the trained model
lr = joblib.load('delivery_delay.sav')

# Define the feature names in the correct order
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction App')
st.write('Enter the feature values to predict delivery delay.')

# Create input fields for each feature
input_features = {}
for feature in feature_names:
    input_features[feature] = st.number_input(f'Enter {feature}:', value=0.0, step=0.1)

if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_features])

    # Make prediction
    prediction = lr.predict(input_df)

    # Display the prediction
    if prediction[0] == 1:
        st.success('Prediction: Delivery Delay is likely (1)')
    else:
        st.success('Prediction: Delivery Delay is not likely (0)')
