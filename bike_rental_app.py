import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model
class BikeRentalPredictor(nn.Module):
    def __init__(self, input_size):
        super(BikeRentalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load model and preprocessing artifacts
model = BikeRentalPredictor(input_size=18)  # Change input_size to 18 # Load the trained model with the correct input size
model.load_state_dict(torch.load('bike_rental_model.pth'))
model.eval()  # Set the model to evaluation mode

scaler = joblib.load('scaler.pkl')  # Load the saved StandardScaler
unique_categories = joblib.load('unique_categories.pkl')  # Load one-hot encoding categories
feature_columns = joblib.load('feature_columns.pkl')  # Load feature column names

# Preprocessing function for user input
def preprocess_input(data):
    # Debugging: print the original data
    st.write("Original input data:", data)

    # Cyclical encoding for Hour and Month
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # Drop original columns
    data.drop(columns=['Hour', 'Month'], inplace=True)

    # One-hot encode using unique categories
    for col in unique_categories:
        for cat in unique_categories[col]:
            data[cat] = (data[col] == cat.split('_')[-1]).astype(int)
        data.drop(columns=[col], inplace=True)

    # Ensure columns align with feature_columns
    for col in feature_columns:
        if col not in data:
            data[col] = 0  # Add missing columns
    data = data[feature_columns]  # Reorder columns to match training

    # Debugging: print processed data
    st.write("Processed data for model:", data)

    # Normalize input data
    data = scaler.transform(data)

    return data

# Streamlit UI for user input
st.title('Bike Rental Prediction App')

st.header('Enter the following details to predict bike rentals:')
hour = st.slider('Hour of the day', min_value=0, max_value=23, value=12)
month = st.selectbox('Month', list(range(1, 13)))
seasons = st.multiselect('Select Seasons', ['Spring', 'Summer', 'Autumn', 'Winter'], default=['Winter'])
holiday = st.radio('Is it a holiday?', ['Yes', 'No'])
functioning_day = st.radio('Is the day a functioning day?', ['Yes', 'No'])

# Prepare input data for prediction
if st.button('Predict'):
    predictions = []
    
    # Predict for each selected season
    for season in seasons:
        input_data = pd.DataFrame({
            'Hour': [hour],
            'Month': [month],
            'Seasons': [season],
            'Holiday': [holiday],
            'Functioning Day': [functioning_day]
        })

        # Preprocess input
        input_data_processed = preprocess_input(input_data)

        # Convert to tensor
        input_tensor = torch.tensor(input_data_processed, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # Store the prediction with its corresponding season
        predictions.append({'Season': season, 'Prediction': prediction})

    # Show the predictions
    st.write("Predictions for each season:")
    st.write(predictions)

    # Visualization: Bar chart for comparison
    st.subheader("Prediction Visualization")
    st.write("The following bar chart visualizes the predicted bike rentals for each season:")

    # Prepare data for plotting
    seasons_list = [entry['Season'] for entry in predictions]
    prediction_values = [entry['Prediction'] for entry in predictions]

    # Plotting the bar chart
    fig, ax = plt.subplots()
    ax.bar(seasons_list, prediction_values, color='blue')
    ax.set_ylabel('Number of Rentals')
    ax.set_xlabel('Season')
    ax.set_title('Bike Rental Predictions for Different Seasons')

    # Display the plot
    st.pyplot(fig)

