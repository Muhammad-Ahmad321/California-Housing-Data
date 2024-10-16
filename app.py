import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import sklearn.datasets

# Load dataset
house_price_dataset = sklearn.datasets.fetch_california_housing()
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_price_dataframe['price'] = house_price_dataset.target

# Split into features and target
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

# Split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Streamlit app interface
st.title("California Housing Price Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")
MedInc = st.sidebar.slider('Median Income', float(X['MedInc'].min()), float(X['MedInc'].max()), float(X['MedInc'].mean()))
HouseAge = st.sidebar.slider('House Age', float(X['HouseAge'].min()), float(X['HouseAge'].max()), float(X['HouseAge'].mean()))
AveRooms = st.sidebar.slider('Average Rooms', float(X['AveRooms'].min()), float(X['AveRooms'].max()), float(X['AveRooms'].mean()))
AveBedrms = st.sidebar.slider('Average Bedrooms', float(X['AveBedrms'].min()), float(X['AveBedrms'].max()), float(X['AveBedrms'].mean()))
Population = st.sidebar.slider('Population', float(X['Population'].min()), float(X['Population'].max()), float(X['Population'].mean()))
AveOccup = st.sidebar.slider('Average Occupants', float(X['AveOccup'].min()), float(X['AveOccup'].max()), float(X['AveOccup'].mean()))
Latitude = st.sidebar.slider('Latitude', float(X['Latitude'].min()), float(X['Latitude'].max()), float(X['Latitude'].mean()))
Longitude = st.sidebar.slider('Longitude', float(X['Longitude'].min()), float(X['Longitude'].max()), float(X['Longitude'].mean()))

# Input DataFrame
input_data = pd.DataFrame({
    'MedInc': [MedInc],
    'HouseAge': [HouseAge],
    'AveRooms': [AveRooms],
    'AveBedrms': [AveBedrms],
    'Population': [Population],
    'AveOccup': [AveOccup],
    'Latitude': [Latitude],
    'Longitude': [Longitude]
})

st.write("Input Features:")
st.write(input_data)

# Prediction
prediction = model.predict(input_data)
st.write(f"Predicted Median House Value: ${prediction[0] * 100000:.2f}")

# Visualization
if st.checkbox("Show Actual vs Predicted for Training Data"):
    training_data_prediction = model.predict(X_train)
    fig, ax = plt.subplots()
    ax.scatter(Y_train, training_data_prediction)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual Price vs Predicted Price (Training Data)")
    st.pyplot(fig)
