App: https://california-housing-price.streamlit.app/


California Housing Price Prediction:
This project involves analyzing the California Housing dataset and building a predictive model to estimate house prices based on various features. The dataset is fetched using the sklearn.datasets.fetch_california_housing() function from the Scikit-learn library. The data is then processed and visualized to understand the relationships between the features and the target variable (house prices).

Dataset
The dataset consists of housing data for various districts in California. The target variable is the median house value for the district, and the features describe aspects of the district’s population and housing stock.

Features
MedInc: Median income in block group
HouseAge: Median house age in block group
AveRooms: Average number of rooms per household
AveBedrms: Average number of bedrooms per household
Population: Block group population
AveOccup: Average number of household members
Latitude: Block group latitude
Longitude: Block group longitude
Target
Price: The median house value for California districts.
Project Structure
house_price_dataset = sklearn.datasets.fetch_california_housing(): This fetches the California Housing dataset.
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names): The dataset is loaded into a pandas DataFrame with appropriate column names for ease of analysis.
house_price_dataframe['price'] = house_price_dataset.target: The target variable price is added to the DataFrame.
Basic Dataset Exploration:
Shape of Data: Checking the number of rows and columns in the DataFrame.
Missing Values: Checking for any missing values in the dataset.
Statistical Summary: Generating summary statistics for the dataset to understand the distribution of features.
Code Example
Here’s a snippet of the code for loading and exploring the dataset:

python
Copy code
import numpy as np
import pandas as pd
import sklearn.datasets

# Load the California Housing dataset
house_price_dataset = sklearn.datasets.fetch_california_housing()

# Load the dataset into a pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# Add the target variable to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target

# Display the first few rows of the DataFrame
print(house_price_dataframe.head())

# Checking the number of rows and columns
print("Shape of the dataset:", house_price_dataframe.shape)

# Check for missing values
print("Missing values in the dataset:", house_price_dataframe.isnull().sum())

# Statistical summary of the dataset
print("Statistical summary of the dataset:\n", house_price_dataframe.describe())
Getting Started
Prerequisites
Ensure you have the following libraries installed:

Python 3.x
NumPy
Pandas
Scikit-learn
You can install them using the following command:

bash
Copy code
pip install numpy pandas scikit-learn
Running the Project
Clone the repository:
bash
Navigate to the project directory:
bash
Copy code
cd california-housing-price-prediction
Run the Python script:
bash
Copy code
python your_script.py
License
This project is licensed under the MIT License.

