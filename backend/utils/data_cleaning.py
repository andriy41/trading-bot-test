# utils/data_cleaning.py
# backend/utils/data_cleaning.py

import pandas as pd

def clean_data(data):
    """
    Clean and preprocess the data.
    - data: DataFrame with price and volume columns
    """
    # Remove any rows with missing values
    data.dropna(inplace=True)

    # Convert timestamps if necessary
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    return data
