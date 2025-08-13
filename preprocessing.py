# preprocessing.py

import pandas as pd
import joblib

def preprocess(df):
    df_original = df.copy()
    # Date format format transformation
    df_original['Processed Date'] =   pd.to_datetime(df_original['Processed Date'])
    # Extract day of the week as an integer (0-6)
    df_original['day_of_week_numeric'] = df_original['Processed Date'].dt.dayofweek
    # Extract month as an integer (1-12)
    df_original['month_numeric'] = df_original['Processed Date'].dt.month
    df_preprocessed = df_original.drop(columns=['Processed Date', 'Account Name', 'Check Number'])
    return df_preprocessed


