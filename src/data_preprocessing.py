# src/data_preprocessing.py

import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import os

def load_data():
    demographics = pd.read_csv(os.path.join('data/raw/', 'demographics.csv'))
    purchase_history = pd.read_csv(os.path.join('data/raw/', 'purchase_history.csv'))
    browsing_history = pd.read_csv(os.path.join('data/raw/', 'browsing_history.csv'))
    return demographics, purchase_history, browsing_history

def preprocess_data():
    demographics, purchase_history, browsing_history = load_data()
    
    # Basic preprocessing
    purchase_history['purchase_date'] = pd.to_datetime(purchase_history['purchase_date'])
    browsing_history['browsing_date'] = pd.to_datetime(browsing_history['browsing_date'])
    
    # Feature engineering (example)
    # Calculate total purchases and spending per user
    purchase_summary = purchase_history.groupby('customer_id').agg(
        total_spent=pd.NamedAgg(column='price', aggfunc='sum'),
        purchase_count=pd.NamedAgg(column='product_id', aggfunc='count')
    ).reset_index()
    
    # Merge datasets
    consolidated_data = demographics.merge(purchase_summary, on='customer_id', how='left')
    consolidated_data.fillna(0, inplace=True)  # Fill missing values
    
    # Save processed data
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    consolidated_data.to_pickle(os.path.join(PROCESSED_DATA_PATH, 'consolidated_data.pkl'))
    print("Data preprocessed and saved!")

if __name__ == '__main__':
    preprocess_data()
