# src/recommend.py
import os 
import pickle
import pandas as pd
from config import MODELS_PATH, PROCESSED_DATA_PATH

def load_models():
    with open(os.path.join(MODELS_PATH, 'logistic_regression.pkl'), 'rb') as f:
        logistic_model = pickle.load(f)
    return logistic_model

def recommend_products(customer_id):
    model = load_models()
    data = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, 'consolidated_data.pkl'))
    customer_data = data[data['customer_id'] == customer_id]
    
    # Get prediction
    predictions = model.predict(customer_data.drop(columns=['customer_id', 'purchase_likelihood']))
    recommended_products = ["Product A", "Product B"] if predictions else ["Product C", "Product D"]
    
    return recommended_products
