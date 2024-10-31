import pandas as pd
import os

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
browsing_history_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'browsing_history.csv')
demographic_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'demographics.csv')
purchase_history_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'purchase_history.csv')
consolidated_data_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'consolidated_data.pkl')

# Load data with error handling
try:
    browsing_data = pd.read_csv(browsing_history_path)
    demographic_data = pd.read_csv(demographic_path)
    purchase_data = pd.read_csv(purchase_history_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure that the file paths are correct and the files exist.")
    exit(1)

# Create purchase_likelihood based on whether the customer made a purchase after browsing
browsing_data['purchase_likelihood'] = browsing_data['customer_id'].isin(purchase_data['customer_id']).astype(int)

# Consolidate data
consolidated_data = browsing_data.merge(demographic_data, on='customer_id', how='left')

# Calculate total spent and purchase count for each customer
total_spent = purchase_data.groupby('customer_id')['price'].sum().reset_index()
purchase_count = purchase_data.groupby('customer_id')['transaction_id'].count().reset_index()

# Rename columns for merging
total_spent.columns = ['customer_id', 'total_spent']
purchase_count.columns = ['customer_id', 'purchase_count']

# Merge totals and counts
consolidated_data = consolidated_data.merge(total_spent, on='customer_id', how='left')
consolidated_data = consolidated_data.merge(purchase_count, on='customer_id', how='left')

# Fill NaN values
consolidated_data['total_spent'] = consolidated_data['total_spent'].fillna(0)
consolidated_data['purchase_count'] = consolidated_data['purchase_count'].fillna(0)

# Save the consolidated data
consolidated_data.to_pickle(consolidated_data_path)

print("Consolidated data created successfully!")
