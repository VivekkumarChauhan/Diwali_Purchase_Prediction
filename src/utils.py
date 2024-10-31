# src/utils.py

import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
# src/utils.py

def recommend_products(customer_id):
    # Your recommendation logic here
    return {"product_id": "12345", "product_name": "Sample Product"}
