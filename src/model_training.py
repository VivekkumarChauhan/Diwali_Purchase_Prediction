import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from config import PROCESSED_DATA_PATH, MODELS_PATH, LOGISTIC_REGRESSION_PARAMS
from imblearn.over_sampling import SMOTE
import numpy as np

# Define the number of samples you want to generate for the minority class
num_samples_to_generate = 50  # Change this to your requirement

# Create synthetic data
def create_synthetic_data(num_samples):
    synthetic_data = {
        'customer_id': np.random.randint(1000, 2000, num_samples),
        'product_id': np.random.randint(1, 100, num_samples),
        'product_category': np.random.choice(['A', 'B', 'C'], num_samples),
        'browsing_date': pd.date_range(start='2024-10-01', periods=num_samples).tolist(),
        'time_spent': np.random.randint(1, 300, num_samples),
        'viewed': np.random.randint(1, 10, num_samples),
        'purchase_likelihood': np.ones(num_samples),  # This is the minority class
        'age': np.random.randint(18, 65, num_samples),
        'gender': np.random.choice(['M', 'F'], num_samples),
        'city': np.random.choice(['City1', 'City2'], num_samples),
        'state': np.random.choice(['State1', 'State2'], num_samples),
        'marital_status': np.random.choice(['Single', 'Married'], num_samples),
        'income_range': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'total_spent': np.random.uniform(50, 500, num_samples),
        'purchase_count': np.random.randint(1, 5, num_samples),
    }

    return pd.DataFrame(synthetic_data)

def train_logistic_regression():
    # Load the processed data
    data_path = os.path.join(PROCESSED_DATA_PATH, 'consolidated_data.pkl')
    
    # Check if the processed data file exists
    if not os.path.exists(data_path):
        print(f"Error: Processed data file not found at {data_path}")
        return
    
    data = pd.read_pickle(data_path)

    # Create synthetic data
    synthetic_df = create_synthetic_data(num_samples_to_generate)

    # Combine original and synthetic data
    combined_data = pd.concat([data, synthetic_df], ignore_index=True)

    # Save the new dataset for further processing
    combined_data.to_pickle("path_to_your_combined_data.pkl")

    # Check unique values in 'purchase_likelihood'
    print("Unique values in 'purchase_likelihood':", combined_data['purchase_likelihood'].unique())
    print("Count of each class in 'purchase_likelihood':")
    print(combined_data['purchase_likelihood'].value_counts())
    
    # Print the columns to debug
    print("Columns in the dataset:", combined_data.columns.tolist())
    
    # Check if 'purchase_likelihood' exists
    if 'purchase_likelihood' not in combined_data.columns:
        print("Warning: 'purchase_likelihood' column not found in data!")
        print("Exiting training due to missing target variable.")
        return

    # Define features and target variable
    try:
        X = combined_data.drop(columns=['customer_id', 'purchase_likelihood'])
        y = combined_data['purchase_likelihood']
    except KeyError as e:
        print(f"Error during feature/target selection: {e}")
        return

    # Check the unique values in y again
    print("Unique values in target variable after selection:", y.unique())
    print("Count of each class in target variable after selection:")
    print(y.value_counts())

    # Check for NaN values
    if X.isnull().values.any():
        print("NaN values found in features. Filling NaNs with 0...")
        X.fillna(0, inplace=True)  # Replace NaN values

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Check class distribution before SMOTE
    print("Class distribution in target variable before SMOTE:")
    print(y.value_counts())

    # Apply SMOTE for balancing the classes
    smote = SMOTE(random_state=42)

    # Check if there is more than one class
    if y.nunique() <= 1:
        print("Error: Only one class present in target variable. Cannot apply SMOTE.")
        return

    # Apply SMOTE to generate synthetic samples
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Check class distribution after SMOTE
    print("Class distribution in target variable after SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    
    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # Create and train the Logistic Regression model
    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    # Fit the model
    model.fit(X_train, y_train)
    
    # Save the trained model
    os.makedirs(MODELS_PATH, exist_ok=True)
    model_file_path = os.path.join(MODELS_PATH, 'logistic_regression.pkl')
    
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Logistic Regression model trained and saved at {model_file_path}!")

if __name__ == '__main__':
    train_logistic_regression()
