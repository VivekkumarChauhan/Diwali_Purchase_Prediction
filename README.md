# Diwali Purchase Prediction


**BreadcrumbsDiwali Purchase Prediction** The aim of this project is to predict customer purchasing behavior during the Diwali festival, leveraging data analysis and machine learning techniques. The insights gained from the analysis can be used for targeted marketing, inventory management, and improving customer experience.
## Project Description

1. **Overview**
   - The project utilizes a dataset containing various consumer features and purchase behaviors to predict the likelihood of purchases during the Diwali festival.
   - It employs a machine learning model, specifically Logistic Regression, to provide predictions based on user input regarding customer demographics and shopping preferences.
   - Users can input various customer characteristics, and the application returns the predicted likelihood of making a purchase during Diwali.
     
2. **Technology Stack**
- ***Frontend:*** 
  - HTML (for structuring the content)
  - CSS (for styling the user interface)
  - JavaScript (for client-side interactivity)

- ***Backend:***
  - Flask (Python framework for building the web application)

- ***Data Handling:***
  - Pandas (for data manipulation and analysis)
  - NumPy (for numerical operations)

- ***Machine Learning:***
  - Scikit-learn (for implementing machine learning algorithms)
  - Pickle (for model persistence)

- ***Data Storage:***
  - CSV files (for storing raw and processed datasets)

- ***Notebooks:***
  - Jupyter Notebooks (for exploratory data analysis and model evaluation)

- ***Configuration Management:***
  - Python scripts (for configuration settings and utility functions)


## Project Structure

   ```bash
Diwali_Purchase_Prediction/
│
├── data/
│   ├── raw/
│   │   ├── demographics.csv                  # Raw customer demographics dataset
│   │   ├── purchase_history.csv              # Raw purchase history data
│   │   └── browsing_history.csv              # Raw browsing history data
│   ├── processed/
│   │   ├── cleaned_data.csv                  # Cleaned, merged, and preprocessed data
│   │   └── features.csv                      # Final features for model training
│
├── src/
│   ├── data_preprocessing.py                 # Script for data preprocessing and feature engineering
│   ├── model_training.py                     # Model training scripts for various models
│   ├── recommend.py                          # Final recommendation engine script
│   ├── utils.py                              # Helper functions used across the project
│   └── config.py                             # Configuration settings for paths, parameters, etc.
│
├── models/
│   ├── collaborative_filtering.pkl           # Trained collaborative filtering model
│   ├── logistic_regression.pkl               # Trained logistic regression model
│   └── scaler.pkl                            # Scaler for data normalization (if required)
│
├── notebooks/
│   ├── EDA.ipynb                             # Exploratory Data Analysis (EDA) notebook
│   ├── feature_engineering.ipynb             # Feature engineering notebook
│   └── model_evaluation.ipynb                # Model training and evaluation notebook
│
├── app/
│   ├── app.py                                # Flask app to serve the recommendation engine
│   └── templates/
│       └── index.html                        # Frontend UI for the recommendation system
│       └── recommendations.html              # Display recommendations for users
│
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

## Installation Guide

Follow these steps to set up the project on your local machine:

### Prerequisites

- Python 3.x installed
- Pip (Python package manager)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Diwali_Purchase_Prediction.git
   cd Diwali_Purchase_Prediction
2. **Create a Virtual Environment**
    ```bash
    python -m venv .env
3.**Activate the Virtual Environment**
On Windows:
  ```bash
   .env\Scripts\activate
 ```
On macOS/Linux:
  ```bash
    python -m venv .env
 ```
4.**Install Required Packages**
```bash
    pip install flask pandas scikit-learn
```
  OR
```bash
  pip install -r requirements.txt
```
5.**Run the Application**
```bash
    python app/app.py
```
The application will start on http://127.0.0.1:5000/.


6.**Access the Web Interface Open your web browser and navigate to http://127.0.0.1:5000/. You can now input flower measurements and classify them using the KNN or Logistic Regression models.**

## Usage
1.**Enter the area, number of bedrooms, bathrooms, floors, and age of the house in the input fields.**
2.**Click on "Predict Price" to get the estimated price of the house.**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contributing

We welcome contributions to enhance **Diwali Purchase Prediction**! If you're interested in contributing, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:
   ```bash
   git checkout -b feature-branch
3. **Commit your changes**:
   ```bash
   git commit -m 'Add feature'
   ```
4. **Push to the branch**:
   ```bash
    git push origin feature-branch
   ```
5. **Open a Pull Request**.


