# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
MODELS_PATH = os.path.join(BASE_DIR, '..', 'models')

# Hyperparameters for models
LOGISTIC_REGRESSION_PARAMS = {'C': 0.1, 'solver': 'liblinear'}
COLLABORATIVE_FILTERING_PARAMS = {'n_factors': 50, 'n_epochs': 20}

