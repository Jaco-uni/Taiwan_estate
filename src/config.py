import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for raw data (Excel files)
RAW_DATA_PATH = os.path.join(BASE_DIR, "../data/raw/")

# Preprocessed Data Table Name
PROCESSED_TABLE = "processed_tweets"

# Raw Data Table Name
RAW_TABLE = "raw_house"

# Predictions Table Name
PREDICTIONS_TABLE = "predictions"

# model evaluation
EVALUATION_TABLE = "grid_search_results"

# Logging Configuration
LOGGING_LEVEL = "INFO"

# Saved models
MODELS_PATH = "./models/"

