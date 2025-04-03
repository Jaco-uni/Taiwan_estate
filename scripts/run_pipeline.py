import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the src directory to sys.path

import logging
from src.load_data import load_data # STO RICHIAMANDO IL FILE LOAD DATA IN SRC E CHIAMO LA FUNZIONE load_data
from src.make_model import train_model_completo, train_model_lat_lon, train_model_age_mrt 


# Set up logging
logging.basicConfig(filename='./logs/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting RF regression Pipeline...")

    # Step 1: Load data from Excel and store it in SQLite
    logging.info("Loading raw data...")
    load_data()

    # Step 2: Preprocess text data
    logging.info("Preprocessing data is useless...")

    # Step 3: Train sentiment analysis model
    logging.info("Training the model...")
    train_model_completo()
    train_model_lat_lon()
    train_model_age_mrt()

if __name__ == "__main__":
    main()