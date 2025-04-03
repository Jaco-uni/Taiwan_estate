#FASE DI IMPORTO LIBRERIE
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys
import pickle # Adds the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the src directory to sys.path
from src import config
import logging


#FUNZIONE CARICAMENTO DATI
def load_data(): # crea un dataframe partendo da sqlLite
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"""SELECT * FROM {config.PROCESSED_TABLE}"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

#FUNZIONE PER CREARE IL MODELLO RANDOM FOREST BASATO SU TUTTE LE COVARIATE
def train_model_completo(grid_search=False):
    """Trains a Random Forest model with GridSearchCV and saves evaluation metrics to CSV."""
    #df = load_data().head(100) # prendiamo solo un sottoinsieme del dataset (di 100 righe). Sarebbe meglio fare sample
    #df = load_data().sample(1000) # facciamolo solo su 1000
    df = load_data() # carica tutto il dataset
    # Save original indices before vectorization
    
    df_indices = df.index

    # Feature extraction
    print(df.columns)
    X = df[['house_age', 'distance_MRT', 'num_convenience_stores', 'latitude',
       'longitude']]
    y = df['price_per_unit_area']

    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    if grid_search:
        print("Inizio il tuning del modello Random Forest")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print("Modello Random Forest salvato con successo.")

    else:
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("Modello Random Forest salvato con successo.")

    # salviamo il modello
    with open(os.path.join(config.MODELS_PATH, "rf_completo.pickle"), "wb") as file:
        pickle.dump(rf, file)
    print("Modello Random Forest salvato con successo.")

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['prediction'] = y_pred  # Add predictions


    # Compute metrics
    metrics = {"MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred), 
    }

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # saving grid search results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn,
                      if_exists='replace', index=False)
    # Commit and close the connection
    conn.commit()
    conn.close()

#FUNZIONE PER CREARE IL MODELLO RANDOM FOREST BASATO SU LATITUDE E LONGITUDE
def train_model_lat_lon(grid_search=False):
    """Trains a Random Forest model with GridSearchCV and saves evaluation metrics to CSV."""
    #df = load_data().sample(1000) # facciamolo solo su 1000
    df = load_data() # carica tutto il dataset
    # Save original indices before vectorization
    df_indices = df.index

    # Feature extraction
    X = df[['latitude', 'longitude']]# usa TF-IDF per trasformare le parole in vettori
    y = df['price_per_unit_area']


    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    if grid_search:
        print("Inizio il tuning del modello Random Forest")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print("Modello Random Forest salvato con successo.")

    else:
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("Modello Random Forest salvato con successo.")

    # salviamo il modello
    with open(os.path.join(config.MODELS_PATH, "rf_lat.pickle"), "wb") as file:
        pickle.dump(rf, file)
    print("Modello Random Forest salvato con successo.")

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['prediction'] = y_pred  # Add predictions


    # Compute metrics
    metrics = {"MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred), 
    }

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # saving grid search results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn,
                      if_exists='replace', index=False)
    # Commit and close the connection
    conn.commit()
    conn.close()

#FUNZIONE PER CREARE IL MODELLO RANDOM FOREST BASATO SU HOUSE AGE, DISTANCE TO MRT E NUMBER OF CONVENIENCE STORES
def train_model_age_mrt(grid_search=False):
    """Trains a Random Forest model with GridSearchCV and saves evaluation metrics to CSV."""
    #df = load_data().head(100) # prendiamo solo un sottoinsieme del dataset (di 100 righe). Sarebbe meglio fare sample
    #df = load_data().sample(1000) # facciamolo solo su 1000
    df = load_data() # carica tutto il dataset
    # Save original indices before vectorization
    df_indices = df.index

    # Feature extraction
    X = df[['house_age', 'distance_MRT', 'num_convenience_stores']]# usa TF-IDF per trasformare le parole in vettori
    y = df['price_per_unit_area']


    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    if grid_search:
        print("Inizio il tuning del modello Random Forest")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print("Modello Random Forest salvato con successo.")

    else:
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("Modello Random Forest salvato con successo.")

    # salviamo il modello
    with open(os.path.join(config.MODELS_PATH, "rf_RTM.pickle"), "wb") as file:
        pickle.dump(rf, file)
    print("Modello Random Forest salvato con successo.")

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['prediction'] = y_pred  # Add predictions


    # Compute metrics
    metrics = {"MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),   
    }

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # saving grid search results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn,
                      if_exists='replace', index=False)
    # Commit and close the connection
    conn.commit()
    conn.close()