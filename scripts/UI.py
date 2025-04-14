import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the src directory to sys.path
from src import config
from src.load_data import load_data
import streamlit as st
import pickle

# Load the models 
with open(os.path.join(config.MODELS_PATH, "rf_completo.pickle"), "rb") as file:
    modelRF_C = pickle.load(file)

with open(f"{config.MODELS_PATH}rf_lat.pickle", "rb") as f:
        model_rf_lat = pickle.load(f)

with open(f"{config.MODELS_PATH}rf_RTM.pickle", "rb") as f:
        model_rf_RTM = pickle.load(f)

# Creo l'interfaccia utente
st.set_page_config(
    page_title="Calcolo prezzo medio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Calcolo prezzo medio per metro quadro per le case di Taipei")
st.write("In questo spazio puoi calcolare il prezzo medio per metro quadro di un immobile a Taipei, in base a diversi parametri da te inseriti.")


selectedModel = st.selectbox(
         'Quale modello vorresti utilizzare per la sentiment analysis?',
         ('tutti i parametri', 'Latitudine e longitudine', 'Età dell’immobile, distanza dalla stazione MRT più vicina e numero di minimarket nelle vicinanze'))


st.title("Inserisci i parametri richiesti")
if selectedModel == "tutti i parametri":
    latitude = st.number_input("Latitudine", min_value=float(24.932070), max_value=float(25.014590))
    longitude = st.number_input("Longitudine", min_value=float(121.473530), max_value=float(121.566270))
    age = st.number_input("Età dell'immobile", min_value=int(0), max_value=int(43.800000))
    distance = st.number_input(f"Distanza dalla stazione MRT più vicina tra", min_value=float(23.382840), max_value=float(6488.021000))
    num_minimarkets = st.number_input("Numero di minimarket nelle vicinanze", min_value=int(0), max_value=int(10))

    # Bottone per predire
    if st.button("Calcola Prezzo"):
        x_input = np.array([[latitude, longitude, age, distance, num_minimarkets]])
        price_estimation = round(modelRF_C.predict(x_input)[0],2)
        st.write(f'La casa in questione viene a costare',price_estimation,'dollari taiwanesi')

elif selectedModel == "Latitudine e longitudine":
    latitude = st.number_input("Latitudine", min_value=float(24.932070), max_value=float(25.014590))
    longitude = st.number_input("Longitudine", min_value=float(121.473530), max_value=float(121.566270))
    
    # Bottone per predire
    if st.button("Calcola Prezzo"):
        x_input = np.array([[latitude, longitude]])
        price_estimation = round(model_rf_lat.predict(x_input)[0],2)
        st.write(f'La casa in questione viene a costare',price_estimation,'dollari taiwanesi')

elif selectedModel == "Età dell’immobile, distanza dalla stazione MRT più vicina e numero di minimarket nelle vicinanze":
    age = st.number_input("Età dell'immobile", min_value=int(0), max_value=int(43.800000))
    distance = st.number_input(f"Distanza dalla stazione MRT più vicina tra", min_value=float(23.382840), max_value=float(6488.021000))
    num_minimarkets = st.number_input("Numero di minimarket nelle vicinanze", min_value=int(0), max_value=int(10))

    # Bottone per predire
    if st.button("Calcola Prezzo"):
        x_input = np.array([[age, distance, num_minimarkets]])
        price_estimation = round(model_rf_RTM.predict(x_input)[0],2)          
        st.write(f'La casa in questione viene a costare',price_estimation,'dollari taiwanesi')


