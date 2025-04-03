<<<<<<< HEAD
### Sviluppo di un Modello di Regressione e Web App per la Predizione dei Prezzi Immobiliari
Obiettivo:
Sviluppare un modello di regressione per predire il prezzo al metro quadro di immobili nella regione di Sindian, Nuova Taipei, Taiwan, utilizzando il Real Estate Valuation Data Set. Successivamente, creare una web app con Streamlit che permetta agli utenti di ottenere una stima del prezzo inserendo:
• Latitudine 
• longitudine
• Età dell’immobile 
• distanza dalla stazione MRT più vicina 
• numero di minimarket nelle vicinanze.

Passaggi eseguiti:
1. Sviluppo del Modello:
• Costruito tre modelli di regressione per predire il prezzo al metro quadro basato su latitudine e longitudine.
 1. modello, consiste in un modello completo di tutti i parametri
 2. modello, consiste in un modello che sfrutta solo latitudine e longitudine
 3. modello, consiste in un modello che sfrutta • Età dell’immobile, distanza dalla stazione MRT più vicina, numero di minimarket nelle vicinanze.

2. Per eseguire l'applicazione:
Per usufruire del servizio è necessario eseguire il file 'run_pipeline.py', per poi eseguire dal terminale il file UI.py utilizzando il comando: 'python -m streamlit run UI.py'

Una volta aperta l'applicazione via web si può scegliere quale modello utilizzare e modificare o manualmente da tastiera o tramite gli appositi tasti i valori dei parametri. 


3. Visualizzazione dei Dati:
Per la visualizzazione dati è stata creata una mappa interattiva dei prezzi originali utilizzando Tableau.
La mappa rappresenta la distribuzione delle case, in relazione alla dimensione che rappresentà l'età dell'immobile e il colore la vicinanza alla stazione MRT più vicina. Nel grafico affianco si è condotta un'analisi più concentrata sulla distribuzione dei punti in funzione dei parametri età della casa e costo della casa.
Per visulizzare i grafici utilizzare il seguente 
[link](https://public.tableau.com/shared/7BKQJHYQ5?:display_count=n&:origin=viz_share_link)
