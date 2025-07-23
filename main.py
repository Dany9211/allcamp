import streamlit as st
import psycopg2
import pandas as pd

# --- Titolo app ---
st.set_page_config(page_title="Allcamp Viewer", layout="wide")
st.title("Visualizzazione Tabella allcamp da Supabase")

# --- Funzione di connessione a Supabase ---
@st.cache_data
def run_query(query: str):
    """Esegue una query su Supabase e restituisce un DataFrame."""
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        sslmode="require"
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- Carica dati dalla tabella allcamp ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe totali nel dataset:** {len(df)}")
    st.dataframe(df)
except Exception as e:
    st.error(f"Errore durante la connessione o il caricamento: {e}")
