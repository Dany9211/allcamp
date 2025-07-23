import streamlit as st
import psycopg2
import pandas as pd

st.set_page_config(page_title="Test Allcamp", layout="wide")
st.title("Test dataset completo")

@st.cache_data
def run_query(query: str):
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

try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe totali nel dataset:** {len(df)}")
    st.dataframe(df)
except Exception as e:
    st.error(f"Errore durante la connessione: {e}")
