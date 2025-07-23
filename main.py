import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Allcamp Viewer", layout="wide")
st.title("Analisi Completa Tabella allcamp")

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
except Exception as e:
    st.error(f"Errore durante la connessione o il caricamento: {e}")
    st.stop()

# --- Aggiungi colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

st.subheader("Anteprima Dati")
st.dataframe(df.head(50))  # Mostra solo le prime 50 righe per non appesantire l'app

# --- FUNZIONE DISTRIBUZIONE RISULTATI ---
def mostra_distribuzione(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]

    def classifica_risultato(ris):
        home, away = map(int, ris.split("-"))
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df[f"{col_risultato}_class"] = df[col_risultato].apply(classifica_risultato)
    distribuzione = df[f"{col_risultato}_class"].value_counts().reset_index()
    distribuzione.columns = ["Risultato", "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(
        lambda x: round(100/x, 2) if x > 0 else "-"
    )

    st.subheader(f"Distribuzione {titolo}")
    st.table(distribuzione)

    # Winrate 1X2
    count_1 = distribuzione[distribuzione["Risultato"].str.contains("casa vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["1-0","2-0","2-1","3-0","3-1","3-2"])].Conteggio.sum()
    count_2 = distribuzione[distribuzione["Risultato"].str.contains("ospite vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-1","0-2","0-3","1-2","1-3","2-3"])].Conteggio.sum()
    count_x = distribuzione[distribuzione["Risultato"].str.contains("pareggio")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-0","1-1","2-2","3-3"])].Conteggio.sum()

    totale = len(df)
    winrate = [round((count_1/totale)*100,2), round((count_x/totale)*100,2), round((count_2/totale)*100,2)]
    st.subheader(f"WinRate 1-X-2 ({titolo})")
    st.table(pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
        "Conteggio": [count_1, count_x, count_2],
        "WinRate %": winrate,
        "Odd Minima": [round(100/w,2) if w > 0 else "-" for w in winrate]
    }))

# --- Mostra distribuzione risultati ---
if not df.empty:
    if "risultato_ft" in df.columns:
        mostra_distribuzione(df, "risultato_ft", "Risultati Finali (FT)")
    if "risultato_ht" in df.columns:
        mostra_distribuzione(df, "risultato_ht", "Risultati Primo Tempo (HT)")

# --- CALCOLO BTTS e OVER ---
if not df.empty and "risultato_ft" in df.columns:
    temp_ft = df["risultato_ft"].str.split("-", expand=True)
    temp_ft = temp_ft.replace("", np.nan).dropna()
    temp_ft = temp_ft.astype(int)

    df["home_g_ft"] = temp_ft[0]
    df["away_g_ft"] = temp_ft[1]
    df["tot_goals_ft"] = df["home_g_ft"] + df["away_g_ft"]

    total_games = len(df)

    # BTTS
    btts_count = len(df[(df["home_g_ft"] > 0) & (df["away_g_ft"] > 0)])
    perc_btts = round(btts_count / total_games * 100, 2) if total_games > 0 else 0
    odd_btts = round(100 / perc_btts, 2) if perc_btts > 0 else "-"
    st.subheader("BTTS (Both Teams To Score)")
    st.write(f"Partite BTTS SI: {btts_count}")
    st.write(f"Percentuale BTTS SI: {perc_btts}%")
    st.write(f"Odd Minima: {odd_btts}")

    # Over FT
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    over_data = []
    for t in thresholds:
        count_over = (df["tot_goals_ft"] > t).sum()
        perc_over = round(count_over / total_games * 100, 2) if total_games > 0 else 0
        odd_min = round(100 / perc_over, 2) if perc_over > 0 else "-"
        over_data.append([f"Over FT {t}", count_over, perc_over, odd_min])
    over_df = pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    st.subheader("Over Goals (FT)")
    st.table(over_df)
