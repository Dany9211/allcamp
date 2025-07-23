import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi allcamp", layout="wide")
st.title("Analisi Completa (FT, HT, BTTS, Over)")

# --- Funzione di connessione a Supabase ---
@st.cache_data
def run_query(query):
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

# --- Carica dataset dalla tabella allcamp ---
df = run_query('SELECT * FROM "allcamp";')
st.write(f"**Righe totali nel dataset:** {len(df)}")

# --- Aggiungi colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

# --- FILTRI ---
filters = {}

# --- Filtri per odds (range digitabile) ---
st.sidebar.header("Filtri Quote")
for odd_col in ["odd_home", "odd_away", "odd_draw"]:
    if odd_col in df.columns:
        min_val = float(pd.to_numeric(df[odd_col], errors="coerce").min())
        max_val = float(pd.to_numeric(df[odd_col], errors="coerce").max())
        min_input = st.sidebar.text_input(f"Min {odd_col}", value=str(round(min_val, 2)))
        max_input = st.sidebar.text_input(f"Max {odd_col}", value=str(round(max_val, 2)))
        try:
            min_val_user = float(min_input)
            max_val_user = float(max_input)
            filters[odd_col] = (min_val_user, max_val_user)
        except:
            st.sidebar.warning(f"Valori non validi per {odd_col}")

# --- Filtri timing gol ---
timing_options = [
    "Tutti", "0-5", "6-10", "0-10", "11-20", "21-30", "31-39",
    "40-45", "46-55", "56-65", "66-75", "76-85", "76-90", "85-90"
]
st.sidebar.header("Timing Gol")
for col in ["primo_gol_home", "secondo_gol_home", "terzo_gol_home",
            "primo_gol_away", "secondo_gol_away", "terzo_gol_away"]:
    if col in df.columns:
        timing_choice = st.sidebar.selectbox(f"Timing {col}", timing_options, key=col)
        if timing_choice != "Tutti":
            min_t, max_t = map(int, timing_choice.split('-'))
            filters[col] = (min_t, max_t)

# --- Applica filtri ---
filtered_df = df.copy()
for col, val in filters.items():
    if col in ["odd_home", "odd_away", "odd_draw"]:
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        ]
    elif col in ["primo_gol_home", "secondo_gol_home", "terzo_gol_home",
                 "primo_gol_away", "secondo_gol_away", "terzo_gol_away"]:
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        ]

if filters == {}:
    st.info("Nessun filtro attivo: vengono mostrati tutti i risultati.")

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- FUNZIONE DISTRIBUZIONE ---
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
    st.table(pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
        "Conteggio": [count_1, count_x, count_2],
        "WinRate %": winrate,
        "Odd Minima": [round(100/w,2) if w > 0 else "-" for w in winrate]
    }))

# --- STATISTICHE ---
if not filtered_df.empty:
    mostra_distribuzione(filtered_df, "risultato_ft", "Risultati Finali (FT)")
    mostra_distribuzione(filtered_df, "risultato_ht", "Risultati Primo Tempo (HT)")

    # BTTS & Over
    temp_ft = filtered_df["risultato_ft"].str.split("-", expand=True).astype(int)
    filtered_df["home_g_ft"], filtered_df["away_g_ft"] = temp_ft[0], temp_ft[1]
    filtered_df["tot_goals_ft"] = filtered_df["home_g_ft"] + filtered_df["away_g_ft"]

    st.subheader("BTTS (Both Teams To Score)")
    btts = (filtered_df["home_g_ft"] > 0) & (filtered_df["away_g_ft"] > 0)
    st.write(f"Partite BTTS SI: {btts.sum()} ({round(btts.mean()*100,2)}%)")

    st.subheader("Over Goals (FT)")
    over_data = []
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        count = (filtered_df["tot_goals_ft"] > t).sum()
        perc = round((count / len(filtered_df)) * 100, 2)
        over_data.append([f"Over {t}", count, perc, round(100/perc, 2) if perc > 0 else "-"])
    st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))
