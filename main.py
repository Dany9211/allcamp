import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi Next Gol e stats live", layout="wide")
st.title("Analisi Tabella allcamp")

# --- Funzione connessione ---
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

# --- Caricamento dati ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento: {e}")
    st.stop()

# --- Aggiunta colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

# --- Slider minuti ---
st.sidebar.header("Filtro Minuti per analisi")
start_min = st.sidebar.slider("Minuto Iniziale", 1, 90, 1)
end_min = st.sidebar.slider("Minuto Finale", start_min, 120, 45)

# --- Filtro DataFrame ---
filtered_df = df.copy()

def parse_minutaggio(val):
    return [int(x) for x in str(val).split(";") if x.isdigit()]

def is_in_timeband(gol_home, gol_away):
    return any(start_min <= g <= end_min for g in gol_home + gol_away)

filtered_df["gol_home"] = df["minutaggio_gol"].apply(parse_minutaggio)
filtered_df["gol_away"] = df["minutaggio_gol_away"].apply(parse_minutaggio)
filtered_df = filtered_df[filtered_df.apply(lambda row: is_in_timeband(row["gol_home"], row["gol_away"]), axis=1)]

st.write(f"**Partite nel range {start_min}-{end_min}:** {len(filtered_df)}")

# --- Distribuzione Gol per timeband 5 minuti ---
timebands = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 45),
             (46, 50), (51, 55), (56, 60), (61, 65), (66, 70), (71, 75), (76, 80), (81, 85), (86, 90), (91, 120)]
st.subheader("Distribuzione Gol ogni 5 minuti")
risultati = []
for start, end in timebands:
    count = 0
    for _, row in filtered_df.iterrows():
        gol_home = row["gol_home"]
        gol_away = row["gol_away"]
        if any(start <= g <= end for g in gol_home + gol_away):
            count += 1
    perc = round(count / len(filtered_df) * 100, 2) if len(filtered_df) > 0 else 0
    odd_min = round(100 / perc, 2) if perc > 0 else "-"
    label = f"{start}-{end}" if end < 91 else "90+"
    risultati.append([label, count, perc, odd_min])

st.table(pd.DataFrame(risultati, columns=["Timeframe (min)", "Partite con Gol", "Percentuale %", "Odd Minima"]))

# --- Risultati esatti ---
def mostra_risultati(df, col):
    df_val = df[df[col].notna() & df[col].str.contains("-")].copy()
    risultati = df_val[col].value_counts().reset_index()
    risultati.columns = ["Risultato", "Conteggio"]
    risultati["%"] = (risultati["Conteggio"] / len(df_val) * 100).round(2)
    risultati["Odd Min"] = risultati["%"].apply(lambda x: round(100 / x, 2) if x > 0 else "-")
    return risultati

if not filtered_df.empty:
    st.subheader("Risultati Esatti HT")
    st.table(mostra_risultati(filtered_df, "risultato_ht"))

    st.subheader("Risultati Esatti FT")
    st.table(mostra_risultati(filtered_df, "risultato_ft"))

# --- WinRate ---
def calcola_winrate(col):
    df_val = filtered_df[filtered_df[col].notna() & filtered_df[col].str.contains("-")]
    esiti = {"1": 0, "X": 0, "2": 0}
    for ris in df_val[col]:
        try:
            h, a = map(int, ris.split("-"))
            if h > a:
                esiti["1"] += 1
            elif h < a:
                esiti["2"] += 1
            else:
                esiti["X"] += 1
        except:
            continue
    tot = sum(esiti.values())
    tab = []
    for k, v in esiti.items():
        perc = round(v / tot * 100, 2) if tot > 0 else 0
        odd = round(100 / perc, 2) if perc > 0 else "-"
        tab.append([k, v, perc, odd])
    return pd.DataFrame(tab, columns=["Esito", "Conteggio", "%", "Odd Min"])

st.subheader("WinRate HT")
st.table(calcola_winrate("risultato_ht"))

st.subheader("WinRate FT")
st.table(calcola_winrate("risultato_ft"))

# --- BTTS SI ---
def calcola_btts(df):
    btts_si = df[(df["gol_home_ft"] > 0) & (df["gol_away_ft"] > 0)]
    perc = round(len(btts_si) / len(df) * 100, 2) if len(df) > 0 else 0
    odd_min = round(100 / perc, 2) if perc > 0 else "-"
    return len(btts_si), perc, odd_min

bt, perc_bt, odd_bt = calcola_btts(filtered_df)
st.subheader("BTTS SI")
st.write(f"BTTS SI: {bt} ({perc_bt}%) - Odd Minima: {odd_bt}")

# --- Over Goals ---
def calcola_over_goals(df, col, soglie):
    tab = []
    for soglia in soglie:
        count = df[(df[col].str.split("-").str[0].astype(int) + df[col].str.split("-").str[1].astype(int)) > soglia].shape[0]
        perc = round(count / len(df) * 100, 2) if len(df) > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        tab.append([f"Over {soglia}", count, perc, odd_min])
    return pd.DataFrame(tab, columns=["Mercato", "Conteggio", "%", "Odd Minima"])

st.subheader("Over Goals HT")
st.table(calcola_over_goals(filtered_df, "risultato_ht", [0.5, 1.5, 2.5]))

st.subheader("Over Goals FT")
st.table(calcola_over_goals(filtered_df, "risultato_ft", [0.5, 1.5, 2.5, 3.5, 4.5]))
