import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Allcamp Viewer", layout="wide")
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

filters = {}

# --- FILTRO LEAGUE ---
if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona League", leagues)
    if selected_league != "Tutte":
        filters["league"] = selected_league

# --- FILTRO ANNO ---
if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
    if selected_anno != "Tutti":
        filters["anno"] = selected_anno

# --- FILTRO GIORNATA ---
if "giornata" in df.columns:
    giornata_min = int(df["giornata"].min())
    giornata_max = int(df["giornata"].max())
    giornata_range = st.sidebar.slider(
        "Seleziona Giornata",
        min_value=giornata_min,
        max_value=giornata_max,
        value=(giornata_min, giornata_max)
    )
    filters["giornata"] = giornata_range

# --- FILTRO RISULTATO HT ---
if "risultato_ht" in df.columns:
    ht_results = ["Tutti"] + sorted(df["risultato_ht"].dropna().unique())
    selected_ht = st.sidebar.selectbox("Seleziona Risultato HT", ht_results)
    if selected_ht != "Tutti":
        filters["risultato_ht"] = selected_ht

# --- FILTRO SQUADRA HOME ---
if "home_team" in df.columns:
    home_teams = ["Tutte"] + sorted(df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
    if selected_home != "Tutte":
        filters["home_team"] = selected_home

# --- FILTRO SQUADRA AWAY ---
if "away_team" in df.columns:
    away_teams = ["Tutte"] + sorted(df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)
    if selected_away != "Tutte":
        filters["away_team"] = selected_away

# --- FILTRI QUOTE MANUALI ---
def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        st.sidebar.write(f"Range attuale {col_name}: {col_min} - {col_max}")

        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="")

        if min_val.strip() != "" and max_val.strip() != "":
            try:
                min_val = float(min_val)
                max_val = float(max_val)
                filters[col_name] = (min_val, max_val)
            except:
                st.sidebar.warning(f"Valori non validi per {col_name}")

st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- PULSANTE RESET ---
if st.sidebar.button("🔄 Reset Filtri"):
    st.session_state.clear()
    st.rerun()

# --- APPLICA FILTRI ---
filtered_df = df.copy()

for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        mask = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "giornata":
        mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

if filters == {}:
    st.info("Nessun filtro attivo: vengono mostrati tutti i risultati.")

st.subheader("Dati Filtrati")
st.dataframe(filtered_df.head(50))
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- FUNZIONI ---
def calcola_winrate(df, col_risultato):
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))]
    risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for ris in df_valid[col_risultato]:
        try:
            home, away = map(int, ris.split("-"))
            if home > away:
                risultati["1 (Casa)"] += 1
            elif home < away:
                risultati["2 (Trasferta)"] += 1
            else:
                risultati["X (Pareggio)"] += 1
        except:
            continue
    totale = len(df_valid)
    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale) * 100, 2) if totale > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    return stats, totale

def mostra_winrate_combinato(df):
    stats_ht, totale_ht = calcola_winrate(df, "risultato_ht")
    stats_ft, totale_ft = calcola_winrate(df, "risultato_ft")

    combined_data = []
    for i in range(3):
        combined_data.append([
            stats_ft[i][0], 
            stats_ht[i][1], stats_ht[i][2], stats_ht[i][3], 
            stats_ft[i][1], stats_ft[i][2], stats_ft[i][3]
        ])

    df_combined = pd.DataFrame(combined_data, columns=[
        "Esito",
        "Conteggio HT", "WinRate HT %", "Odd Minima HT",
        "Conteggio FT", "WinRate FT %", "Odd Minima FT"
    ])
    st.subheader("WinRate Combinato (HT & FT)")
    st.table(df_combined)

def analizza_gol_da_minuto(df):
    st.subheader("Analisi Gol da Minuto Selezionato")
    minuto_selezionato = st.slider("Minuto di riferimento", 0, 45, 20)

    partite_con_gol = 0
    for _, row in df.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
        gol_tot = gol_home + gol_away
        if any(minuto_selezionato <= g <= 45 for g in gol_tot):
            partite_con_gol += 1

    totale = len(df)
    perc = round((partite_con_gol / totale) * 100, 2) if totale > 0 else 0
    st.write(f"Partite con gol tra {minuto_selezionato} e 45': {partite_con_gol} ({perc}%)")

    # Quote medie
    odds_cols = ["odd_home", "odd_draw", "odd_away"]
    df_odds = df[odds_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", "."), errors="coerce"))
    st.write("Quote medie sulle partite con gol successivi:")
    st.write(df_odds.mean().round(2))

# --- STATISTICHE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    mostra_winrate_combinato(filtered_df)
    analizza_gol_da_minuto(filtered_df)
