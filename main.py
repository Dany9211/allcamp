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


# --- NUOVA FUNZIONALITÀ ---
def analizza_gol_da_minuto(df):
    st.subheader("Analisi Gol da Minuto selezionato")
    
    minuto_inizio = st.slider("Seleziona Minuto Corrente", 0, 45, 20)
    risultati_ht = ["0-0", "0-1", "1-0", "1-1", "Altro"]
    risultato_corrente = st.selectbox("Risultato corrente al minuto selezionato", risultati_ht)
    
    partite_target = 0
    over_05 = 0
    over_15 = 0
    timeframe_counter = { "20-30":0, "31-40":0, "41-45":0 }

    for _, row in df.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
        gol_tot = gol_home + gol_away
        
        # Gol dopo il minuto selezionato
        gol_successivi = [g for g in gol_tot if minuto_inizio < g <= 45]

        # Controllo risultato corrente
        ht_gol_home = sum(1 for g in gol_home if g <= minuto_inizio)
        ht_gol_away = sum(1 for g in gol_away if g <= minuto_inizio)
        current_score = f"{ht_gol_home}-{ht_gol_away}"
        if risultato_corrente != "Altro" and current_score != risultato_corrente:
            continue
        
        partite_target += 1
        if len(gol_successivi) >= 1:
            over_05 += 1
        if len(gol_successivi) >= 2:
            over_15 += 1
        
        # Conteggio timeframe
        for g in gol_successivi:
            if 20 <= g <= 30:
                timeframe_counter["20-30"] += 1
            elif 31 <= g <= 40:
                timeframe_counter["31-40"] += 1
            elif 41 <= g <= 45:
                timeframe_counter["41-45"] += 1

    st.write(f"**Partite analizzate:** {partite_target}")
    st.write(f"Over 0.5 HT successivo: {over_05} ({round(over_05/partite_target*100,2) if partite_target>0 else 0}%)")
    st.write(f"Over 1.5 HT successivo: {over_15} ({round(over_15/partite_target*100,2) if partite_target>0 else 0}%)")
    st.write("Distribuzione gol successivi per timeframe (20-30, 31-40, 41-45):")
    st.write(timeframe_counter)

    if partite_target > 0:
        st.write("**Pattern comuni sulle quote:**")
        st.write(df[["odd_home", "odd_draw", "odd_away"]].mean().round(2))


# --- STATISTICHE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    temp_ft = filtered_df["risultato_ft"].str.split("-", expand=True)
    temp_ft = temp_ft.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    filtered_df["home_g_ft"], filtered_df["away_g_ft"] = temp_ft[0], temp_ft[1]
    filtered_df["tot_goals_ft"] = filtered_df["home_g_ft"] + filtered_df["away_g_ft"]

    analizza_gol_da_minuto(filtered_df)
