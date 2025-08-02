import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi Next Gol e stats live", layout="wide")
st.title("Analisi Tabella allcamp")

# --- Funzione connessione al database ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database
    ogni volta che l'applicazione si aggiorna.
    """
    try:
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
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        st.stop()
        return pd.DataFrame()

# --- Caricamento dati iniziali ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto.")
        st.stop()
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento del database: {e}")
    st.stop()

# --- Aggiunta colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

filters = {}

# --- FILTRI INIZIALI ---
st.sidebar.header("Filtri Dati")

# Filtro League (Campionato) - Deve essere il primo per filtrare le squadre
if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)
    if selected_league != "Tutte":
        filters["league"] = selected_league
    
    # Crea un DataFrame temporaneo per filtrare le squadre in base al campionato
    if selected_league != "Tutte":
        filtered_teams_df = df[df["league"] == selected_league]
    else:
        filtered_teams_df = df.copy()
else:
    filtered_teams_df = df.copy()
    selected_league = "Tutte"

# Filtro Anno
if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
    if selected_anno != "Tutti":
        filters["anno"] = selected_anno

# Filtro Giornata
if "giornata" in df.columns:
    giornata_min = int(df["giornata"].min()) if not df["giornata"].isnull().all() else 1
    giornata_max = int(df["giornata"].max()) if not df["giornata"].isnull().all() else 38
    giornata_range = st.sidebar.slider(
        "Seleziona Giornata",
        min_value=giornata_min,
        max_value=giornata_max,
        value=(giornata_min, giornata_max)
    )
    filters["giornata"] = giornata_range

# --- FILTRI SQUADRE (ora dinamici) ---
if "home_team" in filtered_teams_df.columns:
    home_teams = ["Tutte"] + sorted(filtered_teams_df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
    if selected_home != "Tutte":
        filters["home_team"] = selected_home

if "away_team" in filtered_teams_df.columns:
    away_teams = ["Tutte"] + sorted(filtered_teams_df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)
    if selected_away != "Tutte":
        filters["away_team"] = selected_away

# --- NUOVO FILTRO: Risultato HT ---
if "risultato_ht" in df.columns:
    ht_results = sorted(df["risultato_ht"].dropna().unique())
    selected_ht_results = st.sidebar.multiselect("Seleziona Risultato HT", ht_results, default=None)
    if selected_ht_results:
        filters["risultato_ht"] = selected_ht_results

# --- FUNZIONE per filtri range ---
def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        
        st.sidebar.write(f"Range attuale {label or col_name}: {col_min} - {col_max}")
        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="")
        
        if min_val.strip() != "" and max_val.strip() != "":
            try:
                filters[col_name] = (float(min_val), float(max_val))
            except ValueError:
                st.sidebar.warning(f"Valori non validi per {label or col_name}. Inserisci numeri.")

st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- APPLICA FILTRI AL DATAFRAME PRINCIPALE ---
filtered_df = df.copy()
for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        mask = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "giornata":
        mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "risultato_ht":
        filtered_df = filtered_df[filtered_df[col].isin(val)]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

st.subheader("Dati Filtrati")
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
st.dataframe(filtered_df.head(50))


# --- FUNZIONE WINRATE ---
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
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

# --- FUNZIONE CALCOLO FIRST TO SCORE ---
def calcola_first_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else: 
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVA FUNZIONE CALCOLO FIRST TO SCORE HT ---
def calcola_first_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        # Considera solo i gol segnati nel primo tempo (minuto <= 45)
        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) <= 45]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) <= 45]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else: 
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- FUNZIONE RISULTATI ESATTI ---
def mostra_risultati_esatti(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))].copy()

    def classifica_risultato(ris):
        try:
            home, away = map(int, ris.split("-"))
        except:
            return "Altro"
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
    distribuzione = df_valid["classificato"].value_counts().reset_index()
    distribuzione.columns = [titolo, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    st.subheader(f"Risultati Esatti {titolo} ({len(df_valid)} partite)")
    styled_df = distribuzione.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (15 MIN) ---
def mostra_distribuzione_timeband(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return
    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 150)]
    label_intervalli = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- NUOVA FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
def mostra_distribuzione_timeband_5min(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 5 minuti è vuoto.")
        return
    intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90), (91, 150)]
    label_intervalli = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "90+"]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE NEXT GOAL ---
def calcola_next_goal(df_to_analyze, start_min, end_min):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Prossimo Gol: Home": 0, "Prossimo Gol: Away": 0, "Nessun prossimo gol": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]

        next_home_goal = min([g for g in gol_home if start_min <= g <= end_min] or [float('inf')])
        next_away_goal = min([g for g in gol_away if start_min <= g <= end_min] or [float('inf')])
        
        if next_home_goal < next_away_goal:
            risultati["Prossimo Gol: Home"] += 1
        elif next_away_goal < next_home_goal:
            risultati["Prossimo Gol: Away"] += 1
        else:
            if next_home_goal == float('inf'):
                risultati["Nessun prossimo gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVE FUNZIONI PER ANALISI RIMONTE ---
def calcola_rimonte(df_to_analyze, titolo_analisi):
    if df_to_analyze.empty:
        return pd.DataFrame(), []

    partite_rimonta_parziale = []
    partite_rimonta_completa = []
    
    df_rimonte = df_to_analyze.copy()
    
    # Aggiungi colonne per i gol HT e FT
    df_rimonte["gol_home_ht"] = pd.to_numeric(df_rimonte["gol_home_ht"], errors='coerce')
    df_rimonte["gol_away_ht"] = pd.to_numeric(df_rimonte["gol_away_ht"], errors='coerce')
    df_rimonte["gol_home_ft"] = pd.to_numeric(df_rimonte["gol_home_ft"], errors='coerce')
    df_rimonte["gol_away_ft"] = pd.to_numeric(df_rimonte["gol_away_ft"], errors='coerce')

    def check_comeback(row):
        # Rimonta Home
        if row["gol_home_ht"] < row["gol_away_ht"] and row["gol_home_ft"] > row["gol_away_ft"]:
            return "Completa - Home"
        if row["gol_home_ht"] < row["gol_away_ht"] and row["gol_home_ft"] == row["gol_away_ft"]:
            return "Parziale - Home"
        # Rimonta Away
        if row["gol_away_ht"] < row["gol_home_ht"] and row["gol_away_ft"] > row["gol_home_ft"]:
            return "Completa - Away"
        if row["gol_away_ht"] < row["gol_home_ht"] and row["gol_away_ft"] == row["gol_home_ft"]:
            return "Parziale - Away"
        return "Nessuna"

    df_rimonte["rimonta"] = df_rimonte.apply(check_comeback, axis=1)
    
    # Filtra e conta i risultati
    rimonte_completa_home = (df_rimonte["rimonta"] == "Completa - Home").sum()
    rimonte_parziale_home = (df_rimonte["rimonta"] == "Parziale - Home").sum()
    rimonte_completa_away = (df_rimonte["rimonta"] == "Completa - Away").sum()
    rimonte_parziale_away = (df_rimonte["rimonta"] == "Parziale - Away").sum()

    totale = len(df_rimonte)
    
    rimonte_data = [
        ["Rimonta Completa (Home)", rimonte_completa_home, round((rimonte_completa_home / totale) * 100, 2) if totale > 0 else 0],
        ["Rimonta Parziale (Home)", rimonte_parziale_home, round((rimonte_parziale_home / totale) * 100, 2) if totale > 0 else 0],
        ["Rimonta Completa (Away)", rimonte_completa_away, round((rimonte_completa_away / totale) * 100, 2) if totale > 0 else 0],
        ["Rimonta Parziale (Away)", rimonte_parziale_away, round((rimonte_parziale_away / totale) * 100, 2) if totale > 0 else 0]
    ]

    df_rimonte_stats = pd.DataFrame(rimonte_data, columns=["Tipo Rimonta", "Conteggio", "Percentuale %"])
    df_rimonte_stats["Odd Minima"] = df_rimonte_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    # Crea la lista di squadre per ogni tipo di rimonta
    squadre_rimonta_completa_home = df_rimonte[df_rimonte["rimonta"] == "Completa - Home"]["home_team"].tolist()
    squadre_rimonta_parziale_home = df_rimonte[df_rimonte["rimonta"] == "Parziale - Home"]["home_team"].tolist()
    squadre_rimonta_completa_away = df_rimonte[df_rimonte["rimonta"] == "Completa - Away"]["away_team"].tolist()
    squadre_rimonta_parziale_away = df_rimonte[df_rimonte["rimonta"] == "Parziale - Away"]["away_team"].tolist()
    
    squadre_rimonte = {
        "Rimonta Completa (Home)": squadre_rimonta_completa_home,
        "Rimonta Parziale (Home)": squadre_rimonta_parziale_home,
        "Rimonta Completa (Away)": squadre_rimonta_completa_away,
        "Rimonta Parziale (Away)": squadre_rimonta_parziale_away
    }

    return df_rimonte_stats, squadre_rimonte

# --- NUOVA FUNZIONE PER TO SCORE ---
def calcola_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_ft"] = pd.to_numeric(df_to_score["gol_home_ft"], errors='coerce')
    df_to_score["gol_away_ft"] = pd.to_numeric(df_to_score["gol_away_ft"], errors='coerce')

    home_to_score_count = (df_to_score["gol_home_ft"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_ft"] > 0).sum()
    
    total_matches = len(df_to_score)
    
    data = [
        ["Home Team to Score", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER TO SCORE HT ---
def calcola_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_ht"] = pd.to_numeric(df_to_score["gol_home_ht"], errors='coerce')
    df_to_score["gol_away_ht"] = pd.to_numeric(df_to_score["gol_away_ht"], errors='coerce')

    home_to_score_count = (df_to_score["gol_home_ht"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_ht"] > 0).sum()
    
    total_matches = len(df_to_score)
    
    data = [
        ["Home Team to Score", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS HT ---
def calcola_btts_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    df_btts_ht = df_to_analyze.copy()
    df_btts_ht["gol_home_ht"] = pd.to_numeric(df_btts_ht["gol_home_ht"], errors='coerce')
    df_btts_ht["gol_away_ht"] = pd.to_numeric(df_btts_ht["gol_away_ht"], errors='coerce')
    
    btts_count = ((df_btts_ht["gol_home_ht"] > 0) & (df_btts_ht["gol_away_ht"] > 0)).sum()
    no_btts_count = len(df_btts_ht) - btts_count
    
    total_matches = len(df_btts_ht)
    
    data = [
        ["BTTS SI HT", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO HT", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS FT ---
def calcola_btts_ft(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    df_btts_ft = df_to_analyze.copy()
    df_btts_ft["gol_home_ft"] = pd.to_numeric(df_btts_ft["gol_home_ft"], errors='coerce')
    df_btts_ft["gol_away_ft"] = pd.to_numeric(df_btts_ft["gol_away_ft"], errors='coerce')
    
    btts_count = ((df_btts_ft["gol_home_ft"] > 0) & (df_btts_ft["gol_away_ft"] > 0)).sum()
    no_btts_count = len(df_btts_ft) - btts_count
    
    total_matches = len(df_btts_ft)
    
    data = [
        ["BTTS SI FT", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO FT", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS DINAMICO ---
def calcola_btts_dinamico(df_to_analyze, start_min, risultati_correnti):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    total_matches = len(df_to_analyze)
    btts_si_count = 0
    
    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))
        
        gol_home_before = sum(1 for g in [int(x) for x in gol_home_str.split(";") if x.isdigit()] if g < start_min)
        gol_away_before = sum(1 for g in [int(x) for x in gol_away_str.split(";") if x.isdigit()] if g < start_min)
        
        gol_home_ft = int(row.get("gol_home_ft", 0))
        gol_away_ft = int(row.get("gol_away_ft", 0))
        
        # Logica per BTTS SI dinamico
        btts_si = False
        if "0-0" in risultati_correnti and gol_home_ft > 0 and gol_away_ft > 0:
            btts_si = True
        elif "1-0" in risultati_correnti and gol_away_ft > gol_away_before:
            btts_si = True
        elif "0-1" in risultati_correnti and gol_home_ft > gol_home_before:
            btts_si = True
        elif "1-1" in risultati_correnti:
            btts_si = True
        elif "2-0" in risultati_correnti and gol_away_ft > gol_away_before:
            btts_si = True
        elif "0-2" in risultati_correnti and gol_home_ft > gol_home_before:
            btts_si = True
        elif "2-1" in risultati_correnti and gol_away_ft > gol_away_before:
            btts_si = True
        elif "1-2" in risultati_correnti and gol_home_ft > gol_home_before:
            btts_si = True
            
        if btts_si:
            btts_si_count += 1

    btts_no_count = total_matches - btts_si_count

    data = [
        ["BTTS SI (Dinamica)", btts_si_count, round((btts_si_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO (Dinamica)", btts_no_count, round((btts_no_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    return df_stats
    
# --- NUOVA FUNZIONE PER BTTS HT DINAMICO ---
def calcola_btts_ht_dinamico(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ht_dinamico = df_to_analyze.copy()
    
    # Assicurati che le colonne siano numeriche
    df_btts_ht_dinamico["gol_home_ht"] = pd.to_numeric(df_btts_ht_dinamico["gol_home_ht"], errors='coerce')
    df_btts_ht_dinamico["gol_away_ht"] = pd.to_numeric(df_btts_ht_dinamico["gol_away_ht"], errors='coerce')
    
    btts_count = ((df_btts_ht_dinamico["gol_home_ht"] > 0) & (df_btts_ht_dinamico["gol_away_ht"] > 0)).sum()
    no_btts_count = len(df_btts_ht_dinamico) - btts_count
    
    total_matches = len(df_btts_ht_dinamico)
    
    data = [
        ["BTTS SI HT (Dinamica)", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO HT (Dinamica)", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER CLEAN SHEET ---
def calcola_clean_sheet(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()
    
    df_clean_sheet = df_to_analyze.copy()
    
    df_clean_sheet["gol_home_ft"] = pd.to_numeric(df_clean_sheet["gol_home_ft"], errors='coerce')
    df_clean_sheet["gol_away_ft"] = pd.to_numeric(df_clean_sheet["gol_away_ft"], errors='coerce')
    
    home_clean_sheet_count = (df_clean_sheet["gol_away_ft"] == 0).sum()
    away_clean_sheet_count = (df_clean_sheet["gol_home_ft"] == 0).sum()
    
    total_matches = len(df_clean_sheet)
    
    data = [
        ["Clean Sheet (Casa)", home_clean_sheet_count, round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Clean Sheet (Trasferta)", away_clean_sheet_count, round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER COMBO MARKETS ---
def calcola_combo_stats(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()
        
    df_combo = df_to_analyze.copy()

    df_combo["gol_home_ft"] = pd.to_numeric(df_combo["gol_home_ft"], errors='coerce')
    df_combo["gol_away_ft"] = pd.to_numeric(df_combo["gol_away_ft"], errors='coerce')
    
    df_combo["tot_goals_ft"] = df_combo["gol_home_ft"] + df_combo["gol_away_ft"]
    
    # BTTS SI + Over 2.5
    btts_over_2_5_count = ((df_combo["gol_home_ft"] > 0) & (df_combo["gol_away_ft"] > 0) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    # Home Win + Over 2.5
    home_win_over_2_5_count = ((df_combo["gol_home_ft"] > df_combo["gol_away_ft"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    # Away Win + Over 2.5
    away_win_over_2_5_count = ((df_combo["gol_away_ft"] > df_combo["gol_home_ft"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    total_matches = len(df_combo)
    
    data = [
        ["BTTS SI + Over 2.5", btts_over_2_5_count, round((btts_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Casa vince + Over 2.5", home_win_over_2_5_count, round((home_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Ospite vince + Over 2.5", away_win_over_2_5_count, round((away_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER MULTI GOL ---
def calcola_multi_gol(df_to_analyze, col_gol, titolo):
    if df_to_analyze.empty:
        return pd.DataFrame()
    
    df_multi_gol = df_to_analyze.copy()
    df_multi_gol[col_gol] = pd.to_numeric(df_multi_gol[col_gol], errors='coerce')
    
    total_matches = len(df_multi_gol)
    
    multi_gol_ranges = [
        ("0-1", lambda x: (x >= 0) & (x <= 1)),
        ("1-2", lambda x: (x >= 1) & (x <= 2)),
        ("2-3", lambda x: (x >= 2) & (x <= 3)),
        ("3+", lambda x: (x >= 3))
    ]
    
    data = []
    for label, condition in multi_gol_ranges:
        count = df_multi_gol[condition(df_multi_gol[col_gol])].shape[0]
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        data.append([f"Multi Gol {label}", count, perc, odd_min])
        
    df_stats = pd.DataFrame(data, columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])
    return df_stats

# --- SEZIONE 1: Analisi Timeband per Campionato ---
st.subheader("1. Analisi Timeband per Campionato")
if selected_league != "Tutte":
    df_league_only = df[df["league"] == selected_league]
    st.write(f"Analisi basata su **{len(df_league_only)}** partite del campionato **{selected_league}**.")
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(df_league_only)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(df_league_only)
else:
    st.write("Seleziona un campionato per visualizzare questa analisi.")

# --- SEZIONE 2: Analisi Timeband per Campionato e Quote ---
st.subheader("2. Analisi Timeband per Campionato e Quote")
st.write(f"Analisi basata su **{len(filtered_df)}** partite filtrate da tutti i parametri della sidebar.")
if not filtered_df.empty:
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(filtered_df)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(filtered_df)
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati.")


# --- NUOVA SEZIONE: Statistiche Pre-Match Complete (Filtri Sidebar) ---
st.subheader("3. Analisi Pre-Match Completa (Filtri Sidebar)")
st.write(f"Analisi completa basata su **{len(filtered_df)}** partite, considerando tutti i filtri del menu a sinistra.")
if not filtered_df.empty:
    
    # Calcolo e visualizzazione media gol
    st.subheader("Media Gol (Pre-Match)")
    df_prematch_goals = filtered_df.copy()
    
    df_prematch_goals["gol_home_ht"] = pd.to_numeric(df_prematch_goals["gol_home_ht"], errors='coerce')
    df_prematch_goals["gol_away_ht"] = pd.to_numeric(df_prematch_goals["gol_away_ht"], errors='coerce')
    df_prematch_goals["gol_home_ft"] = pd.to_numeric(df_prematch_goals["gol_home_ft"], errors='coerce')
    df_prematch_goals["gol_away_ft"] = pd.to_numeric(df_prematch_goals["gol_away_ft"], errors='coerce')
    
    # Media gol HT
    avg_ht_goals = (df_prematch_goals["gol_home_ht"] + df_prematch_goals["gol_away_ht"]).mean()
    # Media gol FT
    avg_ft_goals = (df_prematch_goals["gol_home_ft"] + df_prematch_goals["gol_away_ft"]).mean()
    # Media gol SH (secondo tempo)
    avg_sh_goals = (df_prematch_goals["gol_home_ft"] + df_prematch_goals["gol_away_ft"] - df_prematch_goals["gol_home_ht"] - df_prematch_goals["gol_away_ht"]).mean()
    
    st.table(pd.DataFrame({
        "Periodo": ["HT", "FT", "SH"],
        "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
    }))
    
    mostra_risultati_esatti(filtered_df, "risultato_ht", "HT")
    mostra_risultati_esatti(filtered_df, "risultato_ft", "FT")

    # WinRate con grafico
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("WinRate HT")
        df_winrate_ht = calcola_winrate(filtered_df, "risultato_ht")
        styled_df_ht = df_winrate_ht.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
        st.dataframe(styled_df_ht)
    with col2:
        st.subheader("WinRate FT")
        df_winrate_ft = calcola_winrate(filtered_df, "risultato_ft")
        styled_df_ft = df_winrate_ft.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
        st.dataframe(styled_df_ft)

    # Over Goals HT e FT
    col1, col2 = st.columns(2)
    df_prematch_ht = filtered_df.copy()
    df_prematch_ht["tot_goals_ht"] = pd.to_numeric(df_prematch_ht["gol_home_ht"], errors='coerce') + pd.to_numeric(df_prematch_ht["gol_away_ht"], errors='coerce')
    df_prematch_ht["tot_goals_ft"] = pd.to_numeric(df_prematch_ht["gol_home_ft"], errors='coerce') + pd.to_numeric(df_prematch_ht["gol_away_ft"], errors='coerce')

    with col1:
        st.subheader("Over Goals HT")
        over_ht_data = []
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_prematch_ht["tot_goals_ht"] > t).sum()
            perc = round((count / len(df_prematch_ht)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else "-"
            over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
        df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        styled_over_ht = df_over_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_over_ht)

    with col2:
        st.subheader("Over Goals FT")
        over_ft_data = []
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_prematch_ht["tot_goals_ft"] > t).sum()
            perc = round((count / len(df_prematch_ht)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else "-"
            over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
        df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_over_ft)
        
    # BTTS
    st.subheader("BTTS (Pre-Match)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### HT")
        df_btts_ht = calcola_btts_ht(filtered_df)
        styled_df = df_btts_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
    with col2:
        st.write("### FT")
        df_btts_ft = calcola_btts_ft(filtered_df)
        styled_df = df_btts_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)

    # Multi Gol
    st.subheader("Multi Gol (Pre-Match)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Casa")
        styled_df = calcola_multi_gol(filtered_df, "gol_home_ft", "Home").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
    with col2:
        st.write("### Trasferta")
        styled_df = calcola_multi_gol(filtered_df, "gol_away_ft", "Away").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)

    # First to Score
    st.subheader("First to Score (Pre-Match)")
    styled_df = calcola_first_to_score(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)
    
    # To Score
    st.subheader("To Score (Pre-Match)")
    styled_df = calcola_to_score(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)
    
    # Clean Sheet
    st.subheader("Clean Sheet (Pre-Match)")
    styled_df = calcola_clean_sheet(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)
    
    # Combo Markets
    st.subheader("Combo Markets (Pre-Match)")
    styled_df = calcola_combo_stats(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)
    
    # Analisi Rimonte
    st.subheader("Analisi Rimonte (Pre-Match)")
    rimonte_stats, squadre_rimonte = calcola_rimonte(filtered_df, "Pre-Match")
    if not rimonte_stats.empty:
        styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        
        st.markdown("**Squadre che hanno effettuato rimonte:**")
        for tipo, squadre in squadre_rimonte.items():
            if squadre:
                st.markdown(f"**{tipo}:** {', '.join(squadre)}")
    else:
        st.warning("Nessuna rimonta trovata nel dataset filtrato.")

else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per l'analisi pre-match.")

# --- SEZIONE 4: Analisi Timeband Dinamica (Minuto/Risultato) ---
st.subheader("4. Analisi Timeband Dinamica")
with st.expander("Mostra Analisi Dinamica (Minuto/Risultato)"):
    if not filtered_df.empty:
        # --- ANALISI DAL MINUTO (integrata) ---
        # Cursore unico per il range di minuti
        min_range = st.slider("Seleziona Range Minuti", 1, 90, (45, 90))
        start_min, end_min = min_range[0], min_range[1]

        risultati_correnti = st.multiselect("Risultato corrente al minuto iniziale",
                                            sorted(df["risultato_ht"].unique()) if "risultato_ht" in df.columns else [],
                                            default=["0-0"])

        partite_target = []
        for _, row in filtered_df.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            home_fino = sum(1 for g in gol_home if g < start_min)
            away_fino = sum(1 for g in gol_away if g < start_min)
            risultato_fino = f"{home_fino}-{away_fino}"
            if risultato_fino in risultati_correnti:
                partite_target.append(row)

        if not partite_target:
            st.warning(f"Nessuna partita con risultato selezionato al minuto {start_min}.")
        else:
            df_target = pd.DataFrame(partite_target)
            st.write(f"**Partite trovate:** {len(df_target)}")

            # Calcolo e visualizzazione media gol dinamica
            st.subheader("Media Gol (Dinamica)")
            df_target_goals = df_target.copy()
            
            df_target_goals["gol_home_ht"] = pd.to_numeric(df_target_goals["gol_home_ht"], errors='coerce')
            df_target_goals["gol_away_ht"] = pd.to_numeric(df_target_goals["gol_away_ht"], errors='coerce')
            df_target_goals["gol_home_ft"] = pd.to_numeric(df_target_goals["gol_home_ft"], errors='coerce')
            df_target_goals["gol_away_ft"] = pd.to_numeric(df_target_goals["gol_away_ft"], errors='coerce')
            
            # Media gol HT
            avg_ht_goals_dynamic = (df_target_goals["gol_home_ht"] + df_target_goals["gol_away_ht"]).mean()
            # Media gol FT
            avg_ft_goals_dynamic = (df_target_goals["gol_home_ft"] + df_target_goals["gol_away_ft"]).mean()
            # Media gol SH (secondo tempo)
            avg_sh_goals_dynamic = (df_target_goals["gol_home_ft"] + df_target_goals["gol_away_ft"] - df_target_goals["gol_home_ht"] - df_target_goals["gol_away_ht"]).mean()
            
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals_dynamic:.2f}", f"{avg_ft_goals_dynamic:.2f}", f"{avg_sh_goals_dynamic:.2f}"]
            }))
            
            mostra_risultati_esatti(df_target, "risultato_ht", "HT")
            mostra_risultati_esatti(df_target, "risultato_ft", "FT")

            # WinRate
            st.subheader(f"WinRate (Dinamica)")
            st.write("**HT:**")
            df_winrate_ht_dynamic = calcola_winrate(df_target, "risultato_ht")
            styled_df_ht = df_winrate_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ht)
            st.write("**FT:**")
            df_winrate_ft_dynamic = calcola_winrate(df_target, "risultato_ft")
            styled_df_ft = df_winrate_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ft)

            # Over Goals HT e FT
            col1, col2 = st.columns(2)
            df_target_goals["tot_goals_ht"] = df_target_goals["gol_home_ht"] + df_target_goals["gol_away_ht"]
            df_target_goals["tot_goals_ft"] = df_target_goals["gol_home_ft"] + df_target_goals["gol_away_ft"]
            
            with col1:
                st.subheader("Over Goals HT (Dinamica)")
                over_ht_data_dynamic = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_target_goals["tot_goals_ht"] > t).sum()
                    perc = round((count / len(df_target_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ht_data_dynamic.append([f"Over {t} HT", count, perc, odd_min])
                df_over_ht_dynamic = pd.DataFrame(over_ht_data_dynamic, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ht_dynamic = df_over_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ht_dynamic)
            
            with col2:
                st.subheader("Over Goals FT (Dinamica)")
                over_ft_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_target_goals["tot_goals_ft"] > t).sum()
                    perc = round((count / len(df_target_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ft)
            
            # BTTS
            st.subheader("BTTS (Dinamica)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_btts_ht_dynamic = calcola_btts_ht_dinamico(df_target)
                styled_df = df_btts_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### FT")
                df_btts_ft_dynamic = calcola_btts_dinamico(df_target, start_min, risultati_correnti)
                styled_df = df_btts_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            # Multi Gol
            st.subheader("Multi Gol (Dinamica)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Casa")
                styled_df = calcola_multi_gol(df_target, "gol_home_ft", "Home").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### Trasferta")
                styled_df = calcola_multi_gol(df_target, "gol_away_ft", "Away").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            # First to Score nell'analisi dinamica (HT e FT)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("First to Score HT (Dinamica)")
                styled_df = calcola_first_to_score_ht(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.subheader("First to Score FT (Dinamica)")
                styled_df = calcola_first_to_score(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            # To Score nell'analisi dinamica (HT e FT)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("To Score HT (Dinamica)")
                styled_df = calcola_to_score_ht(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.subheader("To Score FT (Dinamica)")
                styled_df = calcola_to_score(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            # Clean Sheet nell'analisi dinamica
            st.subheader("Clean Sheet (Dinamica)")
            styled_df = calcola_clean_sheet(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Combo Markets nell'analisi dinamica
            st.subheader("Combo Markets (Dinamica)")
            styled_df = calcola_combo_stats(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Next Goal nell'analisi dinamica
            st.subheader("Next Goal (Dinamica)")
            styled_df = calcola_next_goal(df_target, start_min, end_min).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Analisi Rimonte Dinamica
            st.subheader("Analisi Rimonte (Dinamica)")
            rimonte_stats, squadre_rimonte = calcola_rimonte(df_target, "Dinamica")
            if not rimonte_stats.empty:
                styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
                
                st.markdown("**Squadre che hanno effettuato rimonte:**")
                for tipo, squadre in squadre_rimonte.items():
                    if squadre:
                        st.markdown(f"**{tipo}:** {', '.join(squadre)}")
            else:
                st.warning("Nessuna rimonta trovata nel dataset filtrato per questa analisi dinamica.")
            
            # Qui viene mostrata la timeband basata sull'analisi dinamica
            st.subheader("Distribuzione Gol per Timeframe (dinamica)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**15min**")
                mostra_distribuzione_timeband(df_target)
            with col2:
                st.write("**5min**")
                mostra_distribuzione_timeband_5min(df_target)

    else:
        st.warning("Il dataset filtrato è vuoto o mancano le colonne necessarie per l'analisi.")

# --- SEZIONE 5: Analisi Head-to-Head (H2H) ---
st.subheader("5. Analisi Head-to-Head (H2H)")
st.write("Seleziona due squadre per analizzare i loro scontri diretti.")

# Recupera l'elenco completo di tutte le squadre disponibili nel dataset
all_teams = sorted(list(set(df['home_team'].dropna().unique()) | set(df['away_team'].dropna().unique())))
h2h_home_team = st.selectbox("Seleziona Squadra 1", ["Seleziona..."] + all_teams)
h2h_away_team = st.selectbox("Seleziona Squadra 2", ["Seleziona..."] + all_teams)

if h2h_home_team != "Seleziona..." and h2h_away_team != "Seleziona...":
    if h2h_home_team == h2h_away_team:
        st.warning("Seleziona due squadre diverse per l'analisi H2H.")
    else:
        # Filtra il DataFrame per trovare tutti i match tra le due squadre selezionate
        # NOTA: I filtri per le quote della sidebar non vengono applicati qui per avere il dataset H2H completo
        h2h_df = df[((df['home_team'] == h2h_home_team) & (df['away_team'] == h2h_away_team)) |
                    ((df['home_team'] == h2h_away_team) & (df['away_team'] == h2h_home_team))]
        
        if h2h_df.empty:
            st.warning(f"Nessuna partita trovata tra {h2h_home_team} e {h2h_away_team}.")
        else:
            st.write(f"Analisi basata su **{len(h2h_df)}** scontri diretti tra {h2h_home_team} e {h2h_away_team}.")

            # Esegui le stesse analisi pre-match, ma sul DataFrame H2H
            st.subheader(f"Statistiche H2H Complete tra {h2h_home_team} e {h2h_away_team}")
            
            # Media gol
            st.subheader("Media Gol (H2H)")
            df_h2h_goals = h2h_df.copy()
            df_h2h_goals["gol_home_ht"] = pd.to_numeric(df_h2h_goals["gol_home_ht"], errors='coerce')
            df_h2h_goals["gol_away_ht"] = pd.to_numeric(df_h2h_goals["gol_away_ht"], errors='coerce')
            df_h2h_goals["gol_home_ft"] = pd.to_numeric(df_h2h_goals["gol_home_ft"], errors='coerce')
            df_h2h_goals["gol_away_ft"] = pd.to_numeric(df_h2h_goals["gol_away_ft"], errors='coerce')
            avg_ht_goals = (df_h2h_goals["gol_home_ht"] + df_h2h_goals["gol_away_ht"]).mean()
            avg_ft_goals = (df_h2h_goals["gol_home_ft"] + df_h2h_goals["gol_away_ft"]).mean()
            avg_sh_goals = (df_h2h_goals["gol_home_ft"] + df_h2h_goals["gol_away_ft"] - df_h2h_goals["gol_home_ht"] - df_h2h_goals["gol_away_ht"]).mean()
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
            }))
            
            # Risultati Esatti H2H
            mostra_risultati_esatti(h2h_df, "risultato_ht", "HT H2H")
            mostra_risultati_esatti(h2h_df, "risultato_ft", "FT H2H")

            # WinRate H2H
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("WinRate HT H2H")
                df_winrate_ht_h2h = calcola_winrate(h2h_df, "risultato_ht")
                styled_df_ht = df_winrate_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
                st.dataframe(styled_df_ht)
            with col2:
                st.subheader("WinRate FT H2H")
                df_winrate_ft_h2h = calcola_winrate(h2h_df, "risultato_ft")
                styled_df_ft = df_winrate_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
                st.dataframe(styled_df_ft)

            # Over Goals H2H
            col1, col2 = st.columns(2)
            df_h2h_goals["tot_goals_ht"] = pd.to_numeric(df_h2h_goals["gol_home_ht"], errors='coerce') + pd.to_numeric(df_h2h_goals["gol_away_ht"], errors='coerce')
            df_h2h_goals["tot_goals_ft"] = pd.to_numeric(df_h2h_goals["gol_home_ft"], errors='coerce') + pd.to_numeric(df_h2h_goals["gol_away_ft"], errors='coerce')

            with col1:
                st.subheader("Over Goals HT H2H")
                over_ht_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_h2h_goals["tot_goals_ht"] > t).sum()
                    perc = round((count / len(df_h2h_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
                df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ht = df_over_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ht)

            with col2:
                st.subheader("Over Goals FT H2H")
                over_ft_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_h2h_goals["tot_goals_ft"] > t).sum()
                    perc = round((count / len(df_h2h_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ft)
            
            # BTTS H2H
            st.subheader("BTTS (H2H)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_btts_ht_h2h = calcola_btts_ht(h2h_df)
                styled_df = df_btts_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### FT")
                df_btts_ft_h2h = calcola_btts_ft(h2h_df)
                styled_df = df_btts_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            # Multi Gol H2H
            st.subheader("Multi Gol (H2H)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Casa")
                styled_df = calcola_multi_gol(h2h_df, "gol_home_ft", "Home").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### Trasferta")
                styled_df = calcola_multi_gol(h2h_df, "gol_away_ft", "Away").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            # First to Score H2H
            st.subheader("First to Score (H2H)")
            styled_df = calcola_first_to_score(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # To Score H2H
            st.subheader("To Score (H2H)")
            styled_df = calcola_to_score(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Clean Sheet H2H
            st.subheader("Clean Sheet (H2H)")
            styled_df = calcola_clean_sheet(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Combo Markets H2H
            st.subheader("Combo Markets (H2H)")
            styled_df = calcola_combo_stats(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Analisi Rimonte H2H
            st.subheader("Analisi Rimonte (H2H)")
            rimonte_stats, squadre_rimonte = calcola_rimonte(h2h_df, "H2H")
            if not rimonte_stats.empty:
                styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
                
                st.markdown("**Squadre che hanno effettuato rimonte:**")
                for tipo, squadre in squadre_rimonte.items():
                    if squadre:
                        st.markdown(f"**{tipo}:** {', '.join(squadre)}")
            else:
                st.warning("Nessuna rimonta trovata nel dataset filtrato.")


# --- SEZIONE 6: Backtesting Strategie ---
st.subheader("6. Backtesting Strategie")
st.write("Testa una strategia di scommesse sui dati filtrati.")

# Aggiungi un expander per contenere la logica di backtesting
with st.expander("Configura e avvia il Backtest"):
    
    if filtered_df.empty:
        st.warning("Il DataFrame filtrato è vuoto, non è possibile eseguire il backtest.")
    else:
        # Funzione per eseguire il backtest
        def esegui_backtest(df_to_analyze, market, strategy, stake):
            
            # Definizione dei mercati e delle colonne necessarie
            market_map = {
                "1 (Casa)": ("odd_home", lambda row: row["gol_home_ft"] > row["gol_away_ft"]),
                "X (Pareggio)": ("odd_draw", lambda row: row["gol_home_ft"] == row["gol_away_ft"]),
                "2 (Trasferta)": ("odd_away", lambda row: row["gol_home_ft"] < row["gol_away_ft"]),
                "Over 2.5 FT": ("odd_over_2_5", lambda row: (row["gol_home_ft"] + row["gol_away_ft"]) > 2.5),
                "BTTS SI FT": ("odd_btts_si", lambda row: (row["gol_home_ft"] > 0 and row["gol_away_ft"] > 0))
            }
            
            odd_col, win_condition = market_map[market]
            
            # Controllo che le colonne necessarie esistano nel DataFrame
            required_cols = [odd_col, "risultato_ft", "gol_home_ft", "gol_away_ft"]
            for col in required_cols:
                if col not in df_to_analyze.columns:
                    st.warning(f"Impossibile eseguire il backtest: la colonna '{col}' non è presente nel dataset.")
                    return 0, 0, 0, 0.0, 0.0, 0.0, 0.0
            
            vincite = 0
            perdite = 0
            profit_loss = 0.0
            numero_scommesse = 0
            
            # Rimuovi le righe con valori nulli nelle colonne chiave
            df_clean = df_to_analyze.dropna(subset=required_cols).copy()
            
            # Assicurati che le colonne quote e gol siano numeriche
            df_clean[odd_col] = pd.to_numeric(df_clean[odd_col].astype(str).str.replace(",", "."), errors='coerce').fillna(0)
            df_clean["gol_home_ft"] = pd.to_numeric(df_clean["gol_home_ft"], errors='coerce').fillna(0)
            df_clean["gol_away_ft"] = pd.to_numeric(df_clean["gol_away_ft"], errors='coerce').fillna(0)

            for _, row in df_clean.iterrows():
                try:
                    odd = row[odd_col]
                    
                    if odd > 0:
                        is_winning = win_condition(row)
                        
                        if strategy == "Back":
                            if is_winning:
                                vincite += 1
                                profit_loss += (odd - 1) * stake
                            else:
                                perdite += 1
                                profit_loss -= stake
                        elif strategy == "Lay":
                            if is_winning:
                                perdite += 1
                                profit_loss -= (odd - 1) * stake
                            else:
                                vincite += 1
                                profit_loss += stake
                        
                        numero_scommesse += 1
                
                except (ValueError, KeyError):
                    # Gestione di righe con dati mancanti o non validi
                    continue

            investimento_totale = numero_scommesse * stake
            roi = (profit_loss / investimento_totale) * 100 if investimento_totale > 0 else 0
            win_rate = (vincite / numero_scommesse) * 100 if numero_scommesse > 0 else 0
            odd_minima = 100 / win_rate if win_rate > 0 else 0
            
            return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima

        # UI per il backtest
        backtest_market = st.selectbox(
            "Seleziona un mercato da testare",
            ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)", "Over 2.5 FT", "BTTS SI FT"]
        )
        backtest_strategy = st.selectbox(
            "Seleziona la strategia",
            ["Back", "Lay"]
        )
        stake = st.number_input("Stake per scommessa", min_value=1.0, value=1.0, step=0.5)
        
        if st.button("Avvia Backtest"):
            vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima = esegui_backtest(filtered_df, backtest_market, backtest_strategy, stake)
            
            if numero_scommesse > 0:
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                col_met1.metric("Numero Scommesse", numero_scommesse)
                col_met2.metric("Vincite", vincite)
                col_met3.metric("Perdite", perdite)
                col_met4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
                
                col_met5, col_met6 = st.columns(2)
                col_met5.metric("ROI", f"{roi:.2f} %")
                col_met6.metric("Win Rate", f"{win_rate:.2f} %")
                st.metric("Odd Minima per profitto", f"{odd_minima:.2f}")
            elif numero_scommesse == 0:
                st.info("Nessuna scommessa idonea trovata con i filtri e il mercato selezionati.")

