import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import plotly.express as px

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
        partite_con_almeno_due_gol = 0
        
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            
            gol_nel_timeframe = [g for g in gol_home + gol_away if start <= g <= end]
            
            if len(gol_nel_timeframe) >= 1:
                partite_con_gol += 1
            if len(gol_nel_timeframe) >= 2:
                partite_con_almeno_due_gol += 1
        
        perc_1_gol = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        perc_2_gol = round((partite_con_almeno_due_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min_1_gol = round(100 / perc_1_gol, 2) if perc_1_gol > 0 else "-"
        odd_min_2_gol = round(100 / perc_2_gol, 2) if perc_2_gol > 0 else "-"

        risultati.append([label, partite_con_gol, perc_1_gol, partite_con_almeno_due_gol, perc_2_gol, odd_min_1_gol, odd_min_2_gol])
    
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con 1+ Gol", "% con 1+ Gol", "Partite con 2+ Gol", "% con 2+ Gol", "Odd Minima (1+ Gol)", "Odd Minima (2+ Gol)"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['% con 1+ Gol', '% con 2+ Gol'])
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
        partite_con_almeno_due_gol = 0
        
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            
            gol_nel_timeframe = [g for g in gol_home + gol_away if start <= g <= end]
            
            if len(gol_nel_timeframe) >= 1:
                partite_con_gol += 1
            if len(gol_nel_timeframe) >= 2:
                partite_con_almeno_due_gol += 1
        
        perc_1_gol = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        perc_2_gol = round((partite_con_almeno_due_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min_1_gol = round(100 / perc_1_gol, 2) if perc_1_gol > 0 else "-"
        odd_min_2_gol = round(100 / perc_2_gol, 2) if perc_2_gol > 0 else "-"

        risultati.append([label, partite_con_gol, perc_1_gol, partite_con_almeno_due_gol, perc_2_gol, odd_min_1_gol, odd_min_2_gol])
    
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con 1+ Gol", "% con 1+ Gol", "Partite con 2+ Gol", "% con 2+ Gol", "Odd Minima (1+ Gol)", "Odd Minima (2+ Gol)"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['% con 1+ Gol', '% con 2+ Gol'])
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

# --- FUNZIONE BACKTEST ---
def esegui_backtest(df, market, strategy, stake):
    vincite = 0
    perdite = 0
    profit_loss = 0.0

    # Determina le colonne delle quote in base al mercato
    odd_col = ""
    if market == "1 (Casa)":
        odd_col = "odd_home"
        risultato_col = "risultato_ft"
    elif market == "X (Pareggio)":
        odd_col = "odd_draw"
        risultato_col = "risultato_ft"
    elif market == "2 (Trasferta)":
        odd_col = "odd_away"
        risultato_col = "risultato_ft"
    elif market == "Over 2.5 FT":
        odd_col = "odd_over_2_5"
        risultato_col = "gol_home_ft" # Useremo le colonne dei gol per verificare il risultato
        gol_away_col = "gol_away_ft"
    elif market == "BTTS SI FT":
        odd_col = "odd_btts_si"
        risultato_col = "gol_home_ft"
        gol_away_col = "gol_away_ft"

    # Filtra i dati solo per le righe con quote valide
    df_backtest = df.dropna(subset=[odd_col])

    numero_scommesse = 0
    for _, row in df_backtest.iterrows():
        try:
            odd_stake = float(str(row[odd_col]).replace(",", "."))
            # Verifico se la scommessa è valida in base alle quote
            if strategy == "Back":
                # La scommessa Back è considerata valida se la quota è > 1
                if odd_stake > 1:
                    numero_scommesse += 1
                    is_win = False
                    if market in ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"]:
                        # Analisi per mercati 1, X, 2
                        home, away = map(int, str(row[risultato_col]).split("-"))
                        if market == "1 (Casa)" and home > away:
                            is_win = True
                        elif market == "X (Pareggio)" and home == away:
                            is_win = True
                        elif market == "2 (Trasferta)" and home < away:
                            is_win = True
                    elif market == "Over 2.5 FT":
                        # Analisi per Over 2.5
                        gol_home = int(row[risultato_col])
                        gol_away = int(row[gol_away_col])
                        if (gol_home + gol_away) > 2.5:
                            is_win = True
                    elif market == "BTTS SI FT":
                        # Analisi per BTTS SI
                        gol_home = int(row[risultato_col])
                        gol_away = int(row[gol_away_col])
                        if gol_home > 0 and gol_away > 0:
                            is_win = True

                    if is_win:
                        vincite += 1
                        profit_loss += (odd_stake - 1) * stake
                    else:
                        perdite += 1
                        profit_loss -= stake
            
            elif strategy == "Lay":
                # La scommessa Lay è considerata valida se la quota è > 1
                if odd_stake > 1:
                    numero_scommesse += 1
                    is_win = False
                    if market in ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"]:
                        # Analisi per mercati 1, X, 2 in modalità Lay
                        home, away = map(int, str(row[risultato_col]).split("-"))
                        if market == "1 (Casa)" and home < away or home == away: # Se non vince la casa, vinco la scommessa Lay
                            is_win = True
                        elif market == "X (Pareggio)" and home != away:
                            is_win = True
                        elif market == "2 (Trasferta)" and home > away or home == away:
                            is_win = True
                    elif market == "Over 2.5 FT":
                        # Analisi per Lay Over 2.5
                        gol_home = int(row[risultato_col])
                        gol_away = int(row[gol_away_col])
                        if (gol_home + gol_away) <= 2.5:
                            is_win = True
                    elif market == "BTTS SI FT":
                        # Analisi per Lay BTTS SI
                        gol_home = int(row[risultato_col])
                        gol_away = int(row[gol_away_col])
                        if gol_home == 0 or gol_away == 0:
                            is_win = True
                    
                    if is_win:
                        vincite += 1
                        profit_loss += stake # La vincita nel Lay è lo stake
                    else:
                        perdite += 1
                        profit_loss -= (odd_stake - 1) * stake # La perdita è la quota meno 1
        except (ValueError, IndexError):
            # Gestisce i casi in cui i dati non sono validi
            continue

    roi = (profit_loss / (numero_scommesse * stake)) * 100 if (numero_scommesse * stake) > 0 else 0
    win_rate = (vincite / numero_scommesse) * 100 if numero_scommesse > 0 else 0
    odd_minima = round(100 / win_rate, 2) if win_rate > 0 else "-"

    return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima

# --- FUNZIONE per Analisi Pre-Match e Dinamica ---
def mostra_analisi_dinamica_prematch(filtered_df):
    st.subheader("Analisi Statistiche Pre-Match e Dinamiche")
    
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)

    # Conteggio attacchi e attacchi pericolosi
    attacchi_home = filtered_df["attacchi_home"].sum()
    attacchi_away = filtered_df["attacchi_away"].sum()
    pericolosi_home = filtered_df["attacchi_pericolosi_home"].sum()
    pericolosi_away = filtered_df["attacchi_pericolosi_away"].sum()

    col_pre1.metric("Attacchi Totali (Home)", attacchi_home)
    col_pre2.metric("Attacchi Totali (Away)", attacchi_away)
    col_pre3.metric("Attacchi Pericolosi (Home)", pericolosi_home)
    col_pre4.metric("Attacchi Pericolosi (Away)", pericolosi_away)

    # % di attacchi pericolosi su attacchi totali
    perc_pericolosi_home = round((pericolosi_home / attacchi_home) * 100, 2) if attacchi_home > 0 else 0
    perc_pericolosi_away = round((pericolosi_away / attacchi_away) * 100, 2) if attacchi_away > 0 else 0

    st.subheader("Percentuale Attacchi Pericolosi")
    col_perc1, col_perc2 = st.columns(2)
    col_perc1.metric("% Pericolosi (Home)", f"{perc_pericolosi_home}%")
    col_perc2.metric("% Pericolosi (Away)", f"{perc_pericolosi_away}%")

    # Tiri totali
    st.subheader("Analisi Tiri Totali")
    tiri_totali_home = filtered_df["tiri_tot_home"].sum()
    tiri_totali_away = filtered_df["tiri_tot_away"].sum()
    tiri_in_porta_home = filtered_df["tiri_in_porta_home"].sum()
    tiri_in_porta_away = filtered_df["tiri_in_porta_away"].sum()

    col_tiri1, col_tiri2, col_tiri3, col_tiri4 = st.columns(4)
    col_tiri1.metric("Tiri Totali (Home)", tiri_totali_home)
    col_tiri2.metric("Tiri Totali (Away)", tiri_totali_away)
    col_tiri3.metric("Tiri in Porta (Home)", tiri_in_porta_home)
    col_tiri4.metric("Tiri in Porta (Away)", tiri_in_porta_away)
    
    # % tiri in porta su tiri totali
    perc_in_porta_home = round((tiri_in_porta_home / tiri_totali_home) * 100, 2) if tiri_totali_home > 0 else 0
    perc_in_porta_away = round((tiri_in_porta_away / tiri_totali_away) * 100, 2) if tiri_totali_away > 0 else 0

    st.subheader("Percentuale Tiri in Porta")
    col_tiri_perc1, col_tiri_perc2 = st.columns(2)
    col_tiri_perc1.metric("% Tiri in Porta (Home)", f"{perc_in_porta_home}%")
    col_tiri_perc2.metric("% Tiri in Porta (Away)", f"{perc_in_porta_away}%")


# --- ESECUZIONE DELLE ANALISI ---
st.header("Analisi Dati")
if not filtered_df.empty:
    st.subheader("Winrate Risultato Finale")
    st.dataframe(calcola_winrate(filtered_df, "risultato_ft").style.background_gradient(cmap='RdYlGn', subset=['WinRate %']))

    st.subheader("Winrate Risultato Primo Tempo")
    st.dataframe(calcola_winrate(filtered_df, "risultato_ht").style.background_gradient(cmap='RdYlGn', subset=['WinRate %']))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("First to Score FT")
        st.dataframe(calcola_first_to_score(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

    with col2:
        st.subheader("First to Score HT")
        st.dataframe(calcola_first_to_score_ht(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

    mostra_risultati_esatti(filtered_df, "risultato_ft", "FT")
    mostra_risultati_esatti(filtered_df, "risultato_ht", "HT")
    
    st.subheader("Distribuzione Gol per Fascia (15 min)")
    mostra_distribuzione_timeband(filtered_df)

    st.subheader("Distribuzione Gol per Fascia (5 min)")
    mostra_distribuzione_timeband_5min(filtered_df)

    # Analisi Dinamica e Pre-match (sezione mancante precedentemente)
    mostra_analisi_dinamica_prematch(filtered_df)


    # UI per l'analisi Next Goal
    st.header("Analisi Next Gol")
    col_next1, col_next2 = st.columns(2)
    with col_next1:
        start_min = st.number_input("Inizio Minuto (Next Goal)", min_value=0, value=75)
    with col_next2:
        end_min = st.number_input("Fine Minuto (Next Goal)", min_value=1, value=90)

    if st.button("Calcola Next Gol"):
        st.dataframe(calcola_next_goal(filtered_df, start_min, end_min).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
    
    # UI per l'analisi delle rimonte
    st.header("Analisi Rimonte")
    rimonte_stats, squadre_rimonte = calcola_rimonte(filtered_df, "Analisi Rimonte")
    st.subheader("Statistiche Rimonte")
    styled_df_rimonte = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df_rimonte)

    # Backtest
    st.header("Backtest")
    if "odd_home" in filtered_df.columns and "odd_draw" in filtered_df.columns and "odd_away" in filtered_df.columns:
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
            if not filtered_df.empty:
                vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima = esegui_backtest(filtered_df, backtest_market, backtest_strategy, stake)
                
                if numero_scommesse > 0:
                    st.success("Backtest completato con successo!")
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    col_met1.metric("Numero Scommesse", numero_scommesse)
                    col_met2.metric("Vincite", vincite)
                    col_met3.metric("Perdite", perdite)
                    col_met4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
                    
                    col_met5, col_met6, col_met7 = st.columns(3)
                    col_met5.metric("ROI", f"{roi:.2f}%")
                    col_met6.metric("Win Rate", f"{win_rate:.2f}%")
                    col_met7.metric("Odd Minima Teorica", odd_minima)
                else:
                    st.warning("Nessuna scommessa piazzata con i filtri e le impostazioni selezionate.")
            else:
                st.warning("Il DataFrame filtrato è vuoto. Impossibile eseguire il backtest.")
    else:
        st.warning("Le colonne delle quote (odd_home, odd_draw, odd_away, ecc.) non sono presenti nel DataFrame. Impossibile eseguire il backtest.")

