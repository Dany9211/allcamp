Certo, ecco il codice completo e modificato per includere la nuova funzionalità. Ho aggiornato le funzioni `mostra_distribuzione_timeband` (15 minuti) e `mostra_distribuzione_timeband_5min` (5 minuti) per aggiungere la colonna che calcola la percentuale di timeband con almeno 2 gol.

Le modifiche sono evidenziate con un commento che indica **"MODIFICA RICHIESTA: Aggiunta colonna % Timeband con \>= 2 gol"**.

```python
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
# MODIFICA RICHIESTA: Aggiunta colonna % Timeband con >= 2 gol
def mostra_distribuzione_timeband(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return
        
    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 150)]
    label_intervalli = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    risultati = []
    partite_con_almeno_due_gol = []
    totale_partite = len(df_to_analyze)

    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        partite_due_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            
            gol_in_timeband = [g for g in gol_home + gol_away if start <= g <= end]

            if len(gol_in_timeband) > 0:
                partite_con_gol += 1
            if len(gol_in_timeband) >= 2:
                partite_due_gol += 1
        
        perc_con_gol = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        perc_due_gol = round((partite_due_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        
        odd_min_con_gol = round(100 / perc_con_gol, 2) if perc_con_gol > 0 else "-"
        odd_min_due_gol = round(100 / perc_due_gol, 2) if perc_due_gol > 0 else "-"
        
        risultati.append([label, partite_con_gol, perc_con_gol, odd_min_con_gol])
        partite_con_almeno_due_gol.append([label, partite_due_gol, perc_due_gol, odd_min_due_gol])

    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    df_due_gol_result = pd.DataFrame(partite_con_almeno_due_gol, columns=["Timeframe", "Partite con >= 2 Gol", "% con >= 2 Gol", "Odd Minima"])
    
    # Merge dei due DataFrame
    df_final = pd.merge(df_result, df_due_gol_result[["Timeframe", "% con >= 2 Gol"]], on="Timeframe", how="left")

    styled_df = df_final.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %', '% con >= 2 Gol'])
    st.dataframe(styled_df)


# --- FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
# MODIFICA RICHIESTA: Aggiunta colonna % Timeband con >= 2 gol
def mostra_distribuzione_timeband_5min(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 5 minuti è vuoto.")
        return
    
    intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90), (91, 150)]
    label_intervalli = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "90+"]
    risultati = []
    partite_con_almeno_due_gol = []
    totale_partite = len(df_to_analyze)

    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        partite_due_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            
            gol_in_timeband = [g for g in gol_home + gol_away if start <= g <= end]

            if len(gol_in_timeband) > 0:
                partite_con_gol += 1
            if len(gol_in_timeband) >= 2:
                partite_due_gol += 1
        
        perc_con_gol = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        perc_due_gol = round((partite_due_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        
        odd_min_con_gol = round(100 / perc_con_gol, 2) if perc_con_gol > 0 else "-"
        odd_min_due_gol = round(100 / perc_due_gol, 2) if perc_due_gol > 0 else "-"
        
        risultati.append([label, partite_con_gol, perc_con_gol, odd_min_con_gol])
        partite_con_almeno_due_gol.append([label, partite_due_gol, perc_due_gol, odd_min_due_gol])

    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    df_due_gol_result = pd.DataFrame(partite_con_almeno_due_gol, columns=["Timeframe", "Partite con >= 2 Gol", "% con >= 2 Gol", "Odd Minima"])
    
    # Merge dei due DataFrame
    df_final = pd.merge(df_result, df_due_gol_result[["Timeframe", "% con >= 2 Gol"]], on="Timeframe", how="left")

    styled_df = df_final.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %', '% con >= 2 Gol'])
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
    
    st.write(f"Media Gol Totali per Partita (HT): **{avg_ht_goals:.2f}**")
    st.write(f"Media Gol Totali per Partita (FT): **{avg_ft_goals:.2f}**")
    st.write("---")
    
    # Calcolo e visualizzazione WinRate e risultati esatti
    st.subheader("Analisi Esiti Finali e Parziali")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1X2 (FT)**")
        st.dataframe(calcola_winrate(filtered_df, "risultato_ft"))
    with col2:
        st.markdown("**1X2 (HT)**")
        st.dataframe(calcola_winrate(filtered_df, "risultato_ht"))
    with col3:
        st.markdown("**First to Score (FT)**")
        st.dataframe(calcola_first_to_score(filtered_df))
        st.markdown("**First to Score (HT)**")
        st.dataframe(calcola_first_to_score_ht(filtered_df))

    st.write("---")
    
    # Risultati esatti
    col1, col2 = st.columns(2)
    with col1:
        mostra_risultati_esatti(filtered_df, "risultato_ht", "HT")
    with col2:
        mostra_risultati_esatti(filtered_df, "risultato_ft", "FT")

    st.write("---")

    # Over/Under e BTTS
    st.subheader("Analisi Over/Under e BTTS")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Over/Under FT**")
        df_over_under_ft = pd.DataFrame([
            ["Over 0.5", (filtered_df["gol_home_ft"] + filtered_df["gol_away_ft"] > 0.5).sum()],
            ["Over 1.5", (filtered_df["gol_home_ft"] + filtered_df["gol_away_ft"] > 1.5).sum()],
            ["Over 2.5", (filtered_df["gol_home_ft"] + filtered_df["gol_away_ft"] > 2.5).sum()],
            ["Over 3.5", (filtered_df["gol_home_ft"] + filtered_df["gol_away_ft"] > 3.5).sum()],
            ["Over 4.5", (filtered_df["gol_home_ft"] + filtered_df["gol_away_ft"] > 4.5).sum()],
            ["Over 5.5", (filtered_df["gol_home_ft"] + filtered_df["gol_away_ft"] > 5.5).sum()]
        ], columns=["Mercato", "Conteggio"])
        df_over_under_ft["Percentuale %"] = (df_over_under_ft["Conteggio"] / len(filtered_df) * 100).round(2)
        df_over_under_ft["Odd Minima"] = df_over_under_ft["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
        st.dataframe(df_over_under_ft)
    
    with col2:
        st.markdown("**Over/Under HT**")
        df_over_under_ht = pd.DataFrame([
            ["Over 0.5", (filtered_df["gol_home_ht"] + filtered_df["gol_away_ht"] > 0.5).sum()],
            ["Over 1.5", (filtered_df["gol_home_ht"] + filtered_df["gol_away_ht"] > 1.5).sum()],
            ["Over 2.5", (filtered_df["gol_home_ht"] + filtered_df["gol_away_ht"] > 2.5).sum()]
        ], columns=["Mercato", "Conteggio"])
        df_over_under_ht["Percentuale %"] = (df_over_under_ht["Conteggio"] / len(filtered_df) * 100).round(2)
        df_over_under_ht["Odd Minima"] = df_over_under_ht["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
        st.dataframe(df_over_under_ht)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**BTTS HT**")
        st.dataframe(calcola_btts_ht(filtered_df))
    with col4:
        st.markdown("**BTTS FT**")
        st.dataframe(calcola_btts_ft(filtered_df))

    st.write("---")
    
    # Clean Sheet e Combo Markets
    st.subheader("Analisi Clean Sheet e Combo Markets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Clean Sheet (FT)**")
        st.dataframe(calcola_clean_sheet(filtered_df))
    with col2:
        st.markdown("**Combo Markets**")
        st.dataframe(calcola_combo_stats(filtered_df))
        
    st.write("---")
    
    # Multi Gol
    st.subheader("Analisi Multi Gol")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Multi Gol Home Team**")
        st.dataframe(calcola_multi_gol(filtered_df, "gol_home_ft", "Home"))
    with col2:
        st.markdown("**Multi Gol Away Team**")
        st.dataframe(calcola_multi_gol(filtered_df, "gol_away_ft", "Away"))
        
    st.write("---")
    
    # Rimonte
    st.subheader("Analisi Rimonte (Comebacks)")
    df_rimonte, _ = calcola_rimonte(filtered_df, "Rimonte")
    st.dataframe(df_rimonte)
    
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati.")

# --- SEZIONE 4: Analisi Dinamica (Live) ---
st.subheader("4. Analisi Dinamica (Live)")

# Filtri per l'analisi dinamica
st.sidebar.header("Filtri per Analisi Dinamica")
st.sidebar.markdown("Filtra il DataFrame in base allo stato della partita in un minuto specifico.")

if not filtered_df.empty:
    minuto_attuale = st.sidebar.slider("Minuto Attuale", 0, 90, 45)
    risultato_ht = st.sidebar.selectbox("Risultato HT (per analisi dinamica)", sorted(filtered_df["risultato_ht"].dropna().unique()), index=1)

    df_dinamico = filtered_df[filtered_df["risultato_ht"] == risultato_ht]
    st.write(f"Analisi Dinamica basata su **{len(df_dinamico)}** partite con risultato HT di **{risultato_ht}**")

    if not df_dinamico.empty:
        # Analisi Next Goal
        st.subheader("Next Goal (Dinamico)")
        next_goal_df = calcola_next_goal(df_dinamico, minuto_attuale + 1, 90)
        st.dataframe(next_goal_df)

        # Analisi media gol dinamica
        st.subheader("Media Gol (Dinamica)")
        df_dynamic_goals = df_dinamico.copy()
        df_dynamic_goals["gol_home_ft"] = pd.to_numeric(df_dynamic_goals["gol_home_ft"], errors='coerce')
        df_dynamic_goals["gol_away_ft"] = pd.to_numeric(df_dynamic_goals["gol_away_ft"], errors='coerce')
        avg_ft_goals = (df_dynamic_goals["gol_home_ft"] + df_dynamic_goals["gol_away_ft"]).mean()
        st.write(f"Media Gol Totali per Partita (FT): **{avg_ft_goals:.2f}**")
        
        # Analisi Winrate dinamica
        st.subheader("Analisi 1X2 (Dinamica)")
        st.dataframe(calcola_winrate(df_dinamico, "risultato_ft"))

        # Analisi Over/Under dinamica
        st.subheader("Analisi Over/Under (Dinamica)")
        df_dynamic_over_under = pd.DataFrame([
            ["Over 0.5", (df_dinamico["gol_home_ft"] + df_dinamico["gol_away_ft"] > 0.5).sum()],
            ["Over 1.5", (df_dinamico["gol_home_ft"] + df_dinamico["gol_away_ft"] > 1.5).sum()],
            ["Over 2.5", (df_dinamico["gol_home_ft"] + df_dinamico["gol_away_ft"] > 2.5).sum()],
        ], columns=["Mercato", "Conteggio"])
        df_dynamic_over_under["Percentuale %"] = (df_dynamic_over_under["Conteggio"] / len(df_dinamico) * 100).round(2)
        df_dynamic_over_under["Odd Minima"] = df_dynamic_over_under["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
        st.dataframe(df_dynamic_over_under)
        
        # Analisi BTTS dinamica
        st.subheader("Analisi BTTS (Dinamica)")
        st.dataframe(calcola_btts_dinamico(df_dinamico, minuto_attuale, [risultato_ht]))
        
        # Analisi rimonte dinamica
        st.subheader("Analisi Rimonte (Dinamica)")
        df_dynamic_rimonte, _ = calcola_rimonte(df_dinamico, "Rimonte")
        st.dataframe(df_dynamic_rimonte)
        
    else:
        st.warning("Nessuna partita corrisponde ai filtri dinamici selezionati.")

# --- SEZIONE 5: Analisi Head-to-Head (H2H) ---
st.subheader("5. Analisi Head-to-Head (H2H)")
st.write("Confronta le statistiche dirette tra due squadre, ignorando i filtri di quote e giornata.")
if "home_team" in df.columns:
    teams_h2h = sorted(df["home_team"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        h2h_home = st.selectbox("Seleziona Squadra A (H2H)", teams_h2h)
    with col2:
        h2h_away = st.selectbox("Seleziona Squadra B (H2H)", teams_h2h)

    if h2h_home and h2h_away:
        df_h2h = df[((df["home_team"] == h2h_home) & (df["away_team"] == h2h_away)) | 
                    ((df["home_team"] == h2h_away) & (df["away_team"] == h2h_home))]

        if not df_h2h.empty:
            st.write(f"Analisi H2H per **{h2h_home}** vs **{h2h_away}** (totale: {len(df_h2h)} partite)")
            
            # Calcolo e visualizzazione media gol
            st.subheader("Media Gol (H2H)")
            df_h2h_goals = df_h2h.copy()
            df_h2h_goals["gol_home_ft"] = pd.to_numeric(df_h2h_goals["gol_home_ft"], errors='coerce')
            df_h2h_goals["gol_away_ft"] = pd.to_numeric(df_h2h_goals["gol_away_ft"], errors='coerce')
            avg_h2h_goals = (df_h2h_goals["gol_home_ft"] + df_h2h_goals["gol_away_ft"]).mean()
            st.write(f"Media Gol Totali per Partita (H2H): **{avg_h2h_goals:.2f}**")
            
            # Analisi 1X2 H2H
            st.subheader("Analisi 1X2 (H2H)")
            st.dataframe(calcola_winrate(df_h2h, "risultato_ft"))
            
        else:
            st.warning("Nessuna partita H2H trovata tra le due squadre selezionate.")

# --- SEZIONE 6: Backtesting Strategie ---
st.subheader("6. Backtesting Strategie")

if not filtered_df.empty:
    strategy_options = {
        "1X2": ["1 (Home)", "X (Draw)", "2 (Away)"],
        "Over/Under": ["Over 2.5", "Under 2.5"],
        "BTTS": ["BTTS Yes", "BTTS No"]
    }
    
    selected_market = st.selectbox("Seleziona Mercato per Backtest", list(strategy_options.keys()))
    selected_outcome = st.selectbox(f"Seleziona Esito per {selected_market}", strategy_options[selected_market])
    stake = st.number_input("Puntata per partita", min_value=1.0, value=10.0, step=0.5)
    
    if st.button("Esegui Backtest"):
        if selected_market == "1X2":
            wins = 0
            if selected_outcome == "1 (Home)":
                wins = len(filtered_df[filtered_df["gol_home_ft"] > filtered_df["gol_away_ft"]])
            elif selected_outcome == "X (Draw)":
                wins = len(filtered_df[filtered_df["gol_home_ft"] == filtered_df["gol_away_ft"]])
            elif selected_outcome == "2 (Away)":
                wins = len(filtered_df[filtered_df["gol_away_ft"] > filtered_df["gol_home_ft"]])
            
            total_matches = len(filtered_df)
            total_stake = total_matches * stake
            profit = (wins * stake) - total_stake
            
            st.write(f"**Risultati Backtest per {selected_market} - {selected_outcome}**")
            st.write(f"Partite analizzate: {total_matches}")
            st.write(f"Vittorie: {wins}")
            st.write(f"Percentuale di successo: {round((wins/total_matches)*100, 2)}%")
            st.write(f"Investimento totale: {total_stake:.2f} €")
            st.write(f"Profitto/Perdita: {profit:.2f} €")
        
        elif selected_market == "Over/Under":
            wins = 0
            df_temp = filtered_df.copy()
            df_temp["total_goals"] = df_temp["gol_home_ft"] + df_temp["gol_away_ft"]
            
            if selected_outcome == "Over 2.5":
                wins = len(df_temp[df_temp["total_goals"] > 2.5])
            elif selected_outcome == "Under 2.5":
                wins = len(df_temp[df_temp["total_goals"] < 2.5])
            
            total_matches = len(filtered_df)
            total_stake = total_matches * stake
            profit = (wins * stake) - total_stake
            
            st.write(f"**Risultati Backtest per {selected_market} - {selected_outcome}**")
            st.write(f"Partite analizzate: {total_matches}")
            st.write(f"Vittorie: {wins}")
            st.write(f"Percentuale di successo: {round((wins/total_matches)*100, 2)}%")
            st.write(f"Investimento totale: {total_stake:.2f} €")
            st.write(f"Profitto/Perdita: {profit:.2f} €")
        
        elif selected_market == "BTTS":
            wins = 0
            df_temp = filtered_df.copy()
            df_temp["btts"] = (df_temp["gol_home_ft"] > 0) & (df_temp["gol_away_ft"] > 0)
            
            if selected_outcome == "BTTS Yes":
                wins = len(df_temp[df_temp["btts"] == True])
            elif selected_outcome == "BTTS No":
                wins = len(df_temp[df_temp["btts"] == False])
            
            total_matches = len(filtered_df)
            total_stake = total_matches * stake
            profit = (wins * stake) - total_stake
            
            st.write(f"**Risultati Backtest per {selected_market} - {selected_outcome}**")
            st.write(f"Partite analizzate: {total_matches}")
            st.write(f"Vittorie: {wins}")
            st.write(f"Percentuale di successo: {round((wins/total_matches)*100, 2)}%")
            st.write(f"Investimento totale: {total_stake:.2f} €")
            st.write(f"Profitto/Perdita: {profit:.2f} €")
            
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per il backtesting.")
```
