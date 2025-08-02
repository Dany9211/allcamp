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

# --- RISULTATI ESATTI ---
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
    st.table(distribuzione)

# --- FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (15 MIN) ---
def mostra_distribuzione_timeband(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return
    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end) in intervalli:
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([f"{start}-{end}", partite_con_gol, perc, odd_min])
    st.table(pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"]))

# --- NUOVA FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
def mostra_distribuzione_timeband_5min(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 5 minuti è vuoto.")
        return
    intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90)]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end) in intervalli:
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([f"{start}-{end}", partite_con_gol, perc, odd_min])
    st.table(pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"]))


# --- SEZIONE 1: Analisi Timeband per Campionato ---
st.subheader("1. Analisi Timeband per Campionato")
if selected_league != "Tutte":
    df_league_only = df[df["league"] == selected_league]
    st.write(f"Analisi basata su **{len(df_league_only)}** partite del campionato **{selected_league}**.")
    st.write("---")
    st.write("**Distribuzione Gol per Timeframe (15min)**")
    mostra_distribuzione_timeband(df_league_only)
    st.write("**Distribuzione Gol per Timeframe (5min)**")
    mostra_distribuzione_timeband_5min(df_league_only)
else:
    st.write("Seleziona un campionato per visualizzare questa analisi.")

# --- SEZIONE 2: Analisi Timeband per Campionato e Quote ---
st.subheader("2. Analisi Timeband per Campionato e Quote")
st.write(f"Analisi basata su **{len(filtered_df)}** partite filtrate da tutti i parametri della sidebar.")
if not filtered_df.empty:
    st.write("---")
    st.write("**Distribuzione Gol per Timeframe (15min)**")
    mostra_distribuzione_timeband(filtered_df)
    st.write("**Distribuzione Gol per Timeframe (5min)**")
    mostra_distribuzione_timeband_5min(filtered_df)
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati.")

# --- NUOVA SEZIONE: Statistiche Pre-Match HT Over/Under ---
st.subheader("3. Statistiche Pre-Match HT")
# Crea un DataFrame per l'analisi pre-match, escludendo il filtro 'risultato_ht'
df_prematch_ht = df.copy()
prematch_filters = {k: v for k, v in filters.items() if k != 'risultato_ht'}
for col, val in prematch_filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        mask = pd.to_numeric(df_prematch_ht[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
        df_prematch_ht = df_prematch_ht[mask.fillna(True)]
    elif col == "giornata":
        mask = pd.to_numeric(df_prematch_ht[col], errors="coerce").between(val[0], val[1])
        df_prematch_ht = df_prematch_ht[mask.fillna(True)]
    else:
        df_prematch_ht = df_prematch_ht[df_prematch_ht[col] == val]

if not df_prematch_ht.empty:
    st.write(f"Analisi basata su **{len(df_prematch_ht)}** partite (escluso il filtro Risultato HT).")
    df_prematch_ht["tot_goals_ht"] = pd.to_numeric(df_prematch_ht["gol_home_ht"], errors='coerce') + pd.to_numeric(df_prematch_ht["gol_away_ht"], errors='coerce')
    
    over_ht_data = []
    for t in [0.5, 1.5, 2.5]:
        count = (df_prematch_ht["tot_goals_ht"] > t).sum()
        perc = round((count / len(df_prematch_ht)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
    st.table(pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per l'analisi pre-match.")
    

# --- SEZIONE 4: Analisi Timeband Dinamica (Minuto/Risultato) ---
st.subheader("4. Analisi Timeband Dinamica")
with st.expander("Mostra Analisi Dinamica (Minuto/Risultato)"):
    if not filtered_df.empty:
        # --- ANALISI DAL MINUTO (integrata) ---
        start_min, end_min = st.slider("Seleziona intervallo minuti", 1, 90, (20, 45))
        risultati_correnti = st.multiselect("Risultato corrente al minuto iniziale",
                                            ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2"], default=["0-0"])

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

            # --- Calcolo gol nel range ---
            def conta_gol_range(row):
                gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                home_in_range = sum(1 for g in gol_home if start_min <= g <= end_min)
                away_in_range = sum(1 for g in gol_away if start_min <= g <= end_min)
                return home_in_range, away_in_range

            df_target[["home_range", "away_range"]] = df_target.apply(lambda row: pd.Series(conta_gol_range(row)), axis=1)
            df_target["tot_goals_range"] = df_target["home_range"] + df_target["away_range"]

            mostra_risultati_esatti(df_target, "risultato_ft", "FT")

            st.subheader(f"WinRate (Range {start_min}-{end_min})")
            st.write("**HT:**")
            st.table(calcola_winrate(df_target, "risultato_ht"))
            st.write("**FT:**")
            st.table(calcola_winrate(df_target, "risultato_ft"))

            st.subheader(f"Over Goals (Range {start_min}-{end_min})")
            over_data = []
            for t in [0.5, 1.5, 2.5]:
                count = (df_target["tot_goals_range"] > t).sum()
                perc = round((count / len(df_target)) * 100, 2)
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                over_data.append([f"Over {t} (range)", count, perc, odd_min])
            st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

            btts = ((df_target["home_range"] > 0) & (df_target["away_range"] > 0)).sum()
            perc_btts = round(btts / len(df_target) * 100, 2)
            odd_btts = round(100 / perc_btts, 2) if perc_btts > 0 else "-"
            st.subheader(f"BTTS SI (Range {start_min}-{end_min})")
            st.write(f"BTTS SI: {btts} ({perc_btts}%) - Odd Minima: {odd_btts}")
            
            # Qui viene mostrata la timeband basata sull'analisi dinamica
            st.subheader("Distribuzione Gol per Timeframe (15min) **(dinamica)**")
            mostra_distribuzione_timeband(df_target)
            st.subheader("Distribuzione Gol per Timeframe (5min) **(dinamica)**")
            mostra_distribuzione_timeband_5min(df_target)

    else:
        st.warning("Il dataset filtrato è vuoto o mancano le colonne necessarie per l'analisi.")
