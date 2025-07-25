import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Allcamp Viewer", layout="wide")
st.title("Analisi Avanzata allcamp con Range Minuti")

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

# --- Slider Range Minuti ---
st.sidebar.header("Analisi Minuto e Risultato")
minuto_da, minuto_a = st.sidebar.slider("Seleziona Intervallo Minuti", 1, 90, (20, 45))
risultati_possibili = ["Tutti"] + sorted(df["risultato_ht"].dropna().unique())
risultato_corrente = st.sidebar.selectbox("Seleziona Risultato Corrente (HT)", risultati_possibili)

# --- Filtri Classici ---
filters = {}
if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona League", leagues)
    if selected_league != "Tutte":
        filters["league"] = selected_league

if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
    if selected_anno != "Tutti":
        filters["anno"] = selected_anno

if "giornata" in df.columns:
    giornata_min = int(df["giornata"].min())
    giornata_max = int(df["giornata"].max())
    giornata_range = st.sidebar.slider("Seleziona Giornata", giornata_min, giornata_max, (giornata_min, giornata_max))
    filters["giornata"] = giornata_range

# --- Funzione applica filtri ---
filtered_df = df.copy()
for col, val in filters.items():
    if col == "giornata":
        mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

# --- Analisi Gol nel range minuti ---
def analizza_gol_range(df, da_minuto, a_minuto, risultato_sel):
    st.subheader(f"Analisi Gol tra {da_minuto} e {a_minuto} min")

    # Filtro su risultato corrente
    if risultato_sel != "Tutti":
        df = df[df["risultato_ht"] == risultato_sel]

    partite = len(df)
    partite_con_gol = 0

    for _, row in df.iterrows():
        home_goals = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        away_goals = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
        goals = [g for g in home_goals + away_goals if da_minuto <= g <= a_minuto]
        if goals:
            partite_con_gol += 1

    perc = round((partite_con_gol / partite) * 100, 2) if partite > 0 else 0
    st.write(f"Partite con almeno 1 gol tra {da_minuto} e {a_minuto} min: **{partite_con_gol}/{partite} ({perc}%)**")

    return df

# --- Over & BTTS ---
def calcola_over_btts(df):
    st.subheader(f"Over Goals e BTTS")
    risultati = {"Over 0.5 HT": 0, "Over 1.5 HT": 0, "Over 2.5 HT": 0,
                 "Over 0.5 FT": 0, "Over 1.5 FT": 0, "Over 2.5 FT": 0,
                 "Over 3.5 FT": 0, "Over 4.5 FT": 0, "BTTS SI": 0}
    total = len(df)

    for _, row in df.iterrows():
        ht_home = row.get("gol_home_ht", 0)
        ht_away = row.get("gol_away_ht", 0)
        ft_home = row.get("gol_home_ft", 0)
        ft_away = row.get("gol_away_ft", 0)

        if ht_home + ht_away > 0:
            risultati["Over 0.5 HT"] += 1
        if ht_home + ht_away > 1:
            risultati["Over 1.5 HT"] += 1
        if ht_home + ht_away > 2:
            risultati["Over 2.5 HT"] += 1
        if ft_home + ft_away > 0:
            risultati["Over 0.5 FT"] += 1
        if ft_home + ft_away > 1:
            risultati["Over 1.5 FT"] += 1
        if ft_home + ft_away > 2:
            risultati["Over 2.5 FT"] += 1
        if ft_home + ft_away > 3:
            risultati["Over 3.5 FT"] += 1
        if ft_home + ft_away > 4:
            risultati["Over 4.5 FT"] += 1
        if ft_home > 0 and ft_away > 0:
            risultati["BTTS SI"] += 1

    tab = pd.DataFrame(
        [[k, v, round((v/total)*100, 2) if total > 0 else 0, round(100/(v/total*100), 2) if v > 0 else "-"]
         for k, v in risultati.items()],
        columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]
    )
    st.table(tab)

# --- Winrate HT e FT ---
def calcola_winrate_ht_ft(df):
    st.subheader("Winrate HT e FT (Home-Draw-Away)")
    def calcola_esiti(col_ris):
        esiti = {"1": 0, "X": 0, "2": 0}
        valid = df[df[col_ris].str.contains("-")]
        for res in valid[col_ris]:
            h, a = map(int, res.split("-"))
            if h > a:
                esiti["1"] += 1
            elif h == a:
                esiti["X"] += 1
            else:
                esiti["2"] += 1
        tot = len(valid)
        return [(e, c, round((c/tot)*100, 2) if tot > 0 else 0) for e, c in esiti.items()]

    ht = calcola_esiti("risultato_ht")
    ft = calcola_esiti("risultato_ft")

    df_win = pd.DataFrame(
        [[ht[i][0], ht[i][1], ht[i][2], ft[i][1], ft[i][2]] for i in range(3)],
        columns=["Esito", "HT Count", "HT %", "FT Count", "FT %"]
    )
    st.table(df_win)

# --- Distribuzione per fasce Odds ed Elo ---
def distribuzione_fasce(df):
    st.subheader("Distribuzione per Fasce Odds & Elo")
    fasce = {
        "Home Odds 1-1.5": df[(df["odd_home"].astype(float) >= 1) & (df["odd_home"].astype(float) <= 1.5)],
        "Home Odds 1.51-2.0": df[(df["odd_home"].astype(float) > 1.5) & (df["odd_home"].astype(float) <= 2.0)],
        "Home Odds > 2.0": df[df["odd_home"].astype(float) > 2.0]
    }
    stats = []
    for nome, subset in fasce.items():
        stats.append([nome, len(subset), round((len(subset)/len(df))*100, 2)])
    st.table(pd.DataFrame(stats, columns=["Fascia", "Partite", "%"]))

# --- Gol per Timeframe ---
def tabella_timeframe_gol(df):
    st.subheader("Distribuzione Gol per Timeframe")
    intervalli = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    risultati = []
    for (start, end) in intervalli:
        partite_con_gol = 0
        for _, row in df.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / len(df)) * 100, 2) if len(df) > 0 else 0
        risultati.append([f"{start}-{end}", partite_con_gol, perc])
    st.table(pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %"]))

# --- MAIN ---
if not filtered_df.empty:
    filtered_df = analizza_gol_range(filtered_df, minuto_da, minuto_a, risultato_corrente)
    calcola_over_btts(filtered_df)
    calcola_winrate_ht_ft(filtered_df)
    distribuzione_fasce(filtered_df)
    tabella_timeframe_gol(filtered_df)
