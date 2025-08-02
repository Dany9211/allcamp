import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi Next Gol e stats live", layout="wide")
st.title("Analisi Tabella allcamp")

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

try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento: {e}")
    st.stop()

if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)
filters = {}

def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        st.sidebar.write(f"Range attuale {col_name}: {col_min} - {col_max}")
        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="")
        if min_val.strip() and max_val.strip():
            try:
                filters[col_name] = (float(min_val), float(max_val))
            except:
                st.sidebar.warning(f"Valori non validi per {col_name}")

# --- Sidebar Filtri ---
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

if "risultato_ht" in df.columns:
    risultati_ht = ["Tutti"] + sorted(df["risultato_ht"].dropna().unique())
    selected_risultato_ht = st.sidebar.selectbox("Risultato HT", risultati_ht)
    if selected_risultato_ht != "Tutti":
        filters["risultato_ht"] = selected_risultato_ht

for team_col in ["home_team", "away_team"]:
    if team_col in df.columns:
        teams = ["Tutte"] + sorted(df[team_col].dropna().unique())
        selected_team = st.sidebar.selectbox(f"Seleziona {team_col.replace('_', ' ').title()}", teams)
        if selected_team != "Tutte":
            filters[team_col] = selected_team

st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- Applica Filtri ---
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

st.subheader("Dati Filtrati")
st.dataframe(filtered_df.head(50))
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
st.subheader("Distribuzione Gol Globale — Intervalli da 15 minuti (base filtri)")
intervalli_15min = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
risultati_15 = []

for (start, end) in intervalli_15min:
    partite_con_gol = 0
    for _, row in filtered_df.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
        if any(start <= g <= end for g in gol_home + gol_away):
            partite_con_gol += 1
    perc = round((partite_con_gol / len(filtered_df)) * 100, 2)
    odd_min = round(100 / perc, 2) if perc > 0 else "-"
    risultati_15.append([f"{start}-{end}", partite_con_gol, perc, odd_min])

st.table(pd.DataFrame(risultati_15, columns=["Timeframe 15'", "Partite con Gol", "Percentuale %", "Odd Minima"]))
def calcola_winrate(df, col_risultato):
    df_valid = df[df[col_risultato].notna() & df[col_risultato].str.contains("-")]
    risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for ris in df_valid[col_risultato]:
        try:
            home, away = map(int, ris.split("-"))
            if home > away: risultati["1 (Casa)"] += 1
            elif home < away: risultati["2 (Trasferta)"] += 1
            else: risultati["X (Pareggio)"] += 1
        except: continue
    totale = len(df_valid)
    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale) * 100, 2) if totale > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

def mostra_risultati_esatti(df, col_risultato, titolo):
    interessanti = ["0-0", "0-1", "1-0", "1-1", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2", "3-3"]
    df_valid = df[df[col_risultato].notna() & df[col_risultato].str.contains("-")].copy()
    def classifica(ris):
        try:
            h, a = map(int, ris.split("-"))
        except: return "Altro"
        if ris in interessanti: return ris
        if h > a: return "Altro vittoria casa"
        elif h < a: return "Altro vittoria trasferta"
        else: return "Altro pareggio"
    df_valid["classificato"] = df_valid[col_risultato].apply(classifica)
    dist = df_valid["classificato"].value_counts().reset_index()
    dist.columns = [titolo, "Conteggio"]
    dist["Percentuale %"] = (dist["Conteggio"] / len(df_valid) * 100).round(2)
    dist["Odd Minima"] = dist["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    st.subheader(f"Risultati Esatti {titolo} ({len(df_valid)} partite)")
    st.table(dist)

def analizza_da_minuto(df):
    st.subheader("Analisi dinamica (da minuto A a B)")
    start_min, end_min = st.slider("Seleziona intervallo minuti", 1, 90, (20, 45))
    risultati_correnti = st.multiselect("Risultato corrente al minuto iniziale",
                                        ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2"], default=["0-0"])

    partite_target = []
    for _, row in df.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
        home_fino = sum(1 for g in gol_home if g < start_min)
        away_fino = sum(1 for g in gol_away if g < start_min)
        risultato_fino = f"{home_fino}-{away_fino}"
        if risultato_fino in risultati_correnti:
            partite_target.append(row)

    if not partite_target:
        st.warning(f"Nessuna partita con risultato selezionato al minuto {start_min}.")
        return

    df_target = pd.DataFrame(partite_target)
    st.write(f"**Partite trovate:** {len(df_target)}")

    df_target[["home_range", "away_range"]] = df_target.apply(lambda row: pd.Series([
        sum(1 for g in str(row.get("minutaggio_gol", "")).split(";") if g.isdigit() and start_min <= int(g) <= end_min),
        sum(1 for g in str(row.get("minutaggio_gol_away", "")).split(";") if g.isdigit() and start_min <= int(g) <= end_min)
    ]), axis=1)

    df_target["tot_goals_range"] = df_target["home_range"] + df_target["away_range"]

    mostra_risultati_esatti(df_target, "risultato_ht", "HT")
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

    # Distribuzione 5 minuti
    st.subheader("Distribuzione Gol — Intervalli da 5 minuti (solo partite nel range)")
    intervalli_5min = [(i, i + 4) for i in range(1, 91, 5)]
    risultati_5min = []
    for (start, end) in intervalli_5min:
        partite_con_gol = 0
        for _, row in df_target.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / len(df_target)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati_5min.append([f"{start}-{end}", partite_con_gol, perc, odd_min])
    st.table(pd.DataFrame(risultati_5min, columns=["Timeframe 5'", "Partite con Gol", "Percentuale %", "Odd Minima"]))

    # Distribuzione 15 minuti
    st.subheader("Distribuzione Gol — Intervalli da 15 minuti (solo partite nel range)")
    intervalli_15min = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    risultati_15min = []
    for (start, end) in intervalli_15min:
        partite_con_gol = 0
        for _, row in df_target.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol = 0
        for _, row in df_target.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / len(df_target)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati_15min.append([f"{start}-{end}", partite_con_gol, perc, odd_min])

    st.table(pd.DataFrame(risultati_15min, columns=["Timeframe 15'", "Partite con Gol", "Percentuale %", "Odd Minima"]))
