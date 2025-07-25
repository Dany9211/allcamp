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

# --- FILTRI QUOTE ---
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
                filters[col_name] = (float(min_val), float(max_val))
            except:
                st.sidebar.warning(f"Valori non validi per {col_name}")

st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

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
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"]), totale

# --- NUOVA FUNZIONE ANALISI ---
def analizza_da_minuto(df):
    st.subheader("Analisi dal minuto selezionato")
    minuto_sel = st.slider("Seleziona Minuto di riferimento", 1, 90, 20)
    risultati_possibili = sorted(set(list(df["risultato_ht"].dropna().unique()) + ["0-0", "1-0", "0-1"]))
    risultato_corrente = st.selectbox("Risultato corrente al minuto selezionato", risultati_possibili)

    partite_target = []
    for _, row in df.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]

        home_fino = sum(1 for g in gol_home if g < minuto_sel)
        away_fino = sum(1 for g in gol_away if g < minuto_sel)
        risultato_fino = f"{home_fino}-{away_fino}"

        if risultato_corrente == risultato_fino:
            partite_target.append(row)

    if not partite_target:
        st.warning(f"Nessuna partita con risultato {risultato_corrente} al minuto {minuto_sel}.")
        return

    df_target = pd.DataFrame(partite_target)
    st.write(f"**Partite trovate:** {len(df_target)}")

    # --- WINRATE HT ---
    st.subheader("WinRate HT")
    ht_winrate, totale_ht = calcola_winrate(df_target, "risultato_ht")
    st.write(f"Totale partite HT: {totale_ht}")
    st.table(ht_winrate)

    # --- WINRATE FT ---
    st.subheader("WinRate FT")
    ft_winrate, totale_ft = calcola_winrate(df_target, "risultato_ft")
    st.write(f"Totale partite FT: {totale_ft}")
    st.table(ft_winrate)

    # --- OVER HT ---
    if "risultato_ht" in df_target.columns:
        temp_ht = df_target["risultato_ht"].str.split("-", expand=True)
        temp_ht = temp_ht.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        df_target["home_g_ht"], df_target["away_g_ht"] = temp_ht[0], temp_ht[1]
        df_target["tot_goals_ht"] = df_target["home_g_ht"] + df_target["away_g_ht"]

        st.subheader("Over Goals HT")
        over_data = []
        for t in [0.5, 1.5, 2.5]:
            count = (df_target["tot_goals_ht"] > t).sum()
            perc = round((count / len(df_target)) * 100, 2)
            over_data.append([f"Over {t} HT", count, perc, round(100/perc, 2) if perc > 0 else "-"])
        st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    # --- OVER FT ---
    if "risultato_ft" in df_target.columns:
        temp_ft = df_target["risultato_ft"].str.split("-", expand=True)
        temp_ft = temp_ft.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        df_target["home_g_ft"], df_target["away_g_ft"] = temp_ft[0], temp_ft[1]
        df_target["tot_goals_ft"] = df_target["home_g_ft"] + df_target["away_g_ft"]

        st.subheader("Over Goals FT")
        over_data_ft = []
        for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
            count = (df_target["tot_goals_ft"] > t).sum()
            perc = round((count / len(df_target)) * 100, 2)
            over_data_ft.append([f"Over {t} FT", count, perc, round(100/perc, 2) if perc > 0 else "-"])
        st.table(pd.DataFrame(over_data_ft, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

        # --- BTTS ---
        btts = (df_target["home_g_ft"] > 0) & (df_target["away_g_ft"] > 0)
        count_btts = btts.sum()
        perc_btts = round((count_btts / len(df_target)) * 100, 2)
        odd_btts = round(100 / perc_btts, 2) if perc_btts > 0 else "-"
        st.subheader("BTTS (Both Teams To Score)")
        st.write(f"BTTS SI: {count_btts} partite ({perc_btts}%) - Odd Minima BTTS: {odd_btts}")

    # --- DISTRIBUZIONE GOL PER TIMEFRAME ---
    st.subheader("Distribuzione Gol per Timeframe")
    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    risultati = []
    total_partite = len(df_target)

    for (start, end) in intervalli:
        partite_con_gol = 0
        for _, row in df_target.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / total_partite) * 100, 2) if total_partite > 0 else 0
        risultati.append([f"{start}-{end}", partite_con_gol, perc])

    df_ris = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %"])
    st.table(df_ris)

# --- ESECUZIONE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    analizza_da_minuto(filtered_df)
