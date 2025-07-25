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

filters = {}

# --- FILTRI ---
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
    giornata_range = st.sidebar.slider(
        "Seleziona Giornata",
        min_value=giornata_min,
        max_value=giornata_max,
        value=(giornata_min, giornata_max)
    )
    filters["giornata"] = giornata_range

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
    mostra_top10_league(df)

# --- LABEL ODDS ---
def assegna_label_odds(row):
    try:
        oh = float(str(row["odd_home"]).replace(",", "."))
        oa = float(str(row["odd_away"]).replace(",", "."))
    except:
        return "N/A"

    if oh <= 1.5:
        return "Home strong fav"
    elif 1.51 <= oh <= 2.0:
        return "Home med fav"
    elif 2.01 <= oh <= 2.5 and oa > 3.0:
        return "Home small fav"
    elif oa <= 1.5:
        return "Away strong fav"
    elif 1.51 <= oa <= 2.0:
        return "Away med fav"
    elif 2.01 <= oa <= 2.5 and oh > 3.0:
        return "Away small fav"
    elif oh < 3.0 and oa < 3.0:
        return "Supercompetitive"
    return "Altro"

filtered_df["label_odds"] = filtered_df.apply(assegna_label_odds, axis=1)

# --- TOP 10 LEAGUE ---
def mostra_top10_league(df):
    if "league" in df.columns:
        league_stats = df["league"].value_counts().head(10).reset_index()
        league_stats.columns = ["League", "Conteggio"]
        st.write("**Top 10 League:**")
        st.table(league_stats)

# --- ANALISI DAL MINUTO ---
def analizza_da_minuto(df):
    st.subheader("Analisi dinamica (da minuto A a B)")
    start_min, end_min = st.slider("Seleziona intervallo minuti", 1, 90, (20, 45))
    risultati_correnti = st.multiselect("Risultato corrente al minuto iniziale",
                                        ["0-0", "1-0", "0-1", "1-1"], default=["0-0"])

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

    mostra_risultati_esatti(df_target, "risultato_ht", "HT")
    mostra_risultati_esatti(df_target, "risultato_ft", "FT")

    st.subheader(f"WinRate HT e FT ({len(df_target)} partite)")
    st.write("**HT:**")
    st.table(calcola_winrate(df_target, "risultato_ht"))
    st.write("**FT:**")
    st.table(calcola_winrate(df_target, "risultato_ft"))
    mostra_top10_league(df_target)

    temp_ht = df_target["risultato_ht"].str.split("-", expand=True).apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    df_target["tot_goals_ht"] = temp_ht[0] + temp_ht[1]

    temp_ft = df_target["risultato_ft"].str.split("-", expand=True).apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    df_target["tot_goals_ft"] = temp_ft[0] + temp_ft[1]

    st.subheader(f"Over Goals HT ({len(df_target)} partite)")
    over_ht = []
    for t in [0.5, 1.5, 2.5]:
        count = (df_target["tot_goals_ht"] > t).sum()
        perc = round((count / len(df_target)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        over_ht.append([f"Over {t} HT", count, perc, odd_min])
    st.table(pd.DataFrame(over_ht, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))
    mostra_top10_league(df_target)

    st.subheader(f"Over Goals FT ({len(df_target)} partite)")
    over_ft = []
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        count = (df_target["tot_goals_ft"] > t).sum()
        perc = round((count / len(df_target)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        over_ft.append([f"Over {t} FT", count, perc, odd_min])
    st.table(pd.DataFrame(over_ft, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))
    mostra_top10_league(df_target)

    btts = (df_target["tot_goals_ft"] > 0) & (temp_ft[1] > 0)
    count_btts = btts.sum()
    perc_btts = round(count_btts / len(df_target) * 100, 2)
    odd_btts = round(100 / perc_btts, 2) if perc_btts > 0 else "-"
    st.subheader(f"BTTS SI ({len(df_target)} partite)")
    st.write(f"BTTS SI: {count_btts} ({perc_btts}%) - Odd Minima: {odd_btts}")
    mostra_top10_league(df_target)

    st.subheader(f"Distribuzione Gol per Timeframe ({len(df_target)} partite)")
    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    risultati = []
    for (start, end) in intervalli:
        partite_con_gol = 0
        for _, row in df_target.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / len(df_target)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([f"{start}-{end}", partite_con_gol, perc, odd_min])
    st.table(pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"]))
    mostra_top10_league(df_target)

# --- ESECUZIONE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    analizza_da_minuto(filtered_df)
