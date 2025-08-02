import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi Next Gol e stats live", layout="wide")
st.title("Analisi Tabella allcamp")

# --- Connessione al DB ---
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

# --- Aggiunta colonne risultato ---
df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

# --- Filtri Sidebar ---
filters = {}

def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        st.sidebar.write(f"Range {col_name}: {col_min} - {col_max}")
        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="")
        if min_val.strip() and max_val.strip():
            try:
                filters[col_name] = (float(min_val), float(max_val))
            except:
                st.sidebar.warning(f"Valori non validi per {col_name}")

st.sidebar.header("🎛️ Filtri")
for field in ["league", "anno", "home_team", "away_team", "risultato_ht"]:
    if field in df.columns:
        values = ["Tutti"] + sorted(df[field].dropna().unique())
        selected = st.sidebar.selectbox(f"{field}", values)
        if selected != "Tutti":
            filters[field] = selected

if "giornata" in df.columns:
    giornata_range = st.sidebar.slider("Giornata", int(df["giornata"].min()), int(df["giornata"].max()), (1, 38))
    filters["giornata"] = giornata_range

st.sidebar.header("🎯 Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- Applica filtri ---
filtered_df = df.copy()
for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        filtered_df = filtered_df[pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])]
    elif col == "giornata":
        filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

st.subheader("📄 Dati filtrati")
st.dataframe(filtered_df.head(50))
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- Funzioni ---
def calcola_winrate(df, col):
    df_valid = df[df[col].notna() & df[col].str.contains("-")]
    esiti = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for ris in df_valid[col]:
        try:
            h, a = map(int, ris.split("-"))
            if h > a: esiti["1 (Casa)"] += 1
            elif h < a: esiti["2 (Trasferta)"] += 1
            else: esiti["X (Pareggio)"] += 1
        except: continue
    totale = len(df_valid)
    stats = []
    for esito, count in esiti.items():
        perc = round(count / totale * 100, 2) if totale > 0 else 0
        odd = round(100 / perc, 2) if perc > 0 else "-"
        stats.append([esito, count, perc, odd])
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def mostra_risultati_esatti(df, col, titolo):
    interessanti = ["0-0", "0-1", "1-0", "1-1", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2", "3-3"]
    df_valid = df[df[col].notna() & df[col].str.contains("-")].copy()
    def classifica(ris):
        try: h, a = map(int, ris.split("-"))
        except: return "Altro"
        if ris in interessanti: return ris
        return "Altro vittoria casa" if h > a else "Altro vittoria trasferta" if h < a else "Altro pareggio"
    df_valid["classificato"] = df_valid[col].apply(classifica)
    dist = df_valid["classificato"].value_counts().reset_index()
    dist.columns = [titolo, "Conteggio"]
    dist["Percentuale %"] = round(dist["Conteggio"] / len(df_valid) * 100, 2)
    dist["Odd Minima"] = dist["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    st.subheader(f"📌 Risultati Esatti {titolo}")
    st.table(dist)

def distribuzione_gol(df, intervalli, label):
    risultati = []
    for (start, end) in intervalli:
        count = 0
        for _, row in df.iterrows():
            gol_home = [int(g) for g in str(row.get("minutaggio_gol", "")).split(";") if g.isdigit()]
            gol_away = [int(g) for g in str(row.get("minutaggio_gol_away", "")).split(";") if g.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                count += 1
        perc = round(count / len(df) * 100, 2)
        odd = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([f"{start}-{end}" if end < 91 else "91+", count, perc, odd])
    st.subheader(f"⏱️ {label}")
    st.table(pd.DataFrame(risultati, columns=["Intervallo", "Partite con Gol", "Percentuale %", "Odd Minima"]))

# --- Analisi dinamica ---
def analizza_da_minuto(df):
    st.subheader("🧠 Analisi dinamica minuti")
    start_min, end_min = st.slider("Intervallo minuti", 1, 90, (20, 45))
    risultati_correnti = st.multiselect("Risultato corrente iniziale", ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2"], default=["0-0"])
    partite = []
    for _, row in df.iterrows():
        h_gol = [int(g) for g in str(row.get("minutaggio_gol", "")).split(";") if g.isdigit()]
        a_gol = [int(g) for g in str(row.get("minutaggio_gol_away", "")).split(";") if g.isdigit()]
        h_fino = sum(1 for g in h_gol if g < start_min)
        a_fino = sum(1 for g in a_gol if g < start_min)
        risultato = f"{h_fino}-{a_fino}"
        if risultato in risultati_correnti:
            partite.append(row)
    if not partite:
        st.warning("Nessuna partita trovata")
        return
    df_target = pd.DataFrame(partite)
    df_target[["home_range", "away_range
