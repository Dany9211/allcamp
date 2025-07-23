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
st.write(f"Righe prima di applicare i filtri: {len(filtered_df)}")

for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        mask = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "giornata":
        mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]
    st.write(f"Righe dopo filtro {col}: {len(filtered_df)}")

if filters == {}:
    st.info("Nessun filtro attivo: vengono mostrati tutti i risultati.")

st.subheader("Dati Filtrati")
st.dataframe(filtered_df.head(50))
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- FUNZIONE DISTRIBUZIONE ---
def mostra_distribuzione(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
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

    df[f"{col_risultato}_class"] = df[col_risultato].apply(classifica_risultato)
    distribuzione = df[f"{col_risultato}_class"].value_counts().reset_index()
    distribuzione.columns = ["Risultato", "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(
        lambda x: round(100/x, 2) if x > 0 else "-"
    )

    st.subheader(f"Distribuzione {titolo}")
    st.table(distribuzione)

    count_1 = distribuzione[distribuzione["Risultato"].str.contains("casa vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["1-0","2-0","2-1","3-0","3-1","3-2"])].Conteggio.sum()
    count_2 = distribuzione[distribuzione["Risultato"].str.contains("ospite vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-1","0-2","0-3","1-2","1-3","2-3"])].Conteggio.sum()
    count_x = distribuzione[distribuzione["Risultato"].str.contains("pareggio")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-0","1-1","2-2","3-3"])].Conteggio.sum()

    totale = len(df)
    winrate = [round((count_1/totale)*100,2), round((count_x/totale)*100,2), round((count_2/totale)*100,2)]
    st.subheader(f"WinRate 1-X-2 ({titolo})")
    st.table(pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
        "Conteggio": [count_1, count_x, count_2],
        "WinRate %": winrate,
        "Odd Minima": [round(100/w,2) if w > 0 else "-" for w in winrate]
    }))

# --- STATISTICHE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    temp_ft = filtered_df["risultato_ft"].str.split("-", expand=True)
    temp_ft = temp_ft.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    filtered_df["home_g_ft"], filtered_df["away_g_ft"] = temp_ft[0], temp_ft[1]
    filtered_df["tot_goals_ft"] = filtered_df["home_g_ft"] + filtered_df["away_g_ft"]

    mostra_distribuzione(filtered_df, "risultato_ft", "Risultati Finali (FT)")
    if "risultato_ht" in filtered_df.columns:
        mostra_distribuzione(filtered_df, "risultato_ht", "Risultati Primo Tempo (HT)")

    # --- BTTS e Over ---
    st.subheader("BTTS (Both Teams To Score)")
    btts = (filtered_df["home_g_ft"] > 0) & (filtered_df["away_g_ft"] > 0)
    st.write(f"Partite BTTS SI: {btts.sum()} ({round(btts.mean()*100,2)}%)")

    st.subheader("Over Goals (FT)")
    over_data = []
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        count = (filtered_df["tot_goals_ft"] > t).sum()
        perc = round((count / len(filtered_df)) * 100, 2)
        over_data.append([f"Over {t}", count, perc, round(100/perc, 2) if perc > 0 else "-"])
    st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    # --- FIRST TO SCORE ---
    if "home_primo_gol" in filtered_df.columns and "away_primo_gol" in filtered_df.columns:
        def calcola_first_to_score(row):
            h = pd.to_numeric(row["home_primo_gol"], errors="coerce")
            a = pd.to_numeric(row["away_primo_gol"], errors="coerce")
            if pd.isna(h) and pd.isna(a):
                return "Nessuno"
            if pd.isna(a) or (not pd.isna(h) and h < a):
                return "Home"
            if pd.isna(h) or (not pd.isna(a) and a < h):
                return "Away"
            return "Nessuno"

        filtered_df["first_to_score"] = filtered_df.apply(calcola_first_to_score, axis=1)
        valid_ft = filtered_df[filtered_df["first_to_score"] != "Nessuno"]

        st.subheader("First to Score")
        if not valid_ft.empty:
            fts = valid_ft["first_to_score"].value_counts().reset_index()
            fts.columns = ["First to Score", "Conteggio"]
            fts["Percentuale %"] = (fts["Conteggio"] / len(valid_ft) * 100).round(2)
            fts["Odd Minima"] = fts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.table(fts)
        else:
            st.write("Nessuna partita valida per il calcolo del First to Score.")
