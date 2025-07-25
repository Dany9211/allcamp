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

# --- FILTRI BASE ---
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


# --- ANALISI DA MINUTO SELEZIONATO ---
def analizza_da_minuto(df):
    st.subheader("Analisi Over e BTTS dal minuto selezionato")
    minuto_sel = st.slider("Seleziona Minuto di riferimento", 1, 90, 20)

    # Risultato corrente selezionabile
    risultati_possibili = sorted(set(list(df["risultato_ht"].dropna().unique()) + ["0-0", "1-0", "0-1"]))
    risultato_corrente = st.selectbox("Risultato corrente al minuto selezionato", risultati_possibili)

    partite_target = []
    for idx, row in df.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]

        # Conto gol fino al minuto selezionato
        home_fino = sum(1 for g in gol_home if g < minuto_sel)
        away_fino = sum(1 for g in gol_away if g < minuto_sel)
        risultato_fino = f"{home_fino}-{away_fino}"

        # Verifico se il risultato coincide
        if risultato_corrente == risultato_fino:
            partite_target.append(row)

    if not partite_target:
        st.warning(f"Nessuna partita con risultato {risultato_corrente} al minuto {minuto_sel}.")
        return

    df_target = pd.DataFrame(partite_target)
    st.write(f"**Partite trovate:** {len(df_target)}")

    # --- Calcolo Over/Under ---
    def calcola_over(df, goals_col, soglie, titolo):
        st.write(f"**{titolo}**")
        over_data = []
        for t in soglie:
            count = (df[goals_col] > t).sum()
            perc = round((count / len(df)) * 100, 2)
            over_data.append([f"Over {t}", count, perc, round(100/perc, 2) if perc > 0 else "-"])
        st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    if "risultato_ht" in df_target.columns:
        temp_ht = df_target["risultato_ht"].str.split("-", expand=True)
        temp_ht = temp_ht.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        df_target["home_g_ht"], df_target["away_g_ht"] = temp_ht[0], temp_ht[1]
        df_target["tot_goals_ht"] = df_target["home_g_ht"] + df_target["away_g_ht"]
        calcola_over(df_target, "tot_goals_ht", [0.5, 1.5, 2.5], "Over Goals HT Successivi")

    if "risultato_ft" in df_target.columns:
        temp_ft = df_target["risultato_ft"].str.split("-", expand=True)
        temp_ft = temp_ft.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        df_target["home_g_ft"], df_target["away_g_ft"] = temp_ft[0], temp_ft[1]
        df_target["tot_goals_ft"] = df_target["home_g_ft"] + df_target["away_g_ft"]
        calcola_over(df_target, "tot_goals_ft", [0.5, 1.5, 2.5, 3.5, 4.5], "Over Goals FT")

        btts = (df_target["home_g_ft"] > 0) & (df_target["away_g_ft"] > 0)
        count_btts = btts.sum()
        perc_btts = round((count_btts / len(df_target)) * 100, 2)
        st.write(f"**BTTS SI:** {count_btts} partite ({perc_btts}%)")

    # --- Distribuzione per Label Odds ---
    if "label_odds" in df_target.columns:
        st.write("**Distribuzione per Label Odds**")
        st.table(df_target["label_odds"].value_counts())

# --- STATISTICHE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    analizza_da_minuto(filtered_df)
