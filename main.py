import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo allcamp (FT, HT, BTTS & Over Goals con Odd Minima)")

# --- Funzione di connessione a Supabase ---
@st.cache_data
def run_query(query):
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

# --- Carica dataset dalla tabella allcamp ---
df = run_query('SELECT * FROM "allcamp";')
st.write(f"**Righe totali nel dataset:** {len(df)}")

# --- Aggiungi colonna risultato_ft ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

# --- Aggiungi colonna risultato_ht ---
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

filters = {}

# --- FILTRI ---
for col in df.columns:
    if col.lower() in ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]:
        continue
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data" or \
       any(keyword in col.lower() for keyword in ["primo", "secondo", "terzo", "quarto", "quinto"]):
        continue

    # Proviamo a convertire la colonna in numerico
    col_temp = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    # Se abbiamo almeno 2 valori numerici validi, creiamo uno slider
    if col_temp.notnull().sum() >= 2:
        min_val = float(col_temp.min(skipna=True))
        max_val = float(col_temp.max(skipna=True))
        if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
            step_val = 0.01
            selected_range = st.slider(
                f"Filtro per {col}",
                min_val, max_val,
                (min_val, max_val),
                step=step_val
            )
            filters[col] = (selected_range, col_temp)
    else:
        # Colonna non numerica: selectbox
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 0:
            selected_val = st.selectbox(
                f"Filtra per {col} (opzionale)",
                ["Tutti"] + [str(v) for v in unique_vals]
            )
            if selected_val != "Tutti":
                filters[col] = selected_val

# --- APPLICA FILTRI ---
filtered_df = df.copy()
for col, val in filters.items():
    if isinstance(val, tuple):
        range_vals, col_temp = val
        mask = (col_temp >= range_vals[0]) & (col_temp <= range_vals[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- FUNZIONE DISTRIBUZIONE & WINRATE ---
def mostra_distribuzione(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]

    def classifica_risultato(ris):
        home, away = map(int, ris.split("-"))
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df[f"{col_risultato}_classificato"] = df[col_risultato].apply(classifica_risultato)

    st.subheader(f"Distribuzione {titolo}")
    distribuzione = df[f"{col_risultato}_classificato"].value_counts().reset_index()
    distribuzione.columns = ["Risultato", "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    st.table(distribuzione)

    # WinRate 1X2
    count_1 = distribuzione[distribuzione["Risultato"].str.contains("casa vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["1-0","2-0","2-1","3-0","3-1","3-2"])].Conteggio.sum()
    count_2 = distribuzione[distribuzione["Risultato"].str.contains("ospite vince")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-1","0-2","0-3","1-2","1-3","2-3"])].Conteggio.sum()
    count_x = distribuzione[distribuzione["Risultato"].str.contains("pareggio")].Conteggio.sum() + \
              distribuzione[distribuzione["Risultato"].isin(["0-0","1-1","2-2","3-3"])].Conteggio.sum()

    totale = len(df)
    winrate = [
        round((count_1/totale)*100,2),
        round((count_x/totale)*100,2),
        round((count_2/totale)*100,2)
    ]
    odd_minime = [round(100/w,2) if w>0 else "-" for w in winrate]

    winrate_df = pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
        "Conteggio": [count_1, count_x, count_2],
        "WinRate %": winrate,
        "Odd Minima": odd_minime
    })
    st.subheader(f"WinRate {titolo}")
    st.table(winrate_df)

# --- DISTRIBUZIONI FT & HT ---
if not filtered_df.empty:
    mostra_distribuzione(filtered_df, "risultato_ft", "Risultati Finali (FT)")
    mostra_distribuzione(filtered_df, "risultato_ht", "Risultati Primo Tempo (HT)")
