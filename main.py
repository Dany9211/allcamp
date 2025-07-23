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

# --- COLONNE DA ESCLUDERE ---
exclude_columns = [
    "gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht",
    "sesto_gol_home", "settimo_gol_home", "ottavo_gol_home", "nono_gol_home",
    "sesto_gol_away", "settimo_gol_away", "ottavo_gol_away", "nono_gol_away",
    "sutht", "sutht1", "sutht2", "sutat", "sutat1", "sutat2",
    "sesto_gol_home_", "odd_under_3_5", "odd_under_4_5",
    "odd_over_0_5", "odd_over_1_5", "odd_over_3_5", "odd_over_4_5",
    "odd_under_0_5", "odd_under_1_5",
    "primo_gol_home_", "quarto_gol_home", "quinto_gol_home",
    "quarto_gol_away", "quinto_gol_away",
    "btts_si", "elohomeo", "eloawayo", "formah", "formaa"
]

st.markdown("### Filtri Quote e Altri")

# --- FILTRI STANDARD ---
for col in df.columns:
    if col.lower() in exclude_columns:
        continue
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data":
        continue

    col_temp = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    # Filtro speciale per odd_home, odd_away, odd_draw (input min e max)
    if col.lower() in ["odd_home", "odd_away", "odd_draw"]:
        min_val = float(col_temp.min(skipna=True)) if col_temp.notnull().sum() > 0 else 0
        max_val = float(col_temp.max(skipna=True)) if col_temp.notnull().sum() > 0 else 10
        st.write(f"**Filtro per {col}**")
        min_input = st.text_input(f"Min {col}", str(min_val), key=f"{col}_min")
        max_input = st.text_input(f"Max {col}", str(max_val), key=f"{col}_max")
        try:
            min_input = float(min_input)
            max_input = float(max_input)
            filters[col] = ((min_input, max_input), col_temp)
        except:
            st.warning(f"Valori non validi per {col}, usare numeri validi.")
        continue

    # Filtri standard per altre colonne numeriche
    if col_temp.notnull().sum() >= 2:
        min_val = float(col_temp.min(skipna=True))
        max_val = float(col_temp.max(skipna=True))
        if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
            selected_range = st.slider(
                f"Seleziona range per {col}",
                min_val, max_val,
                (min_val, max_val),
                step=0.01,
                key=f"{col}_slider"
            )
            filters[col] = (selected_range, col_temp)
    else:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 0:
            selected_val = st.selectbox(
                f"Filtra per {col} (opzionale)",
                ["Tutti"] + [str(v) for v in unique_vals],
                key=f"{col}_select"
            )
            if selected_val != "Tutti":
                filters[col] = selected_val

# --- APPLICA FILTRI ---
filtered_df = df.copy()
for col, val in filters.items():
    if isinstance(val, tuple) and isinstance(val[0], (float, int)):
        range_vals, col_temp = val
        mask = (col_temp >= range_vals[0]) & (col_temp <= range_vals[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]

st.subheader("Dati Filtrati")
st.dataframe(filtered_df)
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
