import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.title("Filtro Completo allcamp (FT, HT, BTTS & Over Goals con Odd Minima)")

# --- Funzione di connessione a Supabase ---
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
    if col.lower() == "id" or "minutaggio" in col.lower() or col.lower() == "data" or        any(keyword in col.lower() for keyword in ["primo", "secondo", "terzo", "quarto", "quinto"]):
        continue

    col_temp = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    if col_temp.notnull().sum() > 0:
        min_val = col_temp.min(skipna=True)
        max_val = col_temp.max(skipna=True)
        if pd.notna(min_val) and pd.notna(max_val):
            step_val = 0.01
            selected_range = st.slider(
                f"Filtro per {col}",
                float(min_val), float(max_val),
                (float(min_val), float(max_val)),
                step=step_val
            )
            filters[col] = (selected_range, col_temp)
    else:
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
