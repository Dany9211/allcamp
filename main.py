import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Allcamp Viewer", layout="wide")
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
    st.error(f"Errore: {e}")
    st.stop()

# colonne risultato
if "gol_home_ft" in df and "gol_away_ft" in df:
    df["risultato_ft"] = df["gol_home_ft"].astype(str)+"-"+df["gol_away_ft"].astype(str)
if "gol_home_ht" in df and "gol_away_ht" in df:
    df["risultato_ht"] = df["gol_home_ht"].astype(str)+"-"+df["gol_away_ht"].astype(str)

filters = {}
# filtri league, anno, giornata simili a tuoi (omessi per brevità)...

def add_range_filter(col_name):
    if col_name in df:
        min_=pd.to_numeric(df[col_name],errors="coerce").min()
        max_=pd.to_numeric(df[col_name],errors="coerce").max()
        st.sidebar.write(f"{col_name}: {min_} – {max_}")
        mn = st.sidebar.text_input(f"Min {col_name}", "")
        mx = st.sidebar.text_input(f"Max {col_name}", "")
        if mn and mx:
            try:
                filters[col_name]=(float(mn),float(mx))
            except: pass

st.sidebar.header("Filtri Quote")
for c in ["odd_home","odd_draw","odd_away"]:
    add_range_filter(c)

filtered_df = df.copy()
for col,val in filters.items():
    if col in ["odd_home","odd_draw","odd_away"]:
        filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors='coerce').between(val[0],val[1])]
    else:
        filtered_df = filtered_df[filtered_df[col]==val]
st.subheader("Dati Filtrati")
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
st.dataframe(filtered_df.head(50))

# label_odds esattamente come tuo...

filtered_df["label_odds"] = filtered_df.apply(lambda row: ...
    # come definito prima
, axis=1)

def calcola_winrate(df,col):
    df_v=df[df[col].notna() & df[col].str.contains("-")]
    cnt = {"1":0,"X":0,"2":0}
    for r in df_v[col]:
        try:
            h,a=map(int,r.split("-"))
            if h>a: cnt["1"]+=1
            elif h<a: cnt["2"]+=1
            else: cnt["X"]+=1
        except: pass
    tot=len(df_v)
    out=[]
    for k in ["1","X","2"]:
        c=cnt[k]; perc=round(c/tot*100,2) if tot else 0
        oddm=round(100/perc,2) if perc>0 else "-"
        out.append((k,c,perc,oddm))
    return pd.DataFrame(out,columns=["Esito","Conteggio","WinRate %","Odd Minima"]),tot

def analizza_da_minuto(df):
    st.subheader("📍 Analisi dinamica su intervallo selezionato")
    start_min,end_min = st.slider("Intervallo minuti",1,90,(20,45))
    risultato_corr = st.selectbox("Risultato all'inizio", sorted(df["risultato_ht"].dropna().unique().tolist()+["0-0","1-0","0-1"]))
    partite=[]
    for _,r in df.iterrows():
        gh=[int(x) for x in str(r.get("minutaggio_gol","")).split(";") if x.isdigit()]
        ga=[int(x) for x in str(r.get("minutaggio_gol_away","")).split(";") if x.isdigit()]
        h0=sum(1 for g in gh if g<start_min)
        a0=sum(1 for g in ga if g<start_min)
        if f"{h0}-{a0}"==risultato_corr:
            # conta solo gol tra start‑end
            gh2=[g for g in gh if start_min<=g<=end_min][:2]
            ga2=[g for g in ga if start_min<=g<=end_min][:2]
            r2=r.copy()
            r2["gol_home_range"]=len(gh2)
            r2["gol_away_range"]=len(ga2)
            partite.append(r2)
    if not partite:
        st.warning("Nessuna partita trovata.")
        return
    df_t = pd.DataFrame(partite)
    st.write(f"**Partite trovate:** {len(df_t)}")
    # winrate HT e FT
    df_ht_win, t_ht = calcola_winrate(df_t,"risultato_ht")
    df_ft_win, t_ft = calcola_winrate(df_t,"risultato_ft")
    st.subheader(f"WinRate HT ({t_ht} partite)") ; st.table(df_ht_win)
    st.subheader(f"WinRate FT ({t_ft} partite)") ; st.table(df_ft_win)
    # over HT sulle gol nel range
    df_t["tot_ht_range"]=df_t["gol_home_range"]+df_t["gol_away_range"]
    st.subheader("Over HT sul range")
    over_ht=[]
    for t in [0.5,1.5,2.5]:
        cnt=(df_t["tot_ht_range"]>t).sum()
        perc=round(cnt/len(df_t)*100,2)
        over_ht.append((f"Over {t}",cnt,perc,round(100/perc,2) if perc>0 else "-"))
    st.table(pd.DataFrame(over_ht,columns=["Mercato","Conteggio","Percentuale %","Odd Minima"]))
    # over FT range medesimo
    st.subheader("Over FT sul range")
    over_ft=[]
    for t in [0.5,1.5,2.5,3.5,4.5]:
        cnt=(df_t["tot_ht_range"]>t).sum()
        perc=round(cnt/len(df_t)*100,2)
        over_ft.append((f"Over {t}",cnt,perc,round(100/perc,2) if perc>0 else "-"))
    st.table(pd.DataFrame(over_ft,columns=["Mercato","Conteggio","Percentuale %","Odd Minima"]))
    # BTTS range
    st.subheader("BTTS SI sul range")
    btts=(df_t["gol_home_range"]>0)&(df_t["gol_away_range"]>0)
    cnt=btts.sum();perc=round(cnt/len(df_t)*100,2)
    odd_bt=round(100/perc,2) if perc>0 else "-"
    st.write(f"{cnt} partite ({perc}%) - Odd Minima: {odd_bt}")
    # label odds
    st.subheader("Distribuzione Label Odds")
    ld=df_t["label_odds"].value_counts().reset_index()
    ld.columns=["Label","Conteggio"]; ld["%"]=round(ld["Conteggio"]/len(df_t)*100,2)
    st.table(ld)
    # distribuzione timebands sempre fino al 90
    st.subheader("Distribuzione Gol per Timeframe (fino 90')")
    ints=[(0,15),(16,30),(31,45),(46,60),(61,75),(76,90)]
    out=[]
    for s,e in ints:
        pc=0
        for _,r in df_t.iterrows():
            gh=[int(x) for x in str(r.get("minutaggio_gol","")).split(";") if x.isdigit()]
            ga=[int(x) for x in str(r.get("minutaggio_gol_away","")).split(";") if x.isdigit()]
            if any(s<=g<=e for g in gh+ga):
                pc+=1
        perc=round(pc/len(df_t)*100,2)
        out.append((f"{s}-{e}",pc,perc))
    st.table(pd.DataFrame(out,columns=["Timeframe","Partite con Gol","Percentuale %"]))

# esecuzione
if not filtered_df.empty and "risultato_ft" in filtered_df:
    analizza_da_minuto(filtered_df)
