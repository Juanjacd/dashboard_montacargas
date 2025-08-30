# --- Compat NumPy/Plotly (np.bool) ---
import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3, hashlib, io
from datetime import time as dtime, date as ddate  # 'date' para anclar 22 de julio

# PDF opcional (no romper si no está)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# =================== CONFIG ===================
st.set_page_config(page_title="Órdenes Recibidas por Montacargas", layout="wide")

# cache compat
try:
    _cache = st.cache_data
    def CACHE(**kw): return _cache(**kw)
except AttributeError:
    def CACHE(**kw):
        def _wrap(f): return st.cache(f, allow_output_mutation=True, suppress_st_warning=True)
        return _wrap

if not hasattr(st, "divider"):
    def _divider(): st.markdown("---")
    st.divider = _divider

# ======= Estilos (hero + título + ancho de tablas) =======
st.markdown("""
<style>
:root { --brand:#0f172a; --accent:#2563eb; --muted:#64748b; }
h1.hero-title{
  font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;
  font-weight:800; font-size:28px; margin:0; color:var(--brand);
}
p.hero-sub{ color:var(--muted); margin-top:6px; margin-bottom:0; }
div.hero{
  border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px;
  background: linear-gradient(180deg,#ffffff,#f8fafc);
  margin-bottom: 14px;
}
div.hero .badges span{
  display:inline-block; background:#eef2ff; color:#3730a3;
  padding:4px 10px; border-radius:999px; font-size:12px; margin-right:8px;
}
div[data-testid="stFileUploader"] button{ font-size:0 !important; }
div[data-testid="stFileUploader"] button:after{ content:" Insertar archivo"; font-size:14px; font-weight:600; }
div[data-testid="stDataFrame"]{ width:100% !important; }
div[data-testid="stTable"]{ width:100% !important; }
.block-container{ padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ---- HERO ----
st.markdown("""
<div class="hero">
  <h1 class="hero-title">🚜 Órdenes Recibidas por Montacargas</h1>
  <p class="hero-sub">Monitorea productividad y tiempos muertos por usuario y día. Gaps > 25 min, mismo turno, excluye alimentación.</p>
  <div class="badges" style="margin-top:10px;">
    <span>Turno 1: 05:00–13:55</span>
    <span>Turno 2: 14:00–22:55</span>
    <span>Alimentación T1: 08:20–09:00</span>
    <span>Alimentación T2: 17:25–18:00</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =================== PARÁMETROS ===================
THRESH_MIN = 25
EXCLUSION_WINDOWS = [
    (dtime(8,20), dtime(9,0)),    # Alimentación T1
    (dtime(17,25), dtime(18,0)),  # Alimentación T2
]

# =================== UTILIDADES ===================
def subtract_window(seg_start, seg_end, win_start, win_end):
    if win_end <= seg_start or win_start >= seg_end:
        return [(seg_start, seg_end)]
    parts = []
    if seg_start < win_start:
        parts.append((seg_start, max(seg_start, win_start)))
    if seg_end > win_end:
        parts.append((min(seg_end, win_end), seg_end))
    return [(s,e) for (s,e) in parts if e > s]

def subtract_windows(seg_start, seg_end, date, windows):
    segs = [(seg_start, seg_end)]
    for st_t, en_t in windows:
        st_w = pd.Timestamp.combine(pd.Timestamp(date), st_t)
        en_w = pd.Timestamp.combine(pd.Timestamp(date), en_t)
        new_segs = []
        for s, e in segs:
            new_segs.extend(subtract_window(s, e, st_w, en_w))
        segs = new_segs
        if not segs: break
    return segs

# =================== SIDEBAR ===================
with st.sidebar:
    with st.expander("📥 Carga de datos (ocultar/mostrar)", expanded=False):
        up = st.file_uploader("📎 Inserta aquí tu Excel (.xlsx)", type=["xlsx"],
            help="Columnas: Usuario, Fecha(Confirmación), Hora(Confirmación), Número de orden (opcional)")
        hoja = st.text_input("Hoja", value="Hoja1")

        st.markdown("—")
        st.caption("Histórico en SQLite")
        use_db = st.checkbox("Usar histórico", value=True)
        DB_PATH = st.text_input("Archivo DB", value="montacargas.db")
        c1, c2 = st.columns(2)
        with c1: btn_clear = st.button("🧹 Limpiar histórico")
        with c2: btn_reload = st.button("🔁 Recargar histórico")

    with st.expander("⚙️ Preferencias", expanded=False):
        chart_type = st.selectbox("Tipo para 'Órdenes por usuario'", ["Barra horizontal", "Barra vertical"])

# =================== SQLITE ===================
TABLE = "ordenes"
def ensure_db(path):
    con = sqlite3.connect(path)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            id TEXT PRIMARY KEY, usuario TEXT, fecha TEXT, hora INTEGER,
            turno TEXT, datetime TEXT, orden TEXT
        )
    """)
    con.commit(); con.close()

def make_uid(row):
    base = f"{row.get('Usuario','')}|{row.get('Fecha','')}|{row.get('Datetime','')}|{row.get('Orden','')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def upsert_df(path, df):
    if df.empty: return 0
    con = sqlite3.connect(path); cur = con.cursor()
    df2 = df.copy()
    if "Orden" not in df2.columns: df2["Orden"] = None
    df2["id"] = df2.apply(make_uid, axis=1)
    rows = []
    for _, r in df2.iterrows():
        id_ = r["id"]; usuario = r.get("Usuario")
        fecha = str(r.get("Fecha")) if pd.notnull(r.get("Fecha")) else None
        hora = int(r.get("Hora")) if pd.notnull(r.get("Hora")) else None
        turno = r.get("Turno")
        dt = pd.to_datetime(r.get("Datetime"), errors="coerce")
        dt_iso = dt.isoformat() if pd.notnull(dt) else None
        orden = r.get("Orden")
        rows.append((id_, usuario, fecha, hora, turno, dt_iso, orden))
    cur.executemany(
        f"INSERT OR IGNORE INTO {TABLE} (id,usuario,fecha,hora,turno,datetime,orden) VALUES (?,?,?,?,?,?,?)",
        rows
    )
    con.commit(); con.close()
    return len(rows)

def read_all(path) -> pd.DataFrame:
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
    except Exception:
        df = pd.DataFrame(columns=["id","usuario","fecha","hora","turno","datetime","orden"])
    con.close()
    if df.empty: return df
    df.rename(columns={"usuario":"Usuario","fecha":"Fecha","hora":"Hora","turno":"Turno","datetime":"Datetime","orden":"Orden"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.date
    return df

def clear_db(path):
    con = sqlite3.connect(path); con.execute(f"DELETE FROM {TABLE}"); con.commit(); con.close()

# =================== PARSEO EXCEL ===================
def to_time(x):
    if pd.isna(x): return None
    try:
        if hasattr(x, "hour"):
            return dtime(int(x.hour), int(getattr(x, "minute", 0)), int(getattr(x, "second", 0)))
    except Exception:
        pass
    s = str(x).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            dt = pd.to_datetime(s, format=fmt)
            return dtime(int(dt.hour), int(dt.minute), int(getattr(dt, "second", 0)))
        except Exception:
            pass
    dt = pd.to_datetime(s, errors="coerce")
    return dtime(int(dt.hour), int(dt.minute), int(getattr(dt, "second", 0))) if pd.notnull(dt) else None

@CACHE(show_spinner=False)
def load_excel(file, sheet_name="Hoja1") -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=sheet_name)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    if "usuario" in cols_lower: col_user = cols_lower["usuario"]
    else: raise ValueError("No se encontró la columna 'Usuario'.")

    if "fecha" in cols_lower:
        fecha = pd.to_datetime(df[cols_lower["fecha"]], errors="coerce").dt.date
    elif "fecha confirmación" in cols_lower:
        fecha = pd.to_datetime(df[cols_lower["fecha confirmación"]], dayfirst=True, errors="coerce").dt.date
    else:
        raise ValueError("No se encontró 'Fecha' ni 'Fecha confirmación'.")

    if "hora" in cols_lower:
        hora_t = df[cols_lower["hora"]].apply(to_time)
    elif "hora de confirmación" in cols_lower:
        hora_t = df[cols_lower["hora de confirmación"]].apply(to_time)
    else:
        raise ValueError("No se encontró 'Hora' ni 'Hora de confirmación'.")

    if "orden" in cols_lower:
        orden = df[cols_lower["orden"]].astype(str).str.strip()
    elif "número de orden de transporte" in cols_lower:
        orden = df[cols_lower["número de orden de transporte"]].astype(str).str.strip()
    else:
        orden = None

    out = pd.DataFrame({
        "Usuario": df[cols_lower["usuario"]].astype(str).str.strip(),
        "Fecha": fecha,
        "Time": hora_t,
    })
    out["Hora"] = out["Time"].apply(lambda t: t.hour if t else None)
    out["Orden"] = orden if orden is not None else None

    def turno(t):
        if t is None: return None
        if dtime(5,0) <= t <= dtime(13,55): return "Turno 1"
        if dtime(14,0) <= t <= dtime(22,55): return "Turno 2"
        return None

    out["Turno"] = out["Time"].apply(turno)
    out["Datetime"] = out.apply(
        lambda r: pd.Timestamp.combine(r["Fecha"], r["Time"]) if (pd.notnull(r["Fecha"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    out = out.dropna(subset=["Fecha","Time","Datetime"])
    return out[["Usuario","Fecha","Hora","Time","Turno","Datetime","Orden"]]

# =================== CARGA / HISTÓRICO ===================
if up is None and 'use_db' in locals() and not use_db:
    st.warning("Sube un Excel para empezar o activa el histórico."); st.stop()

df_new = pd.DataFrame()
if up is not None:
    try:
        df_new = load_excel(up, hoja)
    except Exception as e:
        st.error(f"❌ No pude leer el Excel: {e}"); st.stop()

if 'use_db' not in locals():
    use_db = True
if 'DB_PATH' not in locals():
    DB_PATH = "montacargas.db"

if use_db:
    ensure_db(DB_PATH)
    if 'btn_clear' in locals() and btn_clear:
        clear_db(DB_PATH); st.success("Histórico limpiado.")
    if not df_new.empty:
        upsert_df(DB_PATH, df_new)
    df = read_all(DB_PATH)
else:
    df = df_new.copy()

if df.empty:
    st.info("No hay datos para visualizar aún."); st.stop()

df = df[df["Turno"].isin(["Turno 1","Turno 2"])]

# =================== FILTROS ===================
with st.sidebar:
    users = sorted(df["Usuario"].dropna().unique().tolist())
    turns = ["Turno 1","Turno 2"]
    fmin, fmax = df["Fecha"].min(), df["Fecha"].max()
    sel_users = st.multiselect("Usuarios", users, [])
    sel_turns = st.multiselect("Turnos", turns, [])
    sel_range = st.date_input("Rango de fechas", [fmin, fmax])

df_f = df.copy()
if sel_users: df_f = df_f[df_f["Usuario"].isin(sel_users)]
if sel_turns: df_f = df_f[df_f["Turno"].isin(sel_turns)]
if isinstance(sel_range, list) and len(sel_range) == 2:
    d0, d1 = pd.to_datetime(sel_range[0]), pd.to_datetime(sel_range[1])
    df_f = df_f[(pd.to_datetime(df_f["Datetime"]) >= d0) &
                (pd.to_datetime(df_f["Datetime"]) <= d1 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

# =================== CÁLCULO GAPS (una vez) ===================
def compute_dead(df_in: pd.DataFrame):
    df_g = df_in.sort_values(["Usuario","Datetime"]).copy()
    df_g["prev_dt"] = df_g.groupby("Usuario")["Datetime"].shift(1)
    df_g["prev_fecha"] = df_g.groupby("Usuario")["Fecha"].shift(1)
    df_g["prev_turno"] = df_g.groupby("Usuario")["Turno"].shift(1)
    same = df_g["prev_dt"].notna() & (df_g["Fecha"] == df_g["prev_fecha"]) & (df_g["Turno"] == df_g["prev_turno"])
    df_g = df_g[same].copy()

    segments_for_union = []
    rows_list = []

    for _, r in df_g.iterrows():
        start = r["prev_dt"]; end = r["Datetime"]; date = r["Fecha"]
        gap_min = (end - start).total_seconds()/60.0
        if gap_min <= 0: continue
        segs = subtract_windows(start, end, date, EXCLUSION_WINDOWS)
        if not segs: continue
        adj_min = sum((e - s).total_seconds()/60.0 for s, e in segs)
        if adj_min > THRESH_MIN:
            rows_list.append({"Usuario": r["Usuario"], "Fecha": date, "Turno": r["Turno"], "AdjMin": adj_min})
            for s, e in segs:
                segments_for_union.append((date, s, e))

    dead = pd.DataFrame(rows_list)
    plant_hours = 0.0
    if len(segments_for_union):
        from collections import defaultdict
        by_day = defaultdict(list)
        for date, s, e in segments_for_union: by_day[date].append((s, e))
        for date, segs in by_day.items():
            segs.sort(key=lambda x: x[0])
            merged = []
            cur_s, cur_e = segs[0]
            for s, e in segs[1:]:
                if s <= cur_e: cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            plant_hours += sum((e - s).total_seconds()/3600.0 for s, e in merged)
    return dead, round(plant_hours, 2)

dead, plant_hours = compute_dead(df_f)

# =================== INDICADORES (visibles) ===================
st.subheader("📊 Indicadores")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Órdenes (filtrado)", f"{len(df_f):,}")
c2.metric("Usuarios únicos", df_f["Usuario"].nunique())
c3.metric("Turno 1", int((df_f["Turno"]=="Turno 1").sum()))
c4.metric("Turno 2", int((df_f["Turno"]=="Turno 2").sum()))
st.divider()

# =================== DESCARGA DEL EXCEL FILTRADO ===================
st.download_button(
    label="⬇️ Descargar Excel Completo",
    data=df.to_csv(index=False).encode("utf-8"),  # Aquí usa `df` sin filtros
    file_name="ordenes_completas_montacargas.csv",
    mime="text/csv"
)
