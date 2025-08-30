# =========================================================
# DASHBOARD MONTACARGAS — TM + Órdenes OT + Inicio/Fin (auto horas extra)
# =========================================================

# ---------------- [S0] Imports y setup -------------------
from cfg import APP_TITLE, APP_TAGLINE

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3, hashlib, re, unicodedata
from datetime import time as dtime
from typing import Optional, List
from collections import Counter

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------------- [S0.1] Cache compat -------------------
try:
    _cache = st.cache_data
    def CACHE(**kw): return _cache(**kw)
except AttributeError:
    def CACHE(**kw):
        def _wrap(f): return st.cache(f, allow_output_mutation=True, suppress_st_warning=True)
        return _wrap

def clear_cache_compat() -> bool:
    cleared = False
    for attr in ("cache_data", "cache_resource"):
        if hasattr(st, attr):
            try:
                getattr(st, attr).clear(); cleared = True
            except Exception:
                pass
    try:
        if hasattr(st, "caching") and hasattr(st.caching, "clear_cache"):
            st.caching.clear_cache(); cleared = True
    except Exception:
        pass
    return cleared

def rerun_compat():
    try: st.rerun()
    except Exception: st.experimental_rerun()

if not hasattr(st, "divider"):
    def _divider(): st.markdown("---")
    st.divider = _divider

# =========================================================
# [S1] Estilos (con modo oscuro global compatible) + Título responsive
# =========================================================
BASE_STYLE = f"""
<style>
:root{{
  --bg:#ffffff; --panel:#f8fafc; --ink:#0f172a; --muted:#64748b; --border:#e5e7eb;
  --accent:#0ea5e9;
  --header-h: 64px;
}}

/* ===== Header fijo + separación del contenido ===== */
header[data-testid="stHeader"]{{
  height: var(--header-h) !important;
  background-color: var(--bg) !important;
  border-bottom:1px solid var(--border) !important;
  z-index: 500; /* [MOBILE] asegura que no tape controles */
}}
/* Empuja el contenido para que el header no tape el hero */
[data-testid="stAppViewContainer"] > .main{{
  padding-top: calc(var(--header-h) + 12px) !important;
}}
/* Evita doble padding interno de Streamlit */
main .block-container{{ padding-top: 0 !important; }}

/* ===== Colores base ===== */
html, body, #root, .stApp,
main, .main,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"]{{
  background-color: var(--bg) !important;
  color: var(--ink) !important;
}}
section[data-testid="stSidebar"]{{
  background-color: var(--panel) !important;
  border-right:1px solid var(--border);
}}

/* ===== Tarjetas/títulos ===== */
div.hero{{ 
  margin: 0 !important;
  width: 100%;
  border:1px solid var(--border); border-radius:14px;
  padding:14px 16px; background:var(--panel);
}}
.hero-wrap{{ display:flex; flex-direction:column; gap:.25rem; width:100%; }}
h1.hero-title{{ 
  margin:0; line-height:1.15; font-weight:800; color:var(--ink);
  font-size: clamp(20px, 2.6vw + 8px, 34px);
  text-wrap: balance; overflow-wrap:anywhere;
}}
div.hero-sub{{ font-size:clamp(12px, 1.1vw + 8px, 15px); color:var(--muted); }}

h2.section-title{{ font-weight:700; font-size:18px; margin:0; color:var(--ink); }}
div.section{{ border:1px solid var(--border); border-radius:12px; padding:10px 12px; background:var(--panel); margin:14px 0 8px 0; }}

.note-box{{ border:1px solid var(--border); border-radius:12px; padding:12px 14px; background:var(--panel); color:var(--ink); font-size:14px; }}
.kpi-card{{ border:1px solid var(--border); background:var(--panel); border-radius:14px; padding:12px 14px; margin-left:10px; color:var(--ink); max-width:420px; }}
.kpi-title{{ display:flex; align-items:center; gap:8px; font-weight:800; font-size:20px; margin:2px 0 12px 0; }}
.kpi-grid{{ display:grid; grid-template-columns: 1fr; gap:12px; }}
.kpi-item .label{{ color:var(--muted); font-size:13px; margin-bottom:2px; }}
.kpi-item .value{{ color:var(--ink); font-size:32px; font-weight:800; }}

/* ===== Inputs ===== */
[data-baseweb="select"]>div{{ border-radius:10px; border:1px solid var(--border); background:var(--bg); }}
[data-baseweb="select"]>div:focus-within{{ box-shadow:0 0 0 2px var(--accent); border-color:var(--accent); }}
input, textarea{{ background:var(--bg)!important; color:var(--ink)!important;
  border-radius:10px!important; border:1px solid var(--border)!important; }}

/* Usuarios en ejes en negrita */
g.xtick text, g.ytick text{{ font-weight:700; }}

/* ===== [MOBILE] Flecha para abrir/cerrar sidebar siempre visible ===== */
[data-testid="collapsedControl"]{{
  position: fixed !important; /* [MOBILE] que no la tape nada */
  top: 10px; left: 10px;
  z-index: 2000 !important;
  display: flex !important; visibility: visible !important; opacity: 1 !important;
}}
[data-testid="collapsedControl"] svg{{ color: var(--ink) !important; fill: currentColor !important; }}

/* =========================
   [MOBILE] Ajustes responsivos
   ========================= */
@media (max-width: 768px){{
  /* [MOBILE] separa visualmente sidebar del tablero (no se "juntan") */
  section[data-testid="stSidebar"]{{
    position: relative !important;
    z-index: 1200 !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.35);
    border-right:1px solid var(--border);
  }}
  [data-testid="stAppViewContainer"]{{ position: relative; z-index: 1; }}

  /* [MOBILE] scrollbar interno del sidebar para que no invada el tablero */
  section[data-testid="stSidebar"] > div{{
    max-height: calc(100vh - var(--header-h) - 8px);
    overflow-y: auto;
  }}

  /* [MOBILE] padding contenido más compacto */
  .block-container{{ padding: 8px 10px !important; }}

  /* [MOBILE] textos de ejes/leyenda ligeramente más grandes */
  .js-plotly-plot .xtick text, .js-plotly-plot .ytick text{{ font-size:11px !important; }}
  .js-plotly-plot .legend text{{ font-size:11px !important; }}

  /* [MOBILE] evita cortes de labels */
  .main-svg{{ overflow: visible !important; }}
}}

/* [MOBILE] popover del datepicker y selects por encima (no se cierra al cambiar mes) */
[data-baseweb="popover"], .stDateInput [role="dialog"], .stDateInput div[data-baseweb="popover"]{{
  z-index: 3000 !important;
}}
</style>

<div class="hero"><div class="hero-wrap">
  <h1 class="hero-title">{APP_TITLE}</h1>
  <div class="hero-sub">{APP_TAGLINE}</div>
</div></div>
"""

st.markdown(BASE_STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    # [MOBILE] Modo oscuro por defecto para que app + gráficas arranquen consistentes
    dark = st.checkbox("🌙 Modo oscuro", value=True, help="Cambia colores (app + gráficas)")

if dark:
    st.markdown("""
    <style>
    :root{ --bg:#0b1220; --panel:#0f172a; --ink:#e5e7eb; --muted:#cbd5e1; --border:#1f2937; --accent:#22d3ee;}
    html, body, #root, .stApp, main, .main, .block-container,
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"], header[data-testid="stHeader"]{
      background-color: var(--bg) !important; color: var(--ink) !important;
    }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] *{
      background-color: transparent !important; color: var(--ink) !important;
    }
    .hero, .section, .note-box, .kpi-card,
    div[data-testid="stExpander"] details{ background: var(--panel) !important; color: var(--ink) !important; border-color: var(--border) !important; }
    [data-baseweb="select"] > div{ background: var(--panel) !important; border:1px solid var(--border)!important; border-radius:10px!important; }
    .stDateInput input, input, textarea{ background: var(--panel)!important; color: var(--ink)!important; border:1px solid var(--border)!important; }
    </style>
    """, unsafe_allow_html=True)

def apply_plot_theme(fig):
    is_dark = bool(dark)
    fig.update_layout(
        template=("plotly_dark" if is_dark else "plotly_white"),
        paper_bgcolor=("#0f172a" if is_dark else "#ffffff"),
        plot_bgcolor=("#0b1220" if is_dark else "#ffffff"),
        font=dict(color=("#e5e7eb" if is_dark else "#0f172a")),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
        hoverlabel=dict(font_size=12)  # [MOBILE]
    )
    # [MOBILE] ejes en enteros
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", tickformat=",d")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="", tickformat=",d")

ANN_COL  = "#e5e7eb" if dark else "#6B7280"

# =========================================================
# [S2] Reglas de turnos y paletas
# =========================================================
THRESH_MIN = 15
TURNOS = {
    "Turno A": {"start": dtime(5, 0),  "end": dtime(13, 55),
                "lunch": (dtime(8, 20), dtime(9, 0)),
                "fuel":  (dtime(9, 0),  dtime(9, 18))},
    "Turno B": {"start": dtime(14, 0), "end": dtime(22, 55),
                "lunch": (dtime(17, 25), dtime(18, 0)),
                "fuel":  (dtime(18, 0),  dtime(18, 18))},
}
LATE_B_CUTOFF = dtime(3, 0)

ITEM_WAZ = "WA-ZONE"
ITEM_BODEGA = "BODEGA-INT"
EXT_ITEMS = [
    "CALIDAD-OK","TRANSFER","INSPECCIÓN","CARPA",
    ITEM_WAZ, ITEM_BODEGA, "UBIC.SOBRESTOCK","REACOM.SOBRESTOCK"
]

ITEMS_HIDDEN = []  # puedes dejarlo vacío

PALETTES = {
  "Petróleo & Tierra": {
    "CALIDAD-OK":"#2A9D8F","TRANSFER":"#457B9D","INSPECCIÓN":"#3A6B35","CARPA":"#B65E3C",
    ITEM_WAZ:"#C4952B", ITEM_BODEGA:"#6C757D","UBIC.SOBRESTOCK":"#2F855A","REACOM.SOBRESTOCK":"#6D5BD0",
  },
  "Vibrante Tropical": {
    "CALIDAD-OK":"#00B894","TRANSFER":"#0984E3","INSPECCIÓN":"#6C5CE7","CARPA":"#E17055",
    ITEM_WAZ:"#FDCB6E", ITEM_BODEGA:"#636E72","UBIC.SOBRESTOCK":"#00CEC9","REACOM.SOBRESTOCK":"#A29BFE",
  },
  "Neón Pro": {
    "CALIDAD-OK":"#00E676","TRANSFER":"#2979FF","INSPECCIÓN":"#00B0FF","CARPA":"#FF5252",
    ITEM_WAZ:"#FFD54F", ITEM_BODEGA:"#90A4AE","UBIC.SOBRESTOCK":"#69F0AE","REACOM.SOBRESTOCK":"#7C4DFF",
  }
}

# =========================================================
# [S3] Normalización / clasificación
# =========================================================
def _norm_nfd_ascii(s: str) -> str:
    if s is None: return ""
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()

def _norm_compact(s: str) -> str:
    s = _norm_nfd_ascii(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def _norm_tokens(s: str) -> List[str]:
    s = _norm_nfd_ascii(s).lower()
    toks = re.split(r"[^a-z0-9]+", s)
    return [t for t in toks if t]

def pick_col(cols_map: dict, *aliases) -> Optional[str]:
    def normkey(x): return _norm_compact(x)
    norm2orig = {normkey(k): v for k, v in cols_map.items()}
    for alias in aliases:
        key = normkey(alias)
        for nk, orig in norm2orig.items():
            if key in nk or nk in key: return orig
    return None

# [FIX-HORA] Parser robusto para “5:34:48 a. m.” / “p. m.”, NBSP y 12/24h
def to_time(x):
    if pd.isna(x):
        return None
    try:
        if hasattr(x, "hour"):
            return dtime(int(x.hour), int(getattr(x, "minute", 0)), int(getattr(x, "second", 0)))
    except Exception:
        pass
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            dtv = pd.to_datetime(x, unit="d", origin="1899-12-30")
            return dtime(int(dtv.hour), int(dtv.minute), int(dtv.second))
        except Exception:
            pass

    s = str(x).strip()
    if not s:
        return None
    # normaliza espacios duros y AM/PM español
    s_norm = s.replace("\u00A0", " ").replace("\u202F", " ")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    s_norm = re.sub(r"(?i)\ba\.?\s*m\.?\b", "AM", s_norm)
    s_norm = re.sub(r"(?i)\bp\.?\s*m\.?\b", "PM", s_norm)
    s_norm = s_norm.replace("a.m.", "AM").replace("p.m.", "PM").replace("a.m", "AM").replace("p.m", "PM")

    for fmt in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
        try:
            dtv = pd.to_datetime(s_norm, format=fmt)
            return dtime(int(dtv.hour), int(dtv.minute), int(getattr(dtv, "second", 0)))
        except Exception:
            pass
    dtv = pd.to_datetime(s_norm, errors="coerce")
    if pd.notnull(dtv):
        return dtime(int(dtv.hour), int(dtv.minute), int(getattr(dtv, "second", 0)))
    return None

def turno_by_time(t: dtime):
    if t is None: return None
    a_start, a_end = TURNOS["Turno A"]["start"], TURNOS["Turno A"]["end"]
    b_start, b_end = TURNOS["Turno B"]["start"], TURNOS["Turno B"]["end"]
    if a_start <= t < a_end:  return "Turno A"
    if b_start <= t <= b_end: return "Turno B"
    if t < LATE_B_CUTOFF or t > b_end: return "Turno B"
    return None

def has_003(s: str) -> bool:
    s = str(s or "").upper().strip()
    return s.startswith("003") or bool(re.search(r'(^|[^0-9])003([^0-9]|$)', s))

def looks_like_slot_code(s: str) -> bool:
    s = str(s or "").upper().strip()
    if not s or s.startswith("003"): return False
    return bool(re.fullmatch(r'[0-9A-Z][0-9A-Z/-]*', s))

def _contains_any(txt: str, patterns: List[str]) -> bool:
    return any(p in txt for p in patterns)

def _item_base(ubic_proced: str, ubic_dest: str) -> Optional[str]:
    up = str(ubic_proced or ""); ud = str(ubic_dest or "")
    both_txt = f"{up} | {ud}"; compact = _norm_compact(both_txt); toks = set(_norm_tokens(both_txt))
    if "wazone" in compact or ("wa" in toks and "zone" in toks) or "zonav" in compact: return ITEM_WAZ  # [BUGFIX-CONSTANTE]
    if _contains_any(compact, ["transfer","traslado","trasl","transferen","trasfer","transfe","transf"]): return "TRANSFER"
    if _contains_any(compact, ["inspeccion","inspection","inspec","insp","insp."]): return "INSPECCIÓN"
    if _contains_any(compact, ["carpa","carpas"]): return "CARPA"
    if ("calidad" in toks and "ok" in toks) or "calidadok" in compact or "okcalidad" in compact: return "CALIDAD-OK"
    return None

def _norm_label(s: str) -> str:
    s = _norm_nfd_ascii(str(s or "")).lower()
    s = re.sub(r"[^a-z0-9\s\-_/\.]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

CANON_MAP = {
    "calidad ok": "CALIDAD-OK", "calidad-ok": "CALIDAD-OK",
    "transfer": "TRANSFER",
    "inspeccion": "INSPECCIÓN", "inspección": "INSPECCIÓN", "inspection": "INSPECCIÓN", "inspec": "INSPECCIÓN", "insp": "INSPECCIÓN", "insp.": "INSPECCIÓN",
    "carpa": "CARPA", "carpas": "CARPA",
    "wa zone": "WA-ZONE", "wa-zone": "WA-ZONE", "wazone": "WA-ZONE",
    "bodega int": "BODEGA-INT", "bodega-int": "BODEGA-INT", "movimiento en bodega": "BODEGA-INT",
    "ubic.sobrestock": "UBIC.SOBRESTOCK", "ubicacion sobrestock": "UBIC.SOBRESTOCK",
    "reacom.sobrestock": "REACOM.SOBRESTOCK", "reacomodacion sobrestock": "REACOM.SOBRESTOCK",
}

def canon_item_from_text(s: str):
    t = _norm_label(s)
    if not t: return None
    if t in CANON_MAP: return CANON_MAP[t]
    for k, v in CANON_MAP.items():
        if t == k or t.replace("-", " ") == k or k in t: return v
    return None

def item_ext(ubic_proced: str, ubic_dest: str) -> Optional[str]:
    up = str(ubic_proced or "").strip(); ud = str(ubic_dest or "").strip()
    up003 = has_003(up); ud003 = has_003(ud)
    if up003 and ud003:     return "REACOM.SOBRESTOCK"
    if up003 and not ud003: return "REACOM.SOBRESTOCK"
    if ud003 and not up003: return "UBIC.SOBRESTOCK"
    base = _item_base(up, ud)
    if base: return base
    if looks_like_slot_code(up) and looks_like_slot_code(ud): return ITEM_BODEGA
    return None

# =========================================================
# [S4] SQLite (persistencia)
# =========================================================
TABLE = "ordenes"
def ensure_db(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            id TEXT PRIMARY KEY,
            usuario TEXT, fecha TEXT, time TEXT, turno TEXT, datetime TEXT,
            orden TEXT, ubic_proced TEXT, ubic_destino TEXT, itemraw TEXT
        )
    """)
    cur.execute(f"PRAGMA table_info({TABLE})")
    existing_cols = {row[1].lower() for row in cur.fetchall()}
    required = ["id","usuario","fecha","time","turno","datetime","orden","ubic_proced","ubic_destino","itemraw"]
    for col in required:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {col} TEXT")
    con.commit(); con.close()

def make_uid(row):
    dtv = pd.to_datetime(row.get("Datetime"), errors="coerce")
    dtv = pd.NaT if pd.isna(dtv) else dtv.floor("min")
    base = f"{row.get('Usuario','')}|{row.get('Fecha','')}|{str(dtv)}|{str(row.get('Orden') or '')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def upsert_df(path, df):
    if df.empty: return 0
    con = sqlite3.connect(path); cur = con.cursor()
    df2 = df.copy(); df2["Datetime"] = pd.to_datetime(df2["Datetime"]).dt.floor("min")
    df2["id"] = df2.apply(make_uid, axis=1)
    rows = []
    for _, r in df2.iterrows():
        rows.append((r["id"], r.get("Usuario"),
                     str(r.get("Fecha")) if pd.notnull(r.get("Fecha")) else None,
                     str(r.get("Time")) if pd.notnull(r.get("Time")) else None,
                     r.get("Turno"),
                     r.get("Datetime").isoformat() if pd.notnull(r["Datetime"]) else None,
                     str(r.get("Orden") or None),
                     r.get("Ubic.proced"), r.get("Ubicación de destino"),
                     r.get("ItemRaw") if "ItemRaw" in df2.columns else None))
    cur.executemany(
        f"""INSERT OR REPLACE INTO {TABLE}
            (id,usuario,fecha,time,turno,datetime,orden,ubic_proced,ubic_destino,itemraw)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows
    )
    con.commit(); con.close()
    return len(rows)

def read_all(path) -> pd.DataFrame:
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
    except Exception:
        df = pd.DataFrame(columns=["id","usuario","fecha","time","turno","datetime","orden","ubic_proced","ubic_destino","itemraw"])
    con.close()
    if df.empty: return df
    df.rename(columns={"usuario":"Usuario","fecha":"Fecha","time":"TimeStr",
                       "turno":"Turno","datetime":"Datetime","orden":"Orden",
                       "ubic_proced":"Ubic.proced","ubic_destino":"Ubicación de destino",
                       "itemraw":"ItemRaw"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True).dt.date  # [FECHA-DDMM]
    df["Time"] = df["Datetime"].dt.time
    return df

def clear_db(path):
    con = sqlite3.connect(path); con.execute(f"DELETE FROM {TABLE}"); con.commit(); con.close()

# =========================================================
# [S5] Fecha operativa + marca de horas extra
# =========================================================
def apply_oper_day(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Turno" not in d.columns or d["Turno"].isna().any():
        d["Turno"] = d["Time"].apply(turno_by_time)

    extra_mask = (d["Turno"] == "Turno B") & (d["Time"].apply(lambda x: x is not None and x < LATE_B_CUTOFF))
    d["IsExtra"] = extra_mask

    d["FechaOper"] = pd.to_datetime(d["Fecha"])
    d.loc[extra_mask, "FechaOper"] = d.loc[extra_mask, "FechaOper"] - pd.Timedelta(days=1)

    d["DatetimeOper"] = d.apply(
        lambda r: pd.Timestamp.combine(pd.Timestamp(r["FechaOper"]).date(), r["Time"]) if (pd.notnull(r["FechaOper"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    return d

# =========================================================
# [S6] Carga Excel
# =========================================================
@CACHE(show_spinner=False)
def load_excel(file, sheet_name="Hoja1") -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    if sheet_name not in xls.sheet_names: sheet_name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    cols = {c.lower().strip(): c for c in df.columns}

    col_usuario = pick_col(cols, "usuario")
    col_fecha   = pick_col(cols, "fecha", "fecha confirmacion")
    col_hora    = pick_col(cols, "hora", "hora confirmacion")
    col_orden   = pick_col(cols, "orden", "numero de orden de transporte", "ot")
    col_uproc   = pick_col(cols, "ubic.proced", "ubic proced", "ubic procedencia", "ubicacion procedencia", "origen", "ubic origen")
    col_udest   = pick_col(cols, "ubicacion de destino", "ubic destino", "destino", "ubic. destino")
    col_item    = pick_col(cols, "item","ítem","movimiento","tipo movimiento","tipo_movimiento","actividad","proceso","tarea","categoria","clase","operacion","detalle movimiento","detalle del movimiento","mov","nombre movimiento")

    if not col_usuario: raise ValueError("Falta columna 'Usuario'.")
    if not col_fecha:   raise ValueError("Falta columna 'Fecha'.")
    if not col_hora:    raise ValueError("Falta columna 'Hora'.")

    out = pd.DataFrame({
        "Usuario": df[col_usuario].astype(str).str.strip(),
        "Fecha":   pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True).dt.date,  # [FECHA-DDMM]
        "Time":    df[col_hora].apply(to_time),
    })
    out["Hora"] = out["Time"].apply(lambda t: t.hour if t else None)
    out["Orden"] = df[col_orden].astype(str).str.strip() if col_orden else None
    out["Ubic.proced"] = df[col_uproc].astype(str).str.strip() if col_uproc else None
    out["Ubicación de destino"] = df[col_udest].astype(str).str.strip() if col_udest else None
    out["ItemRaw"] = df[col_item].astype(str).str.strip() if col_item else None

    out["Turno"] = out["Time"].apply(turno_by_time)
    out["Datetime"] = out.apply(
        lambda r: pd.Timestamp.combine(r["Fecha"], r["Time"]) if (pd.notnull(r["Fecha"]) and r["Time"] is not None) else pd.NaT,
        axis=1
    )
    out = out.dropna(subset=["Fecha","Time","Datetime"])
    return out[["Usuario","Fecha","Hora","Time","Turno","Datetime","Orden","Ubic.proced","Ubicación de destino","ItemRaw"]]

# =========================================================
# [S7] Sidebar: carga + preferencias + filtros
# =========================================================
with st.sidebar:
    with st.expander("📥 Carga de datos", expanded=False):
        up = st.file_uploader("📎 Excel (.xlsx)", type=["xlsx"])
        if up is not None:
            try:
                xls_tmp = pd.ExcelFile(up); hojas = xls_tmp.sheet_names
                hoja = st.selectbox("Hoja", hojas, index=hojas.index("Hoja1") if "Hoja1" in hojas else 0)
            except Exception:
                hoja = st.text_input("Hoja", value="Hoja1")
            finally:
                if hasattr(up,"seek"): up.seek(0)
        else:
            hoja = st.text_input("Hoja", value="Hoja1")

        st.caption("Histórico SQLite")
        use_db = st.checkbox("Usar histórico", value=True)
        DB_PATH = st.text_input("Archivo DB", value="montacargas.db")
        col1, col2 = st.columns(2)
        with col1: btn_clear = st.button("🧹 Limpiar histórico")
        with col2: btn_reload = st.button("🔁 Recargar histórico")

    with st.expander("⚙️ Preferencias", expanded=False):
        chart_type = st.selectbox("Orientación (Gráfica TM)", ["Barra horizontal", "Barra vertical"])
        pal_name = st.selectbox("🎨 Paleta", list(PALETTES.keys()), index=0)
        st.session_state["pal_name"] = pal_name
        st.session_state["chart_type"] = chart_type

EXT_COLOR_MAP = PALETTES[st.session_state.get("pal_name", "Petróleo & Tierra")]

if up is None and 'use_db' in locals() and not use_db:
    st.warning("Sube un Excel para empezar o activa el histórico."); st.stop()

df_new = pd.DataFrame()
if up is not None:
    try: df_new = load_excel(up, hoja)
    except Exception as e:
        st.error(f"❌ No pude leer el Excel: {e}"); st.stop()

if 'use_db' not in locals(): use_db = True
if 'DB_PATH' not in locals(): DB_PATH = "montacargas.db"

ensure_db(DB_PATH)
if use_db:
    if btn_clear: clear_db(DB_PATH); st.success("Histórico limpiado.")
    if not df_new.empty: upsert_df(DB_PATH, df_new)
    if btn_reload: clear_cache_compat(); st.success("Recargado."); rerun_compat()
    df = read_all(DB_PATH)
else:
    df = df_new.copy()

if df.empty:
    st.info("No hay datos para visualizar aún."); st.stop()

# --- Fecha operativa aplicada + marca IsExtra ---
df = apply_oper_day(df)
df = df[df["Turno"].isin(["Turno A","Turno B"])].copy()

# ---------------- Filtros ----------------
with st.sidebar:
    users = sorted(df["Usuario"].dropna().unique().tolist())
    turns = ["Turno A","Turno B"]
    fmin, fmax = df["FechaOper"].min().date(), df["FechaOper"].max().date()

    sel_users = st.multiselect("Usuarios", users, [])
    sel_turns = st.multiselect("Turnos", turns, [])
    sel_range = st.date_input("Rango de fechas", (fmin, fmax), key="date_range", format="YYYY-MM-DD")  # [MOBILE] clave estable

start_ts, end_ts = (
    (pd.Timestamp(sel_range[0]), pd.Timestamp(sel_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    if isinstance(sel_range, (list, tuple)) and len(sel_range) == 2
    else (pd.Timestamp(fmin), pd.Timestamp(fmax) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
)
df_pre = df.copy()
if sel_users: df_pre = df_pre[df_pre["Usuario"].isin(sel_users)]
if sel_turns: df_pre = df_pre[df_pre["Turno"].isin(sel_turns)]
df_pre = df_pre[(df_pre["DatetimeOper"] >= start_ts) & (df_pre["DatetimeOper"] <= end_ts)]

def classify_any(row) -> Optional[str]:
    raw = canon_item_from_text(row.get("ItemRaw"))
    if raw: return raw
    base = item_ext(row.get("Ubic.proced"), row.get("Ubicación de destino"))
    return base

if "ItemExt" not in df_pre.columns:
    df_pre["ItemExt"] = df_pre.apply(lambda r: classify_any(r), axis=1)

# --------- Ítems sin chips preseleccionados (sidebar derecha) ---------
avail_items = [it for it in EXT_ITEMS if it in set(df_pre["ItemExt"].dropna().unique().tolist())]
default_items = []  # nada preseleccionado
with st.sidebar:
    sel_items = st.multiselect("Ítems", avail_items, default_items, key="items_selector")
# Dataset FINAL: si no hay selección, usar todo
df_f = df_pre[df_pre["ItemExt"].isin(sel_items)].copy() if sel_items else df_pre.copy()
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.download_button("⬇️ Descargar filtrado (CSV)", data=df_f.to_csv(index=False).encode("utf-8"),
                       file_name="filtrado_montacargas.csv", mime="text/csv")

if df_f.empty:
    st.info("No hay datos con el filtro actual."); st.stop()

# =========================================================
# [S8] Helpers visuales / KPIs
# =========================================================
def render_section_title(txt:str):
    st.markdown(f'<div class="section"><h2 class="section-title">{txt}</h2></div>', unsafe_allow_html=True)

def _short_label(user: str, turnoAB: str) -> str:
    return f"{str(user).strip()}-{turnoAB}"

def _responsive_bar_style(fig, n_categories:int):
    if n_categories <= 6:        bargap, bgrp = 0.12, 0.05
    elif n_categories <= 12:     bargap, bgrp = 0.10, 0.04
    elif n_categories <= 24:     bargap, bgrp = 0.08, 0.03
    else:                        bargap, bgrp = 0.06, 0.02
    fig.update_layout(bargap=bargap, bargroupgap=bgrp)

def render_kpis(df_filtered: pd.DataFrame):
    total = len(df_filtered); uniq_users = df_filtered["Usuario"].nunique()
    tA = int((df_filtered["Turno"] == "Turno A").sum())
    tB = int((df_filtered["Turno"] == "Turno B").sum())
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">📊 Indicadores</div>
  <div class="kpi-grid">
    <div class="kpi-item"><div class="label">Órdenes (filtrado)</div><div class="value">{total:,}</div></div>
    <div class="kpi-item"><div class="label">Usuarios únicos</div><div class="value">{uniq_users}</div></div>
    <div class="kpi-item"><div class="label">Turno A</div><div class="value">{tA:,}</div></div>
    <div class="kpi-item"><div class="label">Turno B</div><div class="value">{tB:,}</div></div>
  </div>
</div>
    """, unsafe_allow_html=True)

# =========================================================
# [S9] Vista 1 — TM por usuario/turno (usa df_f con filtro de ítems)
# =========================================================
def view_tm_por_usuario_turno():
    render_section_title("Tiempo Muerto — dos barras por Usuario (Turno A y B), apilado por ítem")

    dtmp = df_f.copy()
    df_g = dtmp.sort_values(["Usuario","DatetimeOper"]).copy()
    df_g["prev_dt"] = df_g.groupby("Usuario")["DatetimeOper"].shift(1)
    df_g["prev_fecha"] = df_g.groupby("Usuario")["FechaOper"].shift(1)
    df_g["prev_turno"] = df_g.groupby("Usuario")["Turno"].shift(1)
    same = df_g["prev_dt"].notna() & (df_g["FechaOper"]==df_g["prev_fecha"]) & (df_g["Turno"]==df_g["prev_turno"])
    df_g = df_g[same].copy()

    EXC = [TURNOS["Turno A"]["lunch"], TURNOS["Turno A"]["fuel"],
           TURNOS["Turno B"]["lunch"], TURNOS["Turno B"]["fuel"]]
    rows = []
    for _, r in df_g.iterrows():
        start = r["prev_dt"]; end = r["DatetimeOper"]; date = r["FechaOper"]
        gap = (end - start).total_seconds()/60.0
        if gap <= 0: continue

        def subtract_window(seg_start, seg_end, win_start, win_end):
            if win_end <= seg_start or win_start >= seg_end:
                return [(seg_start, seg_end)]
            parts = []
            if seg_start < win_start: parts.append((seg_start, max(seg_start, win_start)))
            if seg_end > win_end:     parts.append((min(seg_end, win_end), seg_end))
            return [(s,e) for (s,e) in parts if e > s]
        def subtract_windows(seg_start, seg_end, date, windows):
            segs = [(seg_start, seg_end)]
            for st_t, en_t in windows:
                st_w = pd.Timestamp.combine(pd.Timestamp(date), st_t)
                en_w = pd.Timestamp.combine(pd.Timestamp(date), en_t)
                new = []
                for s, e in segs:
                    new.extend(subtract_window(s, e, st_w, en_w))
                segs = new
                if not segs: break
            return segs

        segs = subtract_windows(start, end, date, EXC)
        if not segs: continue
        adj = sum((e - s).total_seconds()/60.0 for s, e in segs)
        if adj > THRESH_MIN:
            rows.append({"Usuario": r["Usuario"], "Turno": r["Turno"], "ItemExt": r["ItemExt"], "AdjMin": adj})

    dead_ext = pd.DataFrame(rows)
    if dead_ext.empty:
        st.info("No se detectó TM > 15 min con el filtro actual."); return

    tm_ut = dead_ext.groupby(["Usuario","Turno","ItemExt"])["AdjMin"].sum().reset_index()
    tm_ut = tm_ut[tm_ut["ItemExt"].isin(sel_items)] if sel_items else tm_ut

    g = tm_ut.copy()
    g["TurnoAB"] = g["Turno"].str.replace("Turno ","", regex=False)
    g["UsuarioTurnoShort"] = g.apply(lambda r: _short_label(r["Usuario"], r["TurnoAB"]), axis=1)

    order_users = (g.groupby("Usuario")["AdjMin"].sum().sort_values(ascending=False).index.tolist())
    order_axis, present_keys = [], set(g["UsuarioTurnoShort"])
    for u in order_users:
        for ab in ["A","B"]:
            key = _short_label(u, ab)
            if key in present_keys: order_axis.append(key)

    g = g.rename(columns={"AdjMin":"Min"})
    hover_tmpl_h = "Ítem: %{customdata[0]}<br>Minutos TM: %{customdata[1]:.0f}m<extra></extra>"

    chart_is_h = (st.session_state.get("chart_type", "Barra horizontal") == "Barra horizontal")
    if chart_is_h:
        height = max(320, 24*len(order_axis) + 110)  # [MOBILE]
        fig = px.bar(g, x="Min", y="UsuarioTurnoShort", color="ItemExt", orientation="h",
                     barmode="stack",
                     category_orders={"UsuarioTurnoShort": order_axis, "ItemExt": (sel_items if sel_items else avail_items)},
                     color_discrete_map=EXT_COLOR_MAP,
                     custom_data=["ItemExt","Min"], height=height)
        fig.update_traces(hovertemplate=hover_tmpl_h, marker_line_width=0, opacity=0.95, cliponaxis=False)
        fig.update_yaxes(categoryorder="array", categoryarray=order_axis, tickfont=dict(size=12))

        totals = (g.groupby("UsuarioTurnoShort")["Min"].sum().reindex(order_axis))
        fig.add_trace(go.Scatter(x=totals.values, y=totals.index.tolist(), mode="text",
                                 text=[f"{v:.0f} min" for v in totals.values],
                                 textposition="middle right", textfont=dict(size=12, color=ANN_COL),
                                 showlegend=False, hoverinfo="skip"))
        xmax = max(1, float(totals.max()))
        fig.update_xaxes(range=[0, xmax*1.06], tickfont=dict(size=12))
        _responsive_bar_style(fig, len(order_axis))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=110), legend_title_text="Ítem")
    else:
        height = max(420, 24*len(order_axis) + 60)  # [MOBILE]
        fig = px.bar(g, x="UsuarioTurnoShort", y="Min", color="ItemExt", barmode="stack",
                     category_orders={"UsuarioTurnoShort": order_axis, "ItemExt": (sel_items if sel_items else avail_items)},
                     color_discrete_map=EXT_COLOR_MAP,
                     custom_data=["ItemExt","Min"], height=height)
        fig.update_traces(hovertemplate=hover_tmpl_h, marker_line_width=0, opacity=0.95, cliponaxis=False)
        tick_angle = -65 if len(order_axis) > 8 else -30  # [MOBILE]
        fig.update_xaxes(categoryorder="array", categoryarray=order_axis, tickangle=tick_angle, tickfont=dict(size=10))
        totals = (g.groupby("UsuarioTurnoShort")["Min"].sum().reindex(order_axis))
        ymax = float(totals.max())*1.18
        fig.update_yaxes(range=[0, ymax])
        fig.add_trace(go.Bar(x=totals.index.tolist(), y=totals.values,
                             marker_color='rgba(0,0,0,0)', showlegend=False, hoverinfo="skip",
                             text=[f"{v:.0f}" for v in totals.values],  # [MOBILE] sin "min"
                             textposition="outside", textfont=dict(size=10, color=ANN_COL), cliponaxis=False))
        _responsive_bar_style(fig, len(order_axis))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), legend_title_text="Ítem")

    apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# [S10] Vista 2 — Órdenes OT por usuario/turno (usa df_f)
# =========================================================
def view_ordenes_ot():
    render_section_title("Órdenes OT — total de movimientos por usuario y turno")

    cnt = (df_f.groupby(["Usuario","Turno","ItemExt"]).size().reset_index(name="CNT"))
    if cnt.empty:
        st.info("No hay órdenes en el filtro actual para 'Órdenes OT'."); return

    cnt["TurnoAB"] = cnt["Turno"].str.replace("Turno ","", regex=False)
    cnt["UsuarioTurnoShort"] = cnt.apply(lambda r: _short_label(r["Usuario"], r["TurnoAB"]), axis=1)

    order_users = (cnt.groupby("Usuario")["CNT"].sum().sort_values(ascending=False).index.tolist())
    order_axis, present_keys = [], set(cnt["UsuarioTurnoShort"])
    for u in order_users:
        for ab in ["A","B"]:
            k = _short_label(u, ab)
            if k in present_keys: order_axis.append(k)

    n_bars = len(order_axis)
    show_totals = n_bars <= 12  # [MOBILE]
    tick_angle = -65 if n_bars > 8 else -30

    # [MOBILE] Muchísimos usuarios => barras horizontales
    if n_bars > 14:
        hover_tmpl_h = "Ítem: %{customdata[0]}<br>Órdenes: %{customdata[1]:.0f}<br>%{customdata[2]}<extra></extra>"
        height = max(440, 22*n_bars + 120)
        fig = px.bar(
            cnt, y="UsuarioTurnoShort", x="CNT", color="ItemExt", barmode="stack",
            category_orders={"UsuarioTurnoShort": order_axis, "ItemExt": (sel_items if sel_items else avail_items)},
            color_discrete_map=EXT_COLOR_MAP,
            custom_data=["ItemExt","CNT","UsuarioTurnoShort"], height=height, orientation="h"
        )
        fig.update_traces(hovertemplate=hover_tmpl_h, marker_line_width=0, opacity=0.95, cliponaxis=False)
        totals = (cnt.groupby("UsuarioTurnoShort")["CNT"].sum().reindex(order_axis))
        fig.add_trace(go.Scatter(x=totals.values, y=totals.index.tolist(), mode="text",
                                 text=[f"{int(v):,}".replace(",", ".") for v in totals.values],
                                 textposition="middle right", textfont=dict(size=11, color=ANN_COL),
                                 showlegend=False, hoverinfo="skip"))
        xmax = max(1, float(totals.max()))
        fig.update_xaxes(range=[0, xmax*1.08])
        _responsive_bar_style(fig, n_bars)
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=110), legend_title_text="Ítem")
    else:
        hover_tmpl = "Ítem: %{customdata[0]}<br>Órdenes: %{customdata[1]:.0f}<br>%{customdata[2]}<extra></extra>"
        height = max(520, 26*n_bars + 100)  # [MOBILE]
        fig = px.bar(
            cnt, x="UsuarioTurnoShort", y="CNT", color="ItemExt", barmode="stack",
            category_orders={"UsuarioTurnoShort": order_axis, "ItemExt": (sel_items if sel_items else avail_items)},
            color_discrete_map=EXT_COLOR_MAP,
            custom_data=["ItemExt","CNT","UsuarioTurnoShort"], height=height
        )
        fig.update_traces(hovertemplate=hover_tmpl, marker_line_width=0, opacity=0.95, cliponaxis=False)
        fig.update_xaxes(categoryorder="array", categoryarray=order_axis, tickangle=tick_angle, tickfont=dict(size=10))

        totals = (cnt.groupby("UsuarioTurnoShort")["CNT"].sum().reindex(order_axis))
        max_digits = len(str(int(totals.max()))) if len(totals) else 1
        lab_size = max(10, min(14, 15 - max(0, max_digits - 3)))
        pad_frac = 0.16 + 0.01 * (lab_size - 10)
        y_max = float(totals.max()) * (1 + pad_frac)
        fig.update_yaxes(range=[0, y_max], automargin=True)

        if show_totals:
            pixel_up = 6 + lab_size * 1.0
            annotations = []
            for x_val, y_val in totals.items():
                annotations.append(dict(
                    x=x_val, y=y_val, xref="x", yref="y",
                    text=f"{int(y_val):,}".replace(",", "."),
                    showarrow=False, yanchor="bottom", yshift=pixel_up,
                    align="center", font=dict(size=lab_size, color=ANN_COL)
                ))
            prev = list(fig.layout.annotations) if fig.layout.annotations else []
            fig.update_layout(annotations=prev + annotations)

        _responsive_bar_style(fig, n_bars)
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=100), legend_title_text="Ítem")

    apply_plot_theme(fig)

    c1, c2 = st.columns([3, 1])
    with c1: st.plotly_chart(fig, use_container_width=True)
    with c2: render_kpis(df_f)

# =========================================================
# [S11] Vista 3 — Inicio/Fin con +24h solo cuando hubo extra (usa df_pre)
# =========================================================
def minutes_of_day(ts: pd.Timestamp) -> float:
    t = ts.time(); return t.hour*60 + t.minute + t.second/60.0

def fmt_hhmm(minutes: float) -> str:
    m = int(round(minutes)); h = m // 60; mm = m % 60
    return f"{h:02d}:{mm:02d}"

def minutes_for_plot(ts: pd.Timestamp, turno: str) -> float:
    m = minutes_of_day(ts)
    if turno == "Turno B" and ts.time() < LATE_B_CUTOFF:
        m += 24*60
    return m

def most_common(lst: List[str]) -> str:
    lst = [x for x in lst if x and str(x).strip() != ""]
    return Counter(lst).most_common(1)[0][0] if lst else "—"

def classify_any_row(row) -> str:
    raw = canon_item_from_text(row.get("ItemRaw"))
    if raw: return raw
    base = item_ext(row.get("Ubic.proced"), row.get("Ubicación de destino"))
    return base if base else "—"

def view_inicio_fin_turno():
    render_section_title("Inicio y fin de turno — hora por Usuario/Turno (promedio o real)")

    d = df_pre.copy()
    if "ItemExt_any" not in d.columns:
        d["ItemExt_any"] = d.apply(classify_any_row, axis=1)

    recs = []
    for (usr, fecha_op, turno), g in d.sort_values("DatetimeOper").groupby(["Usuario","FechaOper","Turno"]):
        if g.empty: continue

        g = g.copy()
        g["t_vis"] = g["DatetimeOper"].apply(lambda ts: minutes_for_plot(ts, turno))

        r_ini = g.loc[g["t_vis"].idxmin()]
        t_ini_vis = float(r_ini["t_vis"]); it_ini = r_ini["ItemExt_any"]

        lunch_start, _ = TURNOS[turno]["lunch"]
        lunch_mins = lunch_start.hour*60 + lunch_start.minute
        g_pre_l = g[g["t_vis"] < lunch_mins]
        if not g_pre_l.empty:
            r_al = g_pre_l.loc[g_pre_l["t_vis"].idxmax()]
            t_alim_vis = float(r_al["t_vis"]); it_alim = r_al["ItemExt_any"]
        else:
            t_alim_vis = None; it_alim = None

        r_cie = g.loc[g["t_vis"].idxmax()]
        t_cie_vis = float(r_cie["t_vis"]); it_cie = r_cie["ItemExt_any"]

        had_extra = bool(g["IsExtra"].any())

        recs.append({
            "Usuario": usr, "Turno": turno, "FechaOper": fecha_op,
            "t_ini": t_ini_vis, "t_alim": t_alim_vis, "t_cie": t_cie_vis,
            "it_ini": it_ini, "it_alim": it_alim if t_alim_vis is not None else "—", "it_cie": it_cie,
            "extra": had_extra
        })
    if not recs:
        st.info("No se pudieron calcular hitos con el filtro actual."); 
        return

    dd = pd.DataFrame(recs)
    one_day = (dd["FechaOper"].nunique() == 1)

    agg_rows = []
    for (usr, turno), g in dd.groupby(["Usuario","Turno"]):
        if one_day:
            r = g.iloc[-1]; n_dias = 1
            t_ini, t_alim, t_cie = r["t_ini"], r["t_alim"], r["t_cie"]
            it_ini, it_alim, it_cie = r["it_ini"], r["it_alim"], r["it_cie"]
            modo = "Día único"
            extra_info = "Sí" if bool(r["extra"]) else "No"
            extra_days = int(bool(r["extra"]))
        else:
            n_dias = g["FechaOper"].nunique()
            t_ini  = g["t_ini"].mean(skipna=True)
            t_alim = g["t_alim"].mean(skipna=True)
            t_cie  = g["t_cie"].mean(skipna=True)
            it_ini  = most_common(g["it_ini"].tolist())
            it_alim = most_common(g["it_alim"].tolist())
            it_cie  = most_common(g["it_cie"].tolist())
            modo = f"Promedio de {n_dias} días"
            extra_days = int(g["extra"].sum())
            extra_info = f"{extra_days} de {n_dias} días con extra"

        agg_rows.append({
            "Usuario": usr, "Turno": turno, "n_dias": n_dias, "modo": modo,
            "t_ini": t_ini, "t_alim": t_alim, "t_cie": t_cie,
            "it_ini": it_ini, "it_alim": it_alim, "it_cie": it_cie,
            "extra_days": extra_days, "extra_info": extra_info
        })
    agg = pd.DataFrame(agg_rows)
    if agg.empty: st.info("No hay agregaciones para mostrar."); return

    agg["TurnoAB"] = agg["Turno"].str.replace("Turno ","", regex=False)
    agg["UsuarioTurnoShort"] = agg.apply(lambda r: _short_label(r["Usuario"], r["TurnoAB"]), axis=1)
    order_axis = sorted(agg["UsuarioTurnoShort"].unique().tolist())

    rows = []
    for _, r in agg.iterrows():
        t_ini = r["t_ini"]; t_alim = r["t_alim"] if pd.notna(r["t_alim"]) else None
        base_for_end = t_alim if t_alim is not None else t_ini
        seg_ini  = max(0.0, float(t_ini))
        seg_alim = max(0.0, float(t_alim - t_ini)) if t_alim is not None else 0.0
        seg_cie  = max(0.0, float(r["t_cie"] - base_for_end))
        rows += [
            {"UsuarioTurnoShort": r["UsuarioTurnoShort"], "Hito": "Inicio",
             "Seg": seg_ini, "Hora": fmt_hhmm(t_ini), "Info": r["modo"], "Item": r["it_ini"], "Extra": r["extra_info"]},
            {"UsuarioTurnoShort": r["UsuarioTurnoShort"], "Hito": "Antes de alimentación",
             "Seg": seg_alim, "Hora": fmt_hhmm(t_alim) if t_alim is not None else "—",
             "Info": r["modo"], "Item": r["it_alim"] if t_alim is not None else "—", "Extra": r["extra_info"]},
            {"UsuarioTurnoShort": r["UsuarioTurnoShort"], "Hito": "Antes de cierre",
             "Seg": seg_cie, "Hora": fmt_hhmm(r["t_cie"]), "Info": r["modo"], "Item": r["it_cie"], "Extra": r["extra_info"]},
        ]
    m = pd.DataFrame(rows)
    top_per_bar = m.groupby("UsuarioTurnoShort")["Seg"].sum().reindex(order_axis)

    has_any_extra = (agg["extra_days"] > 0).any()
    base_max = 27*60 if has_any_extra else 24*60
    y_max = max(base_max, int(top_per_bar.max()//60 + 2)*60)
    ticks = list(range(0, y_max+1, 60))
    ticktext = [fmt_hhmm(t) for t in ticks]

    hover_tmpl = "Hito: %{customdata[0]}<br>Hora: %{customdata[1]}<br>%{customdata[2]}<br>Ítem más común: %{customdata[3]}<br>Horas extra: %{customdata[4]}<extra></extra>"

    # [MOBILE] Cambia a horizontal si hay muchos usuarios para legibilidad
    many = len(order_axis) > 14
    if many:
        height = max(460, 22*len(order_axis) + 120)
        fig = px.bar(m, y="UsuarioTurnoShort", x="Seg", color="Hito", barmode="stack",
                     category_orders={"UsuarioTurnoShort": order_axis, "Hito": ["Inicio","Antes de alimentación","Antes de cierre"]},
                     color_discrete_map={"Inicio":"#1F77B4","Antes de alimentación":"#E4572E","Antes de cierre":"#2CA02C"},
                     custom_data=["Hito","Hora","Info","Item","Extra"], height=height, orientation="h")
        fig.update_traces(hovertemplate=hover_tmpl, marker_line_width=0, opacity=0.96, cliponaxis=False)
        fig.update_yaxes(categoryorder="array", categoryarray=order_axis, tickfont=dict(size=10))
        fig.update_xaxes(tickvals=ticks, ticktext=ticktext, title="Hora del día (HH:MM)")
    else:
        height = max(480, 26*len(order_axis) + 120)
        fig = px.bar(m, x="UsuarioTurnoShort", y="Seg", color="Hito", barmode="stack",
                     category_orders={"UsuarioTurnoShort": order_axis, "Hito": ["Inicio","Antes de alimentación","Antes de cierre"]},
                     color_discrete_map={"Inicio":"#1F77B4","Antes de alimentación":"#E4572E","Antes de cierre":"#2CA02C"},
                     custom_data=["Hito","Hora","Info","Item","Extra"], height=height)
        fig.update_traces(hovertemplate=hover_tmpl, marker_line_width=0, opacity=0.96, cliponaxis=False)
        fig.update_xaxes(categoryorder="array", categoryarray=order_axis,
                         tickangle=(-65 if len(order_axis) > 8 else -30), tickfont=dict(size=10))
        if len(order_axis) <= 12:
            fig.add_trace(go.Scatter(x=top_per_bar.index.tolist(), y=top_per_bar.values,
                                     mode="text", text=[fmt_hhmm(v) for v in top_per_bar.values],
                                     textposition="top center", textfont=dict(size=11, color=ANN_COL),
                                     showlegend=False, hoverinfo="skip"))
        fig.update_yaxes(tickvals=ticks, ticktext=ticktext, title="Hora del día (HH:MM)")

    _responsive_bar_style(fig, len(order_axis))
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=160), legend_title_text="Hito")
    apply_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# [S12] Render
# =========================================================
view_tm_por_usuario_turno()
st.divider()
view_ordenes_ot()
st.divider()
view_inicio_fin_turno()
# ======================= FIN ==============================
