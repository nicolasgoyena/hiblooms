# encoding: utf-8
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 10:28:58 2025
@author: ngoyenaserv
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import sys, os, importlib.util, traceback

# ========================================
# Import robusto de db_utils desde raíz
# ========================================

# Calcular ruta absoluta del proyecto (un nivel arriba de /pages)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
db_utils_path = os.path.join(project_root, "db_utils.py")

if os.path.exists(db_utils_path):
    spec = importlib.util.spec_from_file_location("db_utils", db_utils_path)
    db_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db_utils)
    get_engine = db_utils.get_engine
    infer_pk = db_utils.infer_pk
else:
    st.error(f"❌ No se encontró db_utils.py en {project_root}")
    st.stop()



# =========================
# CACHÉ Y OPTIMIZACIÓN
# =========================

@st.cache_resource
def get_cached_engine() -> Engine:
    """Mantiene viva la conexión SQLAlchemy durante toda la sesión."""
    return get_engine()

@st.cache_data(ttl=600)
def get_cached_columns(_engine: Engine, table: str):
    """Devuelve columnas de una tabla (cacheadas 10 min)."""
    insp = inspect(_engine)
    return insp.get_columns(table, schema="public")

@st.cache_data(ttl=600)
def count_cached_records(_engine: Engine, table: str, where: str, params: Dict[str, Any]) -> int:
    sql = f'SELECT COUNT(*) FROM "{table}"{where}'
    with _engine.connect() as con:
        c = con.execute(text(sql), params).scalar()
    return int(c or 0)

@st.cache_data(ttl=60)
def fetch_cached_records(_engine: Engine, table: str, where: str, params: Dict[str, Any], order_col: str, limit: int, offset: int):
    sql = f'SELECT * FROM "{table}"{where} ORDER BY "{order_col}" DESC LIMIT :_lim OFFSET :_off'
    p = dict(params)
    p["_lim"] = limit
    p["_off"] = offset
    with _engine.connect() as con:
        df = pd.read_sql(text(sql), con, params=p)
    return df


# =========================
# Utilidades
# =========================

def normalize_drive_url(url: str) -> str:
    """Normaliza URLs de Drive a formato directo (uc?id=...)."""
    if not isinstance(url, str) or not url:
        return ""
    u = url.strip()
    if "drive.google.com/uc?id=" in u:
        return u
    if "drive.google.com/file/d/" in u:
        try:
            file_id = u.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?id={file_id}"
        except Exception:
            return u
    return u

def python_value_for_sql(val):
    if isinstance(val, (date, datetime)):
        return val
    if val == "":
        return None
    return val

def is_textual(coltype: str) -> bool:
    c = coltype.lower()
    return any(x in c for x in ["char", "text", "json", "uuid"])

def is_numeric(coltype: str) -> bool:
    c = coltype.lower()
    return any(x in c for x in ["int", "numeric", "float", "double", "real", "decimal"])

def is_temporal(coltype: str) -> bool:
    c = coltype.lower()
    return any(x in c for x in ["date", "time"])

def pick_display_fields(cols: List[Dict[str, Any]]) -> List[str]:
    names = [c["name"] for c in cols]
    priority = ["name","title","sample_id","reservoir","reservoir_name","point_name","type","category","date","created_at"]
    chosen = [c for c in priority if c in names]
    for n in names:
        if n not in chosen and len(chosen) < 5:
            chosen.append(n)
    return chosen[:5]

def choose_order_column(cols: List[Dict[str, Any]], pk: Optional[str]) -> str:
    if pk:
        return pk
    candidates = ["updated_at", "created_at", "timestamp", "ts", "date"]
    names = [c["name"] for c in cols]
    for c in candidates:
        if c in names:
            return c
    return names[0] if names else "1"

from streamlit_folium import st_folium
import folium

def get_extraction_point_coords(engine, extraction_id):
    """
    Dado un extraction_id (de lab_images), obtiene las coordenadas (lat, lon)
    del punto de extracción asociado, navegando:
    lab_images → samples → extraction_points
    """
    try:
        sql = text("""
            SELECT ep.latitude, ep.longitude
            FROM samples s
            JOIN extraction_points ep
                ON s.extraction_point_id = ep.extraction_point_id
            WHERE s.extraction_id = :eid
            LIMIT 1
        """)
        with engine.connect() as con:
            row = con.execute(sql, {"eid": extraction_id}).fetchone()
        if row and row[0] is not None and row[1] is not None:
            return float(row[0]), float(row[1])
    except Exception as e:
        st.error(f"❌ Error obteniendo coordenadas: {e}")
    return None



# =========================
# CRUD helpers
# =========================

def get_record_by_id(engine: Engine, table: str, pk: str, pk_value: Any) -> Optional[pd.Series]:
    sql = f'SELECT * FROM "{table}" WHERE "{pk}" = :id'
    with engine.connect() as con:
        df = pd.read_sql(text(sql), con, params={"id": pk_value})
    if df.empty:
        return None
    return df.iloc[0]

def insert_record(engine: Engine, table: str, data: Dict[str, Any]):
    cols = ", ".join(f'"{k}"' for k in data.keys())
    vals = ", ".join(f":{k}" for k in data.keys())
    sql = f'INSERT INTO "{table}" ({cols}) VALUES ({vals})'
    with engine.begin() as con:
        con.execute(text(sql), {k: python_value_for_sql(v) for k, v in data.items()})

def update_record(engine: Engine, table: str, pk: str, pk_value: Any, data: Dict[str, Any]):
    sets = ", ".join(f'"{k}" = :{k}' for k in data.keys())
    sql = f'UPDATE "{table}" SET {sets} WHERE "{pk}" = :_pkval'
    params = {k: python_value_for_sql(v) for k, v in data.items()}
    params["_pkval"] = pk_value
    with engine.begin() as con:
        con.execute(text(sql), params)

def delete_record(engine: Engine, table: str, pk: str, pk_value: Any):
    sql = f'DELETE FROM "{table}" WHERE "{pk}" = :_pkval'
    with engine.begin() as con:
        con.execute(text(sql), {"_pkval": pk_value})

def render_input_for_column(colmeta: Dict[str, Any], default=None):
    label = colmeta["name"]
    ctype = str(colmeta.get("type", ""))
    if is_temporal(ctype):
        if "time" in ctype:
            return st.datetime_input(label, value=default if isinstance(default, datetime) else None, format="DD-MM-YYYY HH:mm")
        else:
            return st.date_input(label, value=default if isinstance(default, date) else None, format="DD-MM-YYYY")
    elif is_numeric(ctype):
        return st.number_input(label, value=float(default) if default not in (None, "") else 0.0, step=1.0)
    elif "bool" in ctype.lower():
        return st.checkbox(label, value=bool(default) if default is not None else False)
    else:
        if "text" in ctype.lower() or "json" in ctype.lower():
            return st.text_area(label, value=str(default or ""))
        return st.text_input(label, value=str(default or ""))

# =========================
# UI principal
# =========================

st.set_page_config(page_title="Catálogo HIBLOOMS", layout="wide")
st.title("📖 Catálogo HIBLOOMS")

# Conexión
try:
    engine = get_cached_engine()
except Exception as e:
    st.error(f"❌ Error obteniendo conexión: {e}")
    st.stop()

insp = inspect(engine)
# Obtener todas las tablas del esquema público y filtrar las internas
all_tables = [t for t in insp.get_table_names(schema="public") if t.lower() != "spatial_ref_sys"]


# Detectar modo detalle
# Detectar modo detalle
params = st.query_params
if params.get("page") == "lab_image" and "id" in params:
    record_id = params.get("id")
    table = "lab_images"
    cols = get_cached_columns(engine, table)
    pk = infer_pk(engine, table) or cols[0]["name"]

    row = get_record_by_id(engine, table, pk, record_id)
    if row is None:
        st.error("❌ No se encontró el registro solicitado.")
        st.stop()

    st.subheader(f"🧫 Detalle de imagen #{record_id}")

    # =============================
    # Imagen principal + mapa satélite (centrados y compactos)
    # =============================
    
    # ---- Imagen ----
    st.markdown(
        """
        <h3 style='text-align:center; margin-bottom:12px;'>🧫 Imagen de laboratorio</h3>
        """,
        unsafe_allow_html=True
    )
    
    img_url = normalize_drive_url(str(row.get("image_url", "")))
    if img_url:
        proxy_url = f"https://images.weserv.nl/?url={img_url.replace('https://', '')}"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center;">
                <img src="{proxy_url}" alt="Imagen de laboratorio" style="
                    max-width: 55%;
                    max-height: 350px;
                    height: auto;
                    object-fit: contain;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                ">
            </div>
            <p style='text-align:center; color:#666;'>ID {record_id}</p>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("⚠️ Imagen no disponible.")
    
    
    # ---- Mapa debajo ----
    if "extraction_id" in row and pd.notna(row["extraction_id"]):
        coords = get_extraction_point_coords(engine, row["extraction_id"])
        if coords:
            lat, lon = coords
            # 🔹 Título alineado a la izquierda
            st.markdown(
                """
                <h3 style='text-align:left; margin-top:30px; margin-bottom:10px;'>
                    🗺️ Punto de extracción asociado
                </h3>
                """,
                unsafe_allow_html=True
            )
    
            # 🌍 Fondo satelital natural
            m = folium.Map(location=[lat, lon], zoom_start=15, tiles="Esri.WorldImagery")
    
            # 📍 Marcador rojo visible
            folium.Marker(
                [lat, lon],
                tooltip=f"Extraction ID: {row['extraction_id']}",
                icon=folium.Icon(color="red", icon="map-marker", prefix="fa")
            ).add_to(m)
    
            # ✅ Mapa centrado y con estilo
            st.markdown(
                """
                <style>
                .centered-map {
                    display: flex;
                    justify-content: center;
                    margin: 0 auto;
                    width: 720px;
                    max-width: 90%;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
            st.markdown("<div class='centered-map'>", unsafe_allow_html=True)
            st_folium(m, width=700, height=400)
            st.markdown("</div>", unsafe_allow_html=True)
    
        else:
            st.info("📍 No hay coordenadas disponibles para este extraction_id.")
    else:
        st.info("📍 Este registro no tiene extraction_id asociado.")


    


    # =============================
    # Información del registro
    # =============================
    st.markdown("### 📋 Información del registro")
    df_meta = pd.DataFrame(row).reset_index()
    df_meta.columns = ["Campo", "Valor"]
    st.dataframe(df_meta, hide_index=True, use_container_width=True)

    # =============================
    # Edición del registro
    # =============================
    st.markdown("---")
    edit_mode = st.toggle("✏️ Editar registro", value=False)
    if edit_mode:
        with st.form("form_edit_detail", clear_on_submit=False):
            new_values = {}
            for c in cols:
                cname = c["name"]
                if cname == pk:
                    st.text_input(cname, value=str(row.get(cname)), disabled=True)
                elif cname == "image_url":
                    st.text_input(cname, value="(no editable)", disabled=True)
                else:
                    new_values[cname] = render_input_for_column(c, default=row.get(cname))
            s = st.form_submit_button("Guardar cambios")
            if s:
                try:
                    update_record(engine, table, pk, record_id, new_values)
                    st.success("✅ Cambios guardados.")
                    st.query_params.update(page="lab_image", id=record_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error actualizando: {e}")

    # =============================
    # Botones de acción
    # =============================
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Volver al catálogo"):
            st.query_params.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Borrar registro"):
            try:
                delete_record(engine, table, pk, record_id)
                st.success("✅ Registro eliminado.")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error borrando: {e}")

    st.stop()


with st.sidebar:
    st.header("⚙️ Controles")

    # Diccionario de nombres amigables
    TABLE_LABELS = {
        "reservoirs_spain": "🏞️ Embalses de España",
        "extraction_points": "📍 Puntos de extracción",
        "lab_images": "🧫 Imágenes de laboratorio",
        "insitu_sampling": "🧪 Muestreos in situ",
        "profiles_data": "🌡️ Perfiles de datos",
        "sediment_data": "🪨 Datos de sedimentos",
        "insitu_determinations": "🔬 Determinaciones in situ",
        "rivers_spain": "🌊 Ríos de España",
        "sensor_data": "📈 Datos de sensores",
        "samples": "🧫 Muestras de laboratorio",
    }

    # Ocultar tablas del sistema
    exclude_tables = ["spatial_ref_sys"]
    all_tables = [t for t in insp.get_table_names(schema="public") if t.lower() not in exclude_tables]

    # Crear lista traducida
    table_options = [TABLE_LABELS.get(t, t) for t in all_tables]
    selected_label = st.selectbox("Selecciona una tabla", table_options)

    # Convertir de la etiqueta visible al nombre real de la tabla
    table = next(k for k, v in TABLE_LABELS.items() if v == selected_label)

    st.markdown("---")
    page_size = st.select_slider("Registros por página", options=[20, 50, 100], value=20)
    page = st.session_state.get("page", 1)


# Cachear columnas y metadatos
if "cols_cache" not in st.session_state:
    st.session_state["cols_cache"] = {}

if table not in st.session_state["cols_cache"]:
    st.session_state["cols_cache"][table] = get_cached_columns(engine, table)

cols = st.session_state["cols_cache"][table]
pk = infer_pk(engine, table) or (cols[0]["name"] if cols else None)
order_col = choose_order_column(cols, pk)

where, params_sql = "", {}
offset = (page - 1) * page_size
total = count_cached_records(engine, table, where, params_sql)
df = fetch_cached_records(engine, table, where, params_sql, order_col, page_size, offset)

# ===== Vista especial lab_images =====
if table == "lab_images":
    st.markdown("### 🖼️ Galería de imágenes (haz clic para ver detalle)")
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:20px; justify-content:center;'>", unsafe_allow_html=True)

    n_cols = 4
    rows_chunks = [df.iloc[i:i+n_cols] for i in range(0, len(df), n_cols)]

    for chunk in rows_chunks:
        cols_ui = st.columns(n_cols, gap="large")
        for (ridx, rrow), col_ui in zip(chunk.iterrows(), cols_ui):
            with col_ui:
                img_url = normalize_drive_url(str(rrow.get("image_url", "")))
                proxy_url = f"https://images.weserv.nl/?url={img_url.replace('https://', '')}" if img_url else ""
                extraction_id = rrow.get("extraction_id", "(sin extraction_id)")
                record_id = rrow.get(pk)

                # ✅ Botón completo que redirige en la misma pestaña
                if st.button(
                    f"🧪 Extraction ID: {extraction_id}\n(ID {record_id})",
                    key=f"btn_{record_id}",
                    use_container_width=True
                ):
                    st.query_params.clear()
                    st.query_params.update(page="lab_image", id=record_id)
                    st.rerun()

                # Imagen de previsualización
                st.markdown(
                    f"""
                    <div style="text-align:center; border:1px solid #ddd; border-radius:10px;
                                padding:10px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                        {"<img src='" + proxy_url + "' style='max-width:100%; height:180px; border-radius:8px; object-fit:cover;'>" if proxy_url else "<p>⚠️ Sin imagen</p>"}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ===== Tablas normales =====
if df.empty:
    st.info("No se han encontrado registros.")
else:
    # Calcular índice global (no reiniciado por página)
    df.index = df.index + 1 + offset
    st.dataframe(df, use_container_width=True)

# =====================
# Paginación final (centrada, funcional y con texto "Ir a página:")
# =====================

total_pages = max(1, (total + page_size - 1) // page_size)
start_rec = offset + 1 if total > 0 else 0
end_rec = min(offset + page_size, total)

col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    if page > 1:
        if st.button("⬅️ Anterior"):
            st.session_state["page"] = page - 1
            st.rerun()

with col2:
    # Cabecera centrada
    st.markdown(
        f"""
        <div style='text-align:center; font-size:15px;'>
            Página <b>{page}</b> de <b>{total_pages}</b> · 
            Registros {start_rec}–{end_rec} de {total}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Contenedor flex centrado con texto "Ir a página"
    st.markdown(
        """
        <div style='height:6px;'></div>
        <div style='display:flex; justify-content:center; align-items:center; gap:8px;'>
            <span style='font-size:14px; color:#555;'>Ir a página:</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Campo numérico centrado justo debajo
    center_col = st.columns([4, 1, 4])[1]
    with center_col:
        new_page = st.number_input(
            "",
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            label_visibility="collapsed",
            key="go_to_page",
            format="%d"
        )
        if new_page != page:
            st.session_state["page"] = new_page
            st.rerun()

        # Reducir ancho visual del input
        st.markdown(
            """
            <style>
            div[data-baseweb="input"] > div {
                width: 70px !important;
                text-align: center !important;
                margin: 0 auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

with col3:
    if page < total_pages:
        if st.button("Siguiente ➡️"):
            st.session_state["page"] = page + 1
            st.rerun()






