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

from streamlit_folium import folium_static
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


# =========================
# SUBPÁGINAS DE DETALLE (todas las tablas)
# =========================
params = st.query_params

# ====== Detalle de un registro (genérico o lab_images) ======
if "page" in params and params.get("page") in ["lab_image", "detail"] and "id" in params:
    table = params.get("table", "lab_images") if params.get("page") == "detail" else "lab_images"
    record_id = params.get("id")

    cols = get_cached_columns(engine, table)
    pk = infer_pk(engine, table) or cols[0]["name"]

    row = get_record_by_id(engine, table, pk, record_id)
    if row is None:
        st.error("❌ No se encontró el registro solicitado.")
        st.stop()

    # Encabezado del detalle
    st.subheader(f"📄 Detalle del registro (tabla: {table}, ID: {record_id})")

    # =============================
    # CASO ESPECIAL: lab_images → imagen + mapa
    # =============================
    if table == "lab_images":
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

        # Mapa si existe extraction_id
        if "extraction_id" in row and pd.notna(row["extraction_id"]):
            coords = get_extraction_point_coords(engine, row["extraction_id"])
            if coords:
                lat, lon = coords
                st.markdown("<h3 style='text-align:left;'>🗺️ Punto de extracción asociado</h3>", unsafe_allow_html=True)
                m = folium.Map(location=[lat, lon], zoom_start=15, tiles="Esri.WorldImagery")
                folium.Marker([lat, lon], tooltip="Punto de extracción", icon=folium.Icon(color="red")).add_to(m)
                folium_static(m, width=700, height=400)

    # =============================
    # Información general
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
        with st.form("form_edit_generic", clear_on_submit=False):
            new_values = {}
            for c in cols:
                cname = c["name"]
                if cname == pk:
                    st.text_input(cname, value=str(row.get(cname)), disabled=True)
                else:
                    new_values[cname] = render_input_for_column(c, default=row.get(cname))
            if st.form_submit_button("Guardar cambios"):
                update_record(engine, table, pk, record_id, new_values)
                st.success("✅ Cambios guardados.")
                st.query_params.update(page="detail", table=table, id=record_id)
                st.rerun()

    # =============================
    # Botones inferiores
    # =============================
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Volver al catálogo"):
            current_table = st.query_params.get("table", "lab_images")
            current_page = st.session_state.get("page", 1)
            st.query_params.clear()
            st.query_params.update(table=current_table)
            st.session_state["page"] = current_page
            st.rerun()


    with col2:
        if st.button("🗑️ Borrar registro"):
            delete_record(engine, table, pk, record_id)
            st.success("✅ Registro eliminado.")
            st.query_params.clear()
            st.rerun()

    st.stop()

# ====== Detalle de un grupo de muestras ======
if params.get("page") == "detail" and "group" in params and "time" in params:
    table = params.get("table")
    point_id = params.get("group")
    time_group = params.get("time")

    # Determinar columna temporal
    table_cols = [c["name"] for c in get_cached_columns(engine, table)]
    time_col = next((c for c in ["date", "datetime", "created_at", "timestamp"] if c in table_cols), None)
    if not time_col:
        st.error("❌ No se encontró una columna temporal adecuada en la tabla.")
        st.stop()

    # Recuperar registros del grupo (±30 min)
    time_start = pd.to_datetime(time_group)
    time_end = time_start + pd.Timedelta(minutes=30)
    sql = text(f"""
        SELECT * FROM "{table}"
        WHERE extraction_point_id = :pid
          AND {time_col} BETWEEN :tstart AND :tend
        ORDER BY {time_col} ASC
    """)
    with engine.connect() as con:
        df_group = pd.read_sql(sql, con, params={"pid": point_id, "tstart": time_start, "tend": time_end})

    st.subheader(f"📊 Detalle de grupo — Punto {point_id}, {time_start.strftime('%Y-%m-%d %H:%M')}")
    st.dataframe(df_group, use_container_width=True, hide_index=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Volver al catálogo"):
            current_table = st.query_params.get("table", table)
            current_page = st.session_state.get("page", 1)
            st.query_params.clear()
            st.query_params.update(table=current_table)
            st.session_state["page"] = current_page
            st.rerun()
    
    with col2:
        if st.button("🗑️ Borrar grupo"):
            st.warning("⚠️ Eliminación de grupos completa aún no implementada.")




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
    st.markdown("### 🖼️ Galería de imágenes (clic para ver detalle)")
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

                # Enlace directo clicable en toda la tarjeta
                detail_url = f"?page=lab_image&id={record_id}"

                st.markdown(
                    f"""
                    <a href="{detail_url}" style="text-decoration:none; color:inherit;">
                        <div style="
                            text-align:center;
                            border:1px solid #ccc;
                            border-radius:10px;
                            padding:10px;
                            background:#fff;
                            transition:all 0.2s ease-in-out;
                            box-shadow:0 2px 6px rgba(0,0,0,0.08);
                        " 
                        onmouseover="this.style.boxShadow='0 4px 10px rgba(0,0,0,0.15)'; this.style.transform='scale(1.02)';"
                        onmouseout="this.style.boxShadow='0 2px 6px rgba(0,0,0,0.08)'; this.style.transform='scale(1)';">
                            {"<img src='" + proxy_url + "' style='max-width:100%; height:auto; border-radius:8px;'>" if proxy_url else "<p>⚠️ Sin imagen</p>"}
                            <p style="font-weight:600; margin-top:6px;">🧪 Extraction ID: <span style="color:#1e88e5;">{extraction_id}</span></p>
                            <p style="color:#666;">ID {record_id}</p>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ===== TABLAS NORMALES O AGRUPADAS =====

# Calcular índice global (no reiniciado por página)
df.index = df.index + 1 + offset

# Tablas que deben mostrarse agrupadas
grouped_tables = ["samples", "profiles_data", "insitu_determinations", "insitu_sampling"]

if table in grouped_tables:
    st.markdown(f"### 🧩 Registros agrupados de `{table}` por punto y hora")
    
    if df.empty:
        st.info("No se han encontrado registros.")
    else:
        # Determinar columna temporal
        time_col = next(
            (c for c in ["date", "datetime", "created_at", "timestamp"]
             if c in [col["name"] for col in get_cached_columns(engine, table)]),
            None
        )
        if not time_col:
            st.warning("No se encontró columna temporal para agrupar.")
        else:
            # Agrupar por punto y hora redondeada
            df["hour_group"] = pd.to_datetime(df[time_col]).dt.floor("H")
            grouped = df.groupby(["extraction_point_id", "hour_group"])

            start_idx = offset
            end_idx = offset + page_size
            visible_groups = list(grouped)[start_idx:end_idx]

            for (point_id, hour), group in visible_groups:
                with st.container(border=True):
                    st.markdown(
                        f"#### 📍 Punto {point_id} — {hour.strftime('%Y-%m-%d %H:%M')}"
                    )

                    # Mostrar las 5 primeras filas del grupo
                    st.dataframe(group.head(5), hide_index=True, use_container_width=True)

                    detail_url = f"?page=detail&table={table}&group={point_id}&time={hour.isoformat()}"
                    st.markdown(
                        f"""
                        <a href="{detail_url}" target="_self" style="text-decoration:none;">
                            <button style="
                                background-color:#1e88e5;
                                color:white;
                                border:none;
                                padding:6px 12px;
                                border-radius:6px;
                                cursor:pointer;
                                margin-top:8px;">
                                🔎 Ver detalles
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

            # Calcular número de páginas reales (por grupos, no por filas)
            total_groups = len(grouped)
            total_pages = max(1, (total_groups + page_size - 1) // page_size)
            st.caption(f"Mostrando {len(visible_groups)} grupos (Página {page} de {total_pages})")

else:
    # ============================
    # 🔹 Caso especial: embalses → mapa de polígonos
    # ============================
    # ============================
    # ============================
    # 🔹 Caso especial: embalses → mapa de polígonos estilo shapefile
    # ============================
    if table == "reservoirs_spain":
        st.markdown("### 🗺️ Mapa interactivo de embalses de España")
    
        if df.empty:
            st.info("No hay registros de embalses para mostrar.")
        else:
            from shapely import wkb
            import geopandas as gpd
    
            # --- Conversión robusta de geometrías WKB hex ---
            def safe_load_wkb(geom):
                try:
                    if isinstance(geom, str):
                        return wkb.loads(geom, hex=True)
                    elif isinstance(geom, (bytes, bytearray)):
                        return wkb.loads(geom)
                    elif isinstance(geom, memoryview):
                        return wkb.loads(bytes(geom))
                except Exception:
                    return None
                return None
    
            df = df.copy()
            df["geometry"] = df["geometry"].apply(safe_load_wkb)
            df = df[df["geometry"].notnull() & df["geometry"].apply(lambda g: not g.is_empty)]
    
            if df.empty:
                st.warning("⚠️ Ninguna geometría válida encontrada en la columna 'geometry'.")
                st.stop()
    
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    
            # --- Calcular extensión y crear mapa ---
            bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
            m = folium.Map(location=center, zoom_start=6, tiles="Esri.WorldImagery")
    
            # --- Dibujar embalses uno a uno ---
            nombre_columna = next((c for c in gdf.columns if "name" in c.lower() or "nombre" in c.lower()), None)
            if not nombre_columna:
                nombre_columna = gdf.columns[0]  # usar la primera columna disponible como fallback
    
            for _, row in gdf.iterrows():
                nombre = str(row.get(nombre_columna, "Embalse desconocido"))
    
                geom = row.geometry
                if geom.is_empty:
                    continue
    
                if geom.geom_type == "Point":
                    folium.Marker(
                        location=[geom.y, geom.x],
                        popup=nombre,
                        tooltip=nombre,
                        icon=folium.Icon(color="blue", icon="tint"),
                    ).add_to(m)
    
                elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                    folium.GeoJson(
                        geom.__geo_interface__,
                        name=nombre,
                        tooltip=folium.Tooltip(nombre),
                        style_function=lambda x: {
                            "fillColor": "blue",
                            "color": "blue",
                            "weight": 2,
                            "fillOpacity": 0.4,
                        },
                    ).add_to(m)
    
            # --- Ajustar zoom a la extensión ---
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
            # --- Mostrar mapa centrado en Streamlit ---
            st.markdown(
                """
                <style>
                .centered-map {
                    display: flex;
                    justify-content: center;
                    margin: 0 auto;
                    width: 95%;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
            st.markdown("<div class='centered-map'>", unsafe_allow_html=True)
            folium_static(m, width=1000, height=650)
            st.markdown("</div>", unsafe_allow_html=True)
    


    # ============================
    # 🔹 Tablas normales (por defecto)
    # ============================
    else:
        st.markdown(f"### 📘 Registros de `{table}` (paginación estándar)")
        if df.empty:
            st.info("No se han encontrado registros.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)


# =====================
# Paginación final (dinámica según tipo de tabla)
# =====================

grouped_tables = ["samples", "profiles_data", "insitu_determinations", "insitu_sampling"]

if table in grouped_tables:
    # Calcular total de grupos únicos (no filas)
    time_col = next(
        (c for c in ["date", "datetime", "created_at", "timestamp"]
         if c in [col["name"] for col in get_cached_columns(engine, table)]),
        None
    )
    if time_col and not df.empty:
        df["hour_group"] = pd.to_datetime(df[time_col]).dt.floor("H")
        total_groups = len(df.groupby(["extraction_point_id", "hour_group"]))
    else:
        total_groups = 0

    total_pages = max(1, (total_groups + page_size - 1) // page_size)
    start_rec = (page - 1) * page_size + 1 if total_groups > 0 else 0
    end_rec = min(page * page_size, total_groups)

else:
    # Modo normal: paginación por filas
    total_pages = max(1, (total + page_size - 1) // page_size)
    start_rec = offset + 1 if total > 0 else 0
    end_rec = min(offset + page_size, total)

# =====================
# Controles de navegación (comunes)
# =====================

col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    if page > 1:
        if st.button("⬅️ Anterior"):
            st.session_state["page"] = page - 1
            st.rerun()

with col2:
    st.markdown(
        f"""
        <div style='text-align:center; font-size:15px;'>
            Página <b>{page}</b> de <b>{total_pages}</b> · 
            Registros {start_rec}–{end_rec}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='height:6px;'></div>
        <div style='display:flex; justify-content:center; align-items:center; gap:8px;'>
            <span style='font-size:14px; color:#555;'>Ir a página:</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    center_col = st.columns([4, 1, 4])[1]
    with center_col:
        new_page = st.number_input(
            "",
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            label_visibility="collapsed",
            key=f"go_to_page_{table}",
            format="%d"
        )
        if new_page != page:
            st.session_state["page"] = new_page
            st.rerun()

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
