# encoding: utf-8
import streamlit as st
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from db_utils import get_engine, infer_pk

# =========================
# Utilidades
# =========================

def normalize_drive_url(url: str) -> str:
    """Normaliza URLs de Drive a formato directo (uc?id=...). Si ya lo es, lo deja igual."""
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

def drive_image(url: str):
    u = normalize_drive_url(url or "")
    if u:
        st.image(u, use_container_width=True, caption="image_url")
    else:
        st.info("Sin imagen asociada.")

def python_value_for_sql(val):
    """Convierte widgets Streamlit a valores aptos para SQL (manejo de fechas)."""
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
    """Elige 3-5 campos agradables para mostrar en la tarjeta."""
    names = [c["name"] for c in cols]
    priority = [
        "name","title","sample_id","reservoir","reservoir_name",
        "point_name","type","kind","category","date","created_at","updated_at"
    ]
    chosen = [c for c in priority if c in names]
    for n in names:
        if n not in chosen and len(chosen) < 5:
            chosen.append(n)
    return chosen[:5]

def choose_order_column(cols: List[Dict[str, Any]], pk: Optional[str]) -> str:
    """Elige la mejor columna para ordenar DESC (pk -> fecha -> primera)."""
    if pk:
        return pk
    # Preferir timestamps si existen
    candidates = ["updated_at", "created_at", "timestamp", "ts", "date", "fecha"]
    names = [c["name"] for c in cols]
    for c in candidates:
        if c in names:
            return c
    return names[0] if names else "1"

# =========================
# Filtros avanzados
# =========================

def filter_widgets(cols: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Genera widgets de filtro por cada columna.
    Devuelve un dict: {col: {"mode": "text|num|date|bool|raw", "value": ...}}
    - text  -> text_input (ILIKE %valor%)
    - num   -> (min,max)
    - date  -> (start,end)
    - bool  -> True/False/None (None = sin filtrar)
    - raw   -> text_input exacto (para tipos no detectados)
    """
    st.subheader("🔎 Filtros avanzados")
    filters: Dict[str, Dict[str, Any]] = {}

    with st.expander("Mostrar / ocultar filtros", expanded=False):
        for c in cols:
            cname = c["name"]
            ctype = str(c.get("type", ""))

            if is_temporal(ctype):
                col1, col2 = st.columns(2)
                with col1:
                    start = st.date_input(f"{cname} desde", value=None, format="DD-MM-YYYY", key=f"f_{cname}_start")
                with col2:
                    end = st.date_input(f"{cname} hasta", value=None, format="DD-MM-YYYY", key=f"f_{cname}_end")
                filters[cname] = {"mode": "date", "value": (start, end)}
            elif is_numeric(ctype):
                col1, col2 = st.columns(2)
                with col1:
                    nmin = st.text_input(f"{cname} min", value="", key=f"f_{cname}_min")
                with col2:
                    nmax = st.text_input(f"{cname} max", value="", key=f"f_{cname}_max")
                filters[cname] = {"mode": "num", "value": (nmin, nmax)}
            elif "bool" in ctype.lower():
                opt = st.selectbox(
                    f"{cname} (booleano)",
                    options=["(sin filtro)", "True", "False"],
                    index=0, key=f"f_{cname}_bool"
                )
                val = None if opt == "(sin filtro)" else (opt == "True")
                filters[cname] = {"mode": "bool", "value": val}
            elif is_textual(ctype):
                t = st.text_input(f"{cname} contiene...", value="", key=f"f_{cname}_text", placeholder="búsqueda parcial")
                filters[cname] = {"mode": "text", "value": t}
            else:
                t = st.text_input(f"{cname} (= exacto)", value="", key=f"f_{cname}_raw", placeholder="coincidencia exacta")
                filters[cname] = {"mode": "raw", "value": t}

        st.caption("💡 Deja campos vacíos para no filtrar por esa columna.")

    return filters

def build_where_and_params(filters: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Construye WHERE dinámico y params para una consulta segura.
    Devuelve (where_sql, params_dict)
    """
    conds = []
    params: Dict[str, Any] = {}
    idx = 0

    for cname, meta in filters.items():
        mode = meta["mode"]
        val = meta["value"]

        if mode == "text":
            if val:
                idx += 1
                conds.append(f'"{cname}"::text ILIKE :p{idx}')
                params[f"p{idx}"] = f"%{val}%"
        elif mode == "num":
            nmin, nmax = val
            if nmin not in (None, ""):
                try:
                    float(nmin)
                    idx += 1
                    conds.append(f'"{cname}" >= :p{idx}')
                    params[f"p{idx}"] = float(nmin)
                except Exception:
                    pass
            if nmax not in (None, ""):
                try:
                    float(nmax)
                    idx += 1
                    conds.append(f'"{cname}" <= :p{idx}')
                    params[f"p{idx}"] = float(nmax)
                except Exception:
                    pass
        elif mode == "date":
            start, end = val
            if start:
                idx += 1
                conds.append(f'"{cname}" >= :p{idx}')
                params[f"p{idx}"] = datetime.combine(start, datetime.min.time())
            if end:
                idx += 1
                conds.append(f'"{cname}" <= :p{idx}')
                params[f"p{idx}"] = datetime.combine(end, datetime.max.time())
        elif mode == "bool":
            if val is not None:
                idx += 1
                conds.append(f'"{cname}" = :p{idx}')
                params[f"p{idx}"] = bool(val)
        elif mode == "raw":
            if val:
                idx += 1
                conds.append(f'"{cname}" = :p{idx}')
                params[f"p{idx}"] = val

    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    return where, params

# =========================
# CRUD helpers
# =========================

def fetch_records(engine: Engine, table: str, where: str, params: Dict[str, Any], order_col: str, limit: int, offset: int):
    sql = f'SELECT * FROM "{table}"{where} ORDER BY "{order_col}" DESC LIMIT :_lim OFFSET :_off'
    p = dict(params)
    p["_lim"] = limit
    p["_off"] = offset
    with engine.connect() as con:
        df = pd.read_sql(text(sql), con, params=p)
    return df

def count_records(engine: Engine, table: str, where: str, params: Dict[str, Any]) -> int:
    sql = f'SELECT COUNT(*) FROM "{table}"{where}'
    with engine.connect() as con:
        c = con.execute(text(sql), params).scalar()
    return int(c or 0)

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

def get_table_columns(engine: Engine, table: str) -> List[Dict[str, Any]]:
    insp = inspect(engine)
    return insp.get_columns(table, schema="public")

def render_input_for_column(colmeta: Dict[str, Any], default=None):
    label = colmeta["name"]
    ctype = str(colmeta.get("type", ""))
    if is_temporal(ctype):
        if "time" in ctype:
            return st.datetime_input(label, value=default if isinstance(default, datetime) else None, format="DD-MM-YYYY HH:mm", key=f"dt_{label}")
        else:
            return st.date_input(label, value=default if isinstance(default, date) else None, format="DD-MM-YYYY", key=f"d_{label}")
    elif is_numeric(ctype):
        return st.number_input(label, value=float(default) if default not in (None, "") else 0.0, step=1.0, key=f"n_{label}")
    elif "bool" in ctype.lower():
        return st.checkbox(label, value=bool(default) if default is not None else False, key=f"b_{label}")
    else:
        if "text" in ctype.lower() or "json" in ctype.lower():
            return st.text_area(label, value=str(default or ""), key=f"ta_{label}")
        return st.text_input(label, value=str(default or ""), key=f"t_{label}")

# =========================
# UI principal
# =========================

st.set_page_config(page_title="Catálogo HIBLOOMS", layout="wide")
st.title("📖 Catálogo HIBLOOMS")
st.caption("Explora, filtra, crea y edita registros de todas las tablas (carga inicial ligera).")

# Conexión
try:
    engine = get_engine()
except Exception as e:
    st.error(f"❌ Error obteniendo conexión: {e}")
    st.stop()

insp = inspect(engine)
all_tables = insp.get_table_names(schema="public")
if not all_tables:
    st.warning("No se han encontrado tablas en el esquema 'public'.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controles")
    table = st.selectbox("Tabla", all_tables, index=max(all_tables.index("lab_images") if "lab_images" in all_tables else 0, 0))
    st.markdown("---")
    page_size = st.select_slider("Registros por página", options=[20, 50, 100], value=20)
    page = st.number_input("Página", min_value=1, step=1, value=1)
    st.markdown("---")
    st.subheader("➕ Añadir nuevo")
    add_open = st.checkbox("Abrir formulario de creación", value=False)

# Metadatos
cols = get_table_columns(engine, table)
if not cols:
    st.warning("No se pudieron leer las columnas de la tabla.")
    st.stop()

pk = infer_pk(engine, table) or (cols[0]["name"] if cols else None)
order_col = choose_order_column(cols, pk)

# Filtros
filters = filter_widgets(cols)
where, params = build_where_and_params(filters)

# Conteo y datos
offset = (page - 1) * page_size
total = count_records(engine, table, where, params)
df = fetch_records(engine, table, where, params, order_col=order_col, limit=page_size, offset=offset)

# Info
if where:
    st.info(f"Resultados filtrados: **{total}** · Orden: **{order_col} DESC** · Página **{page}**")
else:
    st.info(f"Mostrando las **{min(total, page_size)}** más recientes (usa filtros para afinar). • Total tabla: **{total}** · Orden: **{order_col} DESC** · Página **{page}**")

# ---------------------------
# Crear registro
# ---------------------------
if add_open:
    with st.expander(f"➕ Crear registro en **{table}**", expanded=True):
        with st.form(f"form_create_{table}", clear_on_submit=False):
            create_values = {}
            for c in cols:
                cname = c["name"]
                # Si la PK parece autoincremental, no pedirla
                if cname == pk and ("identity" in str(c.get("type","")).lower() or "serial" in str(c.get("type","")).lower()):
                    st.text_input(cname, value="(autogenerado)", disabled=True)
                    continue
                create_values[cname] = render_input_for_column(c, default=None)

            submitted = st.form_submit_button("Crear registro")
            if submitted:
                try:
                    data_clean = {k: python_value_for_sql(v) for k, v in create_values.items()}
                    # Evitar insertar PK si es autogenerada
                    if pk in data_clean and ("identity" in str({cc['name']: cc for cc in cols}[pk].get("type","")).lower() or "serial" in str({cc['name']: cc for cc in cols}[pk].get("type","")).lower()):
                        data_clean.pop(pk, None)
                    insert_record(engine, table, data_clean)
                    st.success("✅ Registro creado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creando registro: {e}")

# ---------------------------
# Vista tipo tarjetas compactas (una casilla por registro)
# ---------------------------

st.subheader(f"📋 Registros en {table}")

display_fields = pick_display_fields(cols)
basic_fields = display_fields[:5]  # hasta 5 campos clave por registro

if df.empty:
    st.info("No se han encontrado registros con los criterios actuales.")
else:
    for idx, row in df.iterrows():
        with st.container(border=True):
            # Caso especial: tabla de imágenes
            if table == "lab_images" and "image_url" in row:
                raw_url = str(row["image_url"]) if pd.notna(row["image_url"]) else ""
                clean_url = normalize_drive_url(raw_url)
            
                if clean_url:
                    # Proxy para evitar bloqueos de Google Drive
                    proxy_url = f"https://images.weserv.nl/?url={clean_url.replace('https://', '')}"
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <img src="{proxy_url}" alt="imagen" style="
                                width:auto;
                                max-width:95%;
                                height:220px;
                                object-fit:contain;
                                border-radius:10px;
                                border:1px solid #ccc;
                                background-color:#fafafa;
                            ">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("⚠️ Imagen no disponible.")
            
                extraction_id = row.get("extraction_id", "(sin extraction_id)")
                st.markdown(f"🧪 **Extraction ID:** `{extraction_id}`")
            

            else:
                # Título principal con el ID
                title = f"**#{row[pk]}** — `{table}`" if pk in row else f"`{table}`"
                st.markdown(title)

                # Campos principales en una línea
                col_data = []
                for f in basic_fields:
                    if f in row and pd.notna(row[f]) and f != "image_url":
                        val = row[f]
                        if isinstance(val, float):
                            val = round(val, 4)
                        elif isinstance(val, (datetime, date)):
                            val = val.strftime("%Y-%m-%d")
                        col_data.append(f"**{f}**: {val}")
                if col_data:
                    st.markdown(" · ".join(col_data))

            # Botones de acción alineados en la parte inferior
            b1, b2, b3 = st.columns([1, 1, 1])
            show = b1.button("🔎 Ver", key=f"view_{table}_{idx}")
            edit = b2.button("✏️ Editar", key=f"edit_{table}_{idx}")
            delete = b3.button("🗑️ Borrar", key=f"del_{table}_{idx}") if pk else None

            # --- Ver detalles ---
            if show:
                with st.expander(f"Detalles del registro #{row[pk] if pk else idx}", expanded=True):
                    for c in cols:
                        cname = c["name"]
                        val = row.get(cname)
                        # No mostrar la URL explícita
                        if cname == "image_url":
                            continue
                        if cname == "image_url" and pd.notna(val):
                            drive_image(str(val))
                        else:
                            st.write(f"**{cname}**: {val}")

            # --- Editar ---
            if edit:
                with st.expander(f"Editar registro #{row[pk] if pk else idx}", expanded=True):
                    with st.form(f"form_edit_{table}_{idx}", clear_on_submit=False):
                        new_values = {}
                        for c in cols:
                            cname = c["name"]
                            if cname == pk:
                                st.text_input(cname, value=str(row.get(cname)), disabled=True)
                            elif cname == "image_url":
                                # No editar la URL desde la interfaz
                                st.text_input(cname, value="(no editable)", disabled=True)
                            else:
                                new_values[cname] = render_input_for_column(c, default=row.get(cname))
                        s = st.form_submit_button("Guardar cambios")
                        if s:
                            try:
                                if not pk:
                                    st.warning("No se puede actualizar sin clave primaria.")
                                else:
                                    update_record(engine, table, pk, row[pk], new_values)
                                    st.success("✅ Cambios guardados.")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error actualizando: {e}")

            # --- Borrar ---
            if delete:
                if not pk:
                    st.warning("Esta tabla no tiene PK inferida; no se puede borrar de forma segura.")
                else:
                    if st.checkbox(f"Confirmar borrado #{row[pk]}", key=f"chk_{table}_{idx}"):
                        try:
                            delete_record(engine, table, pk, row[pk])
                            st.success("✅ Registro eliminado.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error borrando: {e}")

# Paginación
total_pages = max(1, (total + page_size - 1) // page_size)
st.markdown(f"Página **{page}** de **{total_pages}**")

