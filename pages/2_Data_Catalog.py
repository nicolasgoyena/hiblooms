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

def drive_image(url: str):
    u = normalize_drive_url(url or "")
    if u:
        st.image(u, use_container_width=True)
    else:
        st.info("Sin imagen asociada.")

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

def get_table_columns(engine: Engine, table: str) -> List[Dict[str, Any]]:
    insp = inspect(engine)
    return insp.get_columns(table, schema="public")

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
    engine = get_engine()
except Exception as e:
    st.error(f"❌ Error obteniendo conexión: {e}")
    st.stop()

insp = inspect(engine)
all_tables = insp.get_table_names(schema="public")

# Detectar si estamos en modo detalle
params = st.query_params
if "page" in params and params.get("page") == ["lab_image"] and "id" in params:
    record_id = params.get("id")[0]
    table = "lab_images"
    cols = get_table_columns(engine, table)
    pk = infer_pk(engine, table) or cols[0]["name"]

    row = get_record_by_id(engine, table, pk, record_id)
    if row is None:
        st.error("❌ No se encontró el registro solicitado.")
        st.stop()

    st.subheader(f"🖼️ Detalle de imagen #{record_id}")

    img_url = normalize_drive_url(str(row.get("image_url", "")))
    if img_url:
        st.image(img_url, use_container_width=True)
    else:
        st.info("⚠️ Imagen no disponible.")

    st.markdown("### 📋 Información del registro")
    df_meta = pd.DataFrame(row).reset_index()
    df_meta.columns = ["Campo", "Valor"]
    st.dataframe(df_meta, hide_index=True, use_container_width=True)

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
                    st.experimental_set_query_params(page="lab_image", id=record_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error actualizando: {e}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Volver al catálogo"):
            st.experimental_set_query_params()
            st.rerun()
    with col2:
        if st.button("🗑️ Borrar registro"):
            try:
                delete_record(engine, table, pk, record_id)
                st.success("✅ Registro eliminado.")
                st.experimental_set_query_params()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error borrando: {e}")

    st.stop()

# ============ Vista general (catálogo normal) ============

if not all_tables:
    st.warning("No se han encontrado tablas en el esquema 'public'.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controles")
    table = st.selectbox("Tabla", all_tables, index=max(all_tables.index("lab_images") if "lab_images" in all_tables else 0, 0))
    st.markdown("---")
    page_size = st.select_slider("Registros por página", options=[20, 50, 100], value=20)
    page = st.number_input("Página", min_value=1, step=1, value=1)

cols = get_table_columns(engine, table)
pk = infer_pk(engine, table) or (cols[0]["name"] if cols else None)
order_col = choose_order_column(cols, pk)

where, params_sql = "", {}
offset = (page - 1) * page_size
total = count_records(engine, table, where, params_sql)
df = fetch_records(engine, table, where, params_sql, order_col, page_size, offset)

if table == "lab_images":
    st.markdown("### 🖼️ Galería de imágenes (clic para ver detalle)")
    st.markdown(
        """
        <div style='display:flex; flex-wrap:wrap; gap:20px; justify-content:center;'>
        """, unsafe_allow_html=True
    )

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

                if st.button(f"🖼️ ID {record_id}", key=f"open_{record_id}"):
                    st.experimental_set_query_params(page="lab_image", id=str(record_id))
                    st.rerun()

                st.markdown(
                    f"""
                    <div style="text-align:center; border:1px solid #ccc; border-radius:10px; padding:10px; background:#fff;">
                        {"<img src='" + proxy_url + "' style='max-width:100%; height:auto; border-radius:8px;'>" if proxy_url else "<p>⚠️ Sin imagen</p>"}
                        <p style="font-weight:600; margin-top:6px;">🧪 Extraction ID: <span style="color:#1e88e5;">{extraction_id}</span></p>
                    </div>
                    """, unsafe_allow_html=True
                )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- Tablas normales ---
if df.empty:
    st.info("No se han encontrado registros.")
else:
    st.dataframe(df, use_container_width=True)

st.markdown(f"Página **{page}** de **{max(1,(total+page_size-1)//page_size)}**")
