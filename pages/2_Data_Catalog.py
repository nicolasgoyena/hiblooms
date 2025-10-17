# encoding: utf-8
import streamlit as st
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from db_utils import get_engine, infer_pk

# ---------------------------
# Helpers
# ---------------------------

def normalize_drive_url(url: str) -> str:
    """Normaliza URLs de Drive a formato directo (uc?id=...). Si ya lo es, lo deja igual."""
    if not isinstance(url, str) or not url:
        return ""
    url = url.strip()
    if "drive.google.com/uc?id=" in url:
        return url  # ya OK
    # formatos comunes: https://drive.google.com/file/d/ID/view?usp=sharing
    if "drive.google.com/file/d/" in url:
        try:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?id={file_id}"
        except Exception:
            return url
    return url

def pick_display_fields(cols: List[Dict[str, Any]]) -> List[str]:
    """
    Elige 3-5 campos 'bonitos' para mostrar rápido en la tarjeta.
    Si hay 'name', 'title', 'date', etc., los prioriza.
    """
    names = [c["name"] for c in cols]
    priority = ["name", "title", "sample_id", "reservoir", "reservoir_name",
                "point_name", "type", "kind", "category", "date", "created_at", "updated_at"]
    chosen = [c for c in priority if c in names]
    for n in names:
        if n not in chosen and len(chosen) < 5:
            chosen.append(n)
    return chosen[:5]

def python_value_for_sql(val):
    """Convierte widgets Streamlit a valores para SQL (especial fechas)."""
    if isinstance(val, (date, datetime)):
        return val
    if val == "":
        return None
    return val

def is_textual(coltype: str) -> bool:
    coltype = coltype.lower()
    return any(x in coltype for x in ["char", "text", "json"])

def is_numeric(coltype: str) -> bool:
    coltype = coltype.lower()
    return any(x in coltype for x in ["int", "numeric", "float", "double", "real", "decimal"])

def is_temporal(coltype: str) -> bool:
    coltype = coltype.lower()
    return any(x in coltype for x in ["date", "time"])

def render_input_for_column(colmeta: Dict[str, Any], default=None):
    """Devuelve valor capturado para una columna según su tipo."""
    label = colmeta["name"]
    ctype = str(colmeta.get("type", ""))
    nullable = colmeta.get("nullable", True)

    if is_temporal(ctype):
        # date o timestamp
        if "time" in ctype:
            # timestamp → datetime
            return st.datetime_input(label, value=default if isinstance(default, datetime) else None, format="DD-MM-YYYY HH:mm", key=f"dt_{label}")
        else:
            # date
            return st.date_input(label, value=default if isinstance(default, date) else None, format="DD-MM-YYYY", key=f"d_{label}")
    elif is_numeric(ctype):
        # numérico → number_input con paso float
        if default is None:
            default = 0
        return st.number_input(label, value=float(default) if default is not None else 0.0, step=1.0, key=f"n_{label}")
    elif "bool" in ctype.lower():
        return st.checkbox(label, value=bool(default) if default is not None else False, key=f"b_{label}")
    else:
        # textual/miscelánea
        if default is None:
            default = ""
        # campos largos
        if "text" in ctype.lower() or "json" in ctype.lower():
            return st.text_area(label, value=str(default), key=f"ta_{label}")
        return st.text_input(label, value=str(default), key=f"t_{label}")

def build_search_where(cols: List[Dict[str, Any]], q: str) -> str:
    """
    Construye un WHERE genérico ILIKE para búsqueda en múltiples columnas.
    Solo usa columnas del esquema (seguras); concatena con OR.
    """
    if not q:
        return ""
    parts = []
    for c in cols:
        cname = c["name"]
        ctype = str(c.get("type", ""))
        if is_textual(ctype) or is_numeric(ctype):
            parts.append(f'"{cname}"::text ILIKE :q')
        elif is_temporal(ctype):
            parts.append(f'TO_CHAR("{cname}"::timestamp, \'YYYY-MM-DD"T"HH24:MI:SS\') ILIKE :q')
    if not parts:
        return ""
    return " WHERE " + " OR ".join(parts)

def fetch_records(engine: Engine, table: str, search: str, limit: int, offset: int, pk: Optional[str], cols: List[Dict[str, Any]]):
    where = build_search_where(cols, search)
    order = f' ORDER BY "{pk}" DESC' if pk else ""
    sql = f'SELECT * FROM "{table}"' + where + order + " LIMIT :limit OFFSET :offset"
    params = {"limit": limit, "offset": offset}
    if where:
        params["q"] = f"%{search}%"
    with engine.connect() as con:
        df = pd.read_sql(text(sql), con, params=params)
    return df

def count_records(engine: Engine, table: str, search: str, cols: List[Dict[str, Any]]):
    where = build_search_where(cols, search)
    sql = f'SELECT COUNT(*) AS c FROM "{table}"' + where
    params = {}
    if where:
        params["q"] = f"%{search}%"
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

def drive_image(url: str):
    u = normalize_drive_url(url or "")
    if u:
        st.image(u, use_container_width=True, caption="image_url")
    else:
        st.info("Sin imagen asociada.")

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Catálogo HIBLOOMS", layout="wide")

st.title("📖 Catálogo HIBLOOMS")
st.caption("Explora, busca, añade y edita registros de todas las tablas.")

# Conexión
try:
    engine = get_engine()
except Exception as e:
    st.error(f"❌ Error obteniendo conexión: {e}")
    st.stop()

# Descubrir tablas
insp = inspect(engine)
all_tables = insp.get_table_names(schema="public")

if not all_tables:
    st.warning("No se han encontrado tablas en el esquema 'public'.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controles")
    table = st.selectbox("Tabla", all_tables, index=max(all_tables.index("lab_images") if "lab_images" in all_tables else 0, 0))
    search = st.text_input("🔎 Búsqueda (texto libre)", value="", placeholder="Escribe y pulsa Enter…")
    page_size = st.select_slider("Registros por página", options=[6, 12, 24, 48], value=12)
    page = st.number_input("Página", min_value=1, step=1, value=1)

    st.markdown("---")
    st.subheader("➕ Añadir nuevo")
    add_open = st.checkbox("Abrir formulario de creación", value=False)

# Metadatos de la tabla
cols = get_table_columns(engine, table)
pk = infer_pk(engine, table) or (cols[0]["name"] if cols else None)

# Conteo + datos página
total = count_records(engine, table, search, cols)
offset = (page - 1) * page_size
df = fetch_records(engine, table, search, page_size, offset, pk, cols)

st.write(f"**Tabla:** `{table}` • **Filas coincidentes:** {total} • **Página:** {page}")

# ---------------------------
# Formulario de creación
# ---------------------------
if add_open:
    with st.expander(f"➕ Crear registro en **{table}**", expanded=True):
        with st.form(f"form_create_{table}", clear_on_submit=False):
            create_values = {}
            for c in cols:
                cname = c["name"]
                if cname == pk and c.get("autoincrement", False):
                    # No pedir PK autoincremental
                    continue
                create_values[cname] = render_input_for_column(c, default=None)

            submitted = st.form_submit_button("Crear registro")
            if submitted:
                try:
                    data_clean = {k: python_value_for_sql(v) for k, v in create_values.items()}
                    # Si PK es autoincremental, no lo incluimos
                    if pk in data_clean and any(s in str(cols_by_name := {cc['name']: cc for cc in cols}[pk].get("type","")).lower() for s in ["serial", "identity"]):
                        data_clean.pop(pk, None)
                    insert_record(engine, table, data_clean)
                    st.success("✅ Registro creado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creando registro: {e}")

# ---------------------------
# Rejilla de tarjetas
# ---------------------------

# Elegir campos a mostrar en tarjeta
display_fields = pick_display_fields(cols)

# Tres columnas responsivas
n_cols = 3
rows = [df.iloc[i:i+n_cols] for i in range(0, len(df), n_cols)]
for chunk in rows:
    cols_ui = st.columns(n_cols, gap="large")
    for (idx, row), col_ui in zip(chunk.iterrows(), cols_ui):
        with col_ui:
            with st.container(border=True):
                # Cabecera con PK
                if pk and pk in row:
                    st.markdown(f"**#{row[pk]}**  —  `{table}`")

                # Imagen si es lab_images y hay image_url
                if table == "lab_images" and "image_url" in row and pd.notna(row["image_url"]):
                    drive_image(str(row["image_url"]))

                # Metadatos clave
                md = []
                for f in display_fields:
                    if f in row and pd.notna(row[f]):
                        val = row[f]
                        if isinstance(val, (datetime, date)):
                            val = val
                        elif isinstance(val, float):
                            val = round(val, 4)
                        md.append(f"**{f}**: {val}")
                if md:
                    st.markdown("\n\n".join(md))

                # Botones acciones
                b1, b2, b3 = st.columns([1, 1, 1])
                with b1:
                    show = st.button("🔎 Ver", key=f"view_{table}_{idx}")
                with b2:
                    edit = st.button("✏️ Editar", key=f"edit_{table}_{idx}")
                with b3:
                    delete = st.button("🗑️ Borrar", key=f"del_{table}_{idx}") if pk else None

                # Vista detallada
                if show:
                    with st.expander(f"Detalles #{row[pk] if pk else idx}", expanded=True):
                        for c in cols:
                            cname = c["name"]
                            st.write(f"**{cname}**: {row.get(cname)}")

                # Edición
                if edit:
                    with st.expander(f"Editar #{row[pk] if pk else idx}", expanded=True):
                        with st.form(f"form_edit_{table}_{idx}", clear_on_submit=False):
                            new_values = {}
                            for c in cols:
                                cname = c["name"]
                                if cname == pk:
                                    st.text_input(cname, value=str(row.get(cname)), disabled=True)
                                    continue
                                new_values[cname] = render_input_for_column(c, default=row.get(cname))
                            s = st.form_submit_button("Guardar cambios")
                            if s:
                                try:
                                    update_record(engine, table, pk, row[pk], new_values)
                                    st.success("✅ Cambios guardados.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error actualizando: {e}")

                # Borrado
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

# ---------------------------
# Paginación info
# ---------------------------
total_pages = max(1, (total + page_size - 1) // page_size)
st.markdown(f"Página **{page}** de **{total_pages}**")
