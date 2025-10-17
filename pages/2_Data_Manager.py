# pages/2_Data_Manager.py
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from db_utils import (
    get_engine, get_columns, infer_pk, read_table,
    insert_row, update_row, delete_row, default_widget_value
)

st.set_page_config(page_title="HIBLOOMS · Data Manager", layout="wide")
st.title("📚 HIBLOOMS · Gestión de Datos (Drive URLs)")

# =========================
# Configuración
# =========================
SCHEMA = "public"
# Ajusta esta lista a tus tablas reales:
ALLOWED_TABLES = [
    "extraction_points",
    "insitu_determinations",
    "insitu_sampling",
    "lab_images",
    "profiles_data",
    "reservoirs_spain",
    "rivers_spain",
    "samples",
    "sediment_data",
    "sensor_data",
]
# Nombres de columnas que contienen la URL de imagen:
IMAGE_URL_COLUMNS = ["image_url", "url", "photo_url"]

PAGE_SIZE = 50
engine = get_engine()

# =========================
# Utilidades de Google Drive
# =========================
def normalize_drive_url(url: str) -> Optional[Dict[str, str]]:
    """
    Acepta enlaces de Drive en formatos habituales y devuelve:
    - view_url  -> uc?export=view&id=...
    - download_url -> uc?export=download&id=...
    - thumb_url -> thumbnail?id=...
    Si no detecta un ID válido, devuelve None.
    """
    if not url:
        return None

    # 1) .../file/d/<ID>/...
    m = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    # 2) ...?id=<ID>  (o url-encoded id%3D<ID>)
    if not m:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url) or \
            re.search(r"[?&]id%3D([a-zA-Z0-9_-]{10,})", url)
    # 3) .../uc?id=<ID> (ya “casi” normalizada)
    if not m:
        m = re.search(r"/uc\?[^ ]*id=([a-zA-Z0-9_-]{10,})", url)
    if not m:
        return None

    file_id = m.group(1)
    view = f"https://drive.google.com/uc?export=view&id={file_id}"
    down = f"https://drive.google.com/uc?export=download&id={file_id}"
    thumb = f"https://drive.google.com/thumbnail?id={file_id}"
    return {"id": file_id, "view_url": view, "download_url": down, "thumb_url": thumb}

def normalize_if_drive(url: Optional[str]) -> Optional[str]:
    """Si es un enlace de Drive, devuelve la view_url normalizada; si no, deja la URL tal cual."""
    if not url:
        return url
    norm = normalize_drive_url(url)
    return norm["view_url"] if norm else url

def maybe_drive_preview(url: str):
    """Muestra previsualización de imagen usando la URL normalizada si es de Drive; si no, la URL tal cual."""
    norm = normalize_drive_url(url)
    show_url = norm["view_url"] if norm else url
    st.image(show_url, caption="Previsualización", use_container_width=True)
    if norm:
        with st.expander("Enlaces del archivo (Drive)"):
            st.write("**Ver:**", norm["view_url"])
            st.write("**Descargar:**", norm["download_url"])

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Opciones")
table = st.sidebar.selectbox("Tabla", options=ALLOWED_TABLES)
search_q = st.sidebar.text_input("Filtro rápido (texto)", placeholder="Buscar en todas las columnas…")
page_idx = st.sidebar.number_input("Página", min_value=1, value=1, step=1)

# =========================
# Metadatos de la tabla
# =========================
cols = get_columns(engine, table, SCHEMA)
pk = infer_pk(engine, table, SCHEMA)
col_names = [c["name"] for c in cols]
searchable_cols = col_names

# =========================
# Visor (lectura con paginado)
# =========================
df, total = read_table(
    engine, table, SCHEMA,
    limit=PAGE_SIZE, offset=(page_idx-1)*PAGE_SIZE,
    search=search_q.strip() or None,
    searchable_cols=searchable_cols
)

c0, c1 = st.columns([3, 1])
with c0:
    st.subheader(f"📄 Registros en `{table}`")
with c1:
    st.metric("Total", total)

image_col = next((c for c in IMAGE_URL_COLUMNS if c in df.columns), None)
if image_col:
    st.caption(
        f"Si `{table}` tiene una columna de imagen (p. ej., **{image_col}**), "
        "puedes pegar una URL de Google Drive; se normaliza a un enlace de vista directa."
    )

st.dataframe(df, use_container_width=True)
st.download_button(
    "⬇️ Exportar CSV (página actual)",
    data=df.to_csv(index=False),
    file_name=f"{table}_page{page_idx}.csv",
    mime="text/csv"
)

st.markdown("---")
st.subheader("✍️ Añadir / Editar")

mode = st.radio("Acción", ["Añadir nuevo", "Editar existente"], horizontal=True)

# =========================
# Formulario dinámico
# =========================
def build_form(initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for c in cols:
        name = c["name"]
        if name == pk:
            if mode == "Editar existente":
                st.text_input(f"{name} (PK)", value=str(initial.get(name)) if initial else "", disabled=True)
            continue

        sql_t = c["type"]
        kind = default_widget_value(sql_t)
        default_val = initial.get(name) if initial else None

        if name in IMAGE_URL_COLUMNS:
            # Campo especial para URL de imagen (Drive o URL directa)
            v = st.text_input(
                name,
                value=str(default_val) if default_val is not None else "",
                placeholder="Pega aquí la URL compartida de Google Drive (o URL directa)"
            )
            # Previsualización inmediata si hay valor
            if v:
                st.caption("Previsualización de imagen a partir de la URL indicada:")
                maybe_drive_preview(v)
            values[name] = v if v else None
        else:
            if kind == "number":
                v = st.text_input(name, value=str(default_val) if default_val is not None else "")
                values[name] = None if v == "" else (float(v) if "." in v else int(v))
            elif kind == "bool":
                v = st.checkbox(name, value=bool(default_val) if default_val is not None else False)
                values[name] = v
            elif kind == "date":
                v = st.text_input(
                    name,
                    value=str(default_val) if default_val is not None else "",
                    placeholder="YYYY-MM-DD o ISO 8601"
                )
                values[name] = v if v else None
            else:
                values[name] = st.text_input(name, value=str(default_val) if default_val is not None else "")

    return values

# =========================
# Acciones
# =========================
if mode == "Añadir nuevo":
    values = build_form()

    # Normaliza automáticamente URLs de Drive en las columnas designadas
    for c in IMAGE_URL_COLUMNS:
        if c in values and values[c]:
            values[c] = normalize_if_drive(values[c])

    if st.button("➕ Insertar registro", type="primary"):
        insert_row(engine, table, SCHEMA, values)
        st.success("Registro insertado.")
        st.rerun()

else:
    if pk is None:
        st.warning("No se ha encontrado PK en esta tabla; para editar/borrar es recomendable tener clave primaria.")
    else:
        if df.empty or pk not in df.columns:
            st.info("No hay registros en la página actual o no se puede localizar la PK para esta vista.")
        else:
            pk_options = df[pk].dropna().astype(str).tolist()
            selected_pk = st.selectbox("Elige registro (PK)", options=pk_options)

            if selected_pk:
                row = df[df[pk].astype(str) == selected_pk].iloc[0].to_dict()
                values = build_form(initial=row)

                # Normaliza automáticamente URLs de Drive en las columnas designadas
                for c in IMAGE_URL_COLUMNS:
                    if c in values and values[c]:
                        values[c] = normalize_if_drive(values[c])

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Guardar cambios", type="primary"):
                        update_row(engine, table, SCHEMA, pk, row[pk], values)
                        st.success("Registro actualizado.")
                        st.rerun()
                with c2:
                    if st.button("🗑️ Borrar registro", type="secondary"):
                        delete_row(engine, table, SCHEMA, pk, row[pk])
                        st.success("Registro eliminado.")
                        st.rerun()

st.markdown("---")
st.caption(
    "Para que las imágenes de Google Drive se vean sin iniciar sesión, "
    "asegúrate de ponerlas en **“Cualquier usuario con el enlace (Lector)”**. "
    "Las URLs de Drive se normalizan al formato `uc?export=view&id=...` automáticamente al guardar."
)
