# hiblooms_modular.py

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import ee
import json

from layout.introduccion import mostrar_pestana_introduccion
from layout.visualizacion import mostrar_pestana_visualizacion
from layout.tablas import mostrar_pestana_tablas

# Configuración inicial de la app
st.set_page_config(initial_sidebar_state="collapsed", page_title="HIBLOOMS – Visor de embalses", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none; }
        h1 a, h2 a, h3 a {
            display: none !important;
            pointer-events: none !important;
            text-decoration: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Autenticación con GEE
if not st.session_state.get("logged_in", False):
    switch_page("login")

try:
    if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:
        json_object = json.loads(st.secrets["GEE_SERVICE_ACCOUNT_JSON"], strict=False)
        service_account = json_object["client_email"]
        credentials = ee.ServiceAccountCredentials(service_account, key_data=json.dumps(json_object))
        ee.Initialize(credentials)
    else:
        st.write("🔍 Intentando inicializar GEE localmente...")
        ee.Initialize()
except Exception as e:
    st.error(f"❌ No se pudo inicializar Google Earth Engine: {str(e)}")
    st.stop()

# ───────────────────────
# Cargar pestañas
tab1, tab2, tab3 = st.tabs(["Introducción", "Visualización", "Tablas"])

with tab1:
    mostrar_pestana_introduccion()

with tab2:
    mostrar_pestana_visualizacion()

with tab3:
    mostrar_pestana_tablas()
