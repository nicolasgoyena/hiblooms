import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Configuración
st.set_page_config(initial_sidebar_state="collapsed", page_title="Inicio de sesión – HIBLOOMS", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

USERNAME = st.secrets["auth"]["username"]
PASSWORD = st.secrets["auth"]["password"]

# Detectar si viene con admin=true
query_params = st.query_params
admin_mode = query_params.get("admin", ["false"])[0].lower() == "true"

# Si es modo admin y no se ha logueado ya
if admin_mode and not st.session_state.get("logged_in", False):
    st.session_state["logged_in"] = True
    switch_page("app")
    st.stop()

# Si ya está logueado (por admin o por login previo)
if st.session_state.get("logged_in", False):
    switch_page("app")
    st.stop()

# Mostrar formulario de login normal
st.title("🔒 Iniciar sesión en HIBLOOMS")

with st.form("login_form"):
    user = st.text_input("Usuario")
    pwd = st.text_input("Contraseña", type="password")
    submit = st.form_submit_button("Iniciar sesión")

if submit:
    if user == USERNAME and pwd == PASSWORD:
        st.session_state["logged_in"] = True
        switch_page("app")
    else:
        st.error("❌ Usuario o contraseña incorrectos")


