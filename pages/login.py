import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Configuración visual
st.set_page_config(initial_sidebar_state="collapsed", page_title="Inicio de sesión – HIBLOOMS", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Datos desde secrets
USERNAME = st.secrets["auth"]["username"]
PASSWORD = st.secrets["auth"]["password"]

# Comprobamos si hay un "admin" en la URL
query_params = st.experimental_get_query_params()
admin_mode = query_params.get("admin", ["false"])[0].lower() == "true"

# Si está en modo admin -> acceso automático
if admin_mode:
    st.session_state["logged_in"] = True
    switch_page("app")
    st.stop()

# Si no -> login normal
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

