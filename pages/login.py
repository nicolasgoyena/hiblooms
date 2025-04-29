import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import requests

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
MY_IP = st.secrets["auth"]["my_ip"]  # Tu IP segura en secrets

# Función para obtener IP pública
def get_public_ip():
    try:
        ip = requests.get('https://api.ipify.org').text
        return ip
    except:
        return None

visitor_ip = get_public_ip()

# Login automático si IP coincide
if visitor_ip == MY_IP:
    st.session_state["logged_in"] = True
    switch_page("app")
else:
    # Formulario de login normal
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
