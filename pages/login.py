import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed", page_title="HIBLOOMS Login", layout="wide")

# ⬇️ Ocultar la navegación lateral de páginas (menu multipage)
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

USERNAME = st.secrets["auth"]["username"]
PASSWORD = st.secrets["auth"]["password"]

st.title("🔒 Iniciar sesión")

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
