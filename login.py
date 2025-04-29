import streamlit as st
from streamlit_extras.switch_page_button import switch_page

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
        switch_page("app")  # Redirige al script principal (app.py)
    else:
        st.error("❌ Usuario o contraseña incorrectos")
