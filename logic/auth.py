# logic/auth.py
import ee
import streamlit as st
import json

def initialize_gee():
    try:
        if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:
            # Convertir el JSON guardado en Streamlit Secrets a un diccionario
            json_object = json.loads(st.secrets["GEE_SERVICE_ACCOUNT_JSON"], strict=False)
            service_account = json_object["client_email"]
            json_object = json.dumps(json_object)

            # Autenticar con la cuenta de servicio
            credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
            ee.Initialize(credentials)
        else:
            st.write("🔍 Intentando inicializar GEE localmente...")
            ee.Initialize()
    except Exception as e:
        st.error(f"❌ No se pudo inicializar Google Earth Engine: {str(e)}")
        st.stop()
