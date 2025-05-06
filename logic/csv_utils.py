# logic/csv_utils.py

import pandas as pd
import streamlit as st

@st.cache_data
def cargar_csv_desde_url(url: str) -> pd.DataFrame:
    """Carga un CSV desde una URL, manejando errores y ajustando la columna de fecha."""
    try:
        df = pd.read_csv(url)

        if 'Time' in df.columns:
            df.rename(columns={'Time': 'Fecha-hora'}, inplace=True)

        df['Fecha-hora'] = pd.to_datetime(df['Fecha-hora'], format='mixed')

        return df
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el CSV desde {url}: {e}")
        return pd.DataFrame()
