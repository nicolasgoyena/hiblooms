# layout/tablas.py

import streamlit as st
import pandas as pd

def mostrar_pestana_tablas():
    st.subheader("Tablas de Índices Calculados")

    if "data_time" not in st.session_state or not st.session_state["data_time"]:
        st.warning("No hay datos disponibles. Primero realiza el cálculo en la pestaña de Visualización.")
        return

    df_time = pd.DataFrame(st.session_state["data_time"]).copy()

    # Renombrar columnas clave
    df_time.rename(columns={"Point": "Ubicación"}, inplace=True)
    df_time["Fecha"] = pd.to_datetime(df_time["Date"], errors='coerce').dt.strftime("%d-%m-%Y %H:%M")

    # Eliminar columnas innecesarias
    df_time.drop(columns=["Date", "Fecha_formateada", "Fecha_dt", "Fecha-hora"], errors='ignore', inplace=True)

    # Agrupar duplicados de medias de embalse
    df_medias = df_time[df_time["Ubicación"] == "Media_Embalse"]
    df_otros = df_time[df_time["Ubicación"] != "Media_Embalse"]

    if not df_medias.empty:
        columnas_valor = [col for col in df_medias.columns if col not in ["Ubicación", "Fecha", "Tipo"]]
        df_medias = df_medias.groupby(["Ubicación", "Fecha", "Tipo"], as_index=False).agg({col: "max" for col in columnas_valor})

    df_time = pd.concat([df_medias, df_otros], ignore_index=True)

    # Unificar columnas si solo se seleccionó un índice por tipo
    cols_clorofila = [c for c in ["Chla_Val_cal", "Chla_Bellus_cal"] if c in df_time.columns]
    cols_ficocianina = [c for c in ["PC_Val_cal", "B5_div_B4"] if c in df_time.columns]

    if len(cols_clorofila) == 1 and "Clorofila (µg/L)" not in df_time.columns:
        df_time["Clorofila (µg/L)"] = df_time[cols_clorofila[0]]

    if len(cols_ficocianina) == 1 and "Ficocianina (µg/L)" not in df_time.columns:
        df_time["Ficocianina (µg/L)"] = df_time[cols_ficocianina[0]]

    # Ordenar columnas
    columnas = list(df_time.columns)
    orden = ["Ubicación", "Fecha", "Tipo"]
    for col in ["Clorofila (µg/L)", "Ficocianina (µg/L)"]:
        if col in columnas:
            orden.append(col)
    otras = [col for col in columnas if col not in orden]
    columnas_ordenadas = orden + otras
    df_time = df_time[columnas_ordenadas]

    df_medias = df_time[df_time["Ubicación"] == "Media_Embalse"]
    df_puntos = df_time[df_time["Ubicación"] != "Media_Embalse"]

    if not df_puntos.empty:
        st.markdown("### 📌 Datos en los puntos de interés")
        st.dataframe(df_puntos.sort_values(by="Fecha").reset_index(drop=True))

    if not df_medias.empty:
        st.markdown("### 💧 Datos de medias del embalse")
        st.dataframe(df_medias.sort_values(by="Fecha").reset_index(drop=True))
