# layout/visualization.py

import streamlit as st
from datetime import datetime, timedelta
from logic.shapefiles import obtener_nombres_embalses, cargar_y_mostrar_embalses, load_reservoir_shapefile, gdf_to_ee_geometry
from logic.dates import get_available_dates
from logic.image_processing import process_sentinel2, calcular_media_diaria_embalse
from logic.points import get_values_at_point
from logic.download import generar_url_geotiff_multibanda
from logic.leyendas import generar_leyenda
from logic.datos_sonda import cargar_csv_desde_url
from streamlit_folium import folium_static
import geemap
import altair as alt
import pandas as pd
import folium
import os
import zipfile
import tempfile

def mostrar_pestana_visualizacion(puntos_interes):

    st.subheader("🔄 Cargar shapefile propio con todos los embalses de tu interés (opcional)")
    st.info("📄 Asegúrate de que el shapefile contiene una columna llamada **'NOMBRE'** con el nombre de cada embalse.")

    uploaded_zip = st.file_uploader("Sube un archivo ZIP con tu shapefile de embalses (proyección EPSG:32630)", type=["zip"])
    custom_shapefile_path = None

    if uploaded_zip is not None:
        temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir.name)
        for file in os.listdir(temp_dir.name):
            if file.endswith(".shp"):
                custom_shapefile_path = os.path.join(temp_dir.name, file)
                break
        if custom_shapefile_path:
            st.success("✅ Shapefile cargado correctamente.")
        else:
            st.error("❌ No se encontró ningún archivo .shp válido en el ZIP.")

    col_mapa, col_controles = st.columns([2, 2])
    with col_mapa:
        st.subheader("Mapa de Embalses")
        mapa = geemap.Map(center=[42.0, 0.5], zoom=8)
        cargar_y_mostrar_embalses(mapa, shapefile_path=custom_shapefile_path or "shapefiles/embalses_hiblooms.shp", nombre_columna="NOMBRE")
        folium_static(mapa, width=1000, height=600)

    with col_controles:
        st.subheader("Selección de Embalse")
        nombres_embalses = obtener_nombres_embalses(custom_shapefile_path or "shapefiles/embalses_hiblooms.shp")
        reservoir_name = st.selectbox("Selecciona un embalse:", nombres_embalses)

        if reservoir_name:
            gdf = load_reservoir_shapefile(reservoir_name, shapefile_path=custom_shapefile_path or "shapefiles/embalses_hiblooms.shp")
            if gdf is not None:
                aoi = gdf_to_ee_geometry(gdf)

                st.subheader("Selecciona un porcentaje máximo de nubosidad:")
                max_cloud_percentage = st.slider("Porcentaje máximo de nubosidad:", 0, 100, 10)

                st.subheader("Selecciona el intervalo de fechas:")
                date_range = st.date_input("Rango de fechas:", value=(datetime.today() - timedelta(days=15), datetime.today()))
                start_date, end_date = [d.strftime("%Y-%m-%d") for d in date_range]

                st.subheader("Selecciona los índices a visualizar:")
                available_indices = ["MCI", "B5_div_B4", "NDCI_ind", "PC_Val_cal", "Chla_Val_cal", "Chla_Bellus_cal"]
                selected_indices = st.multiselect("Índices:", available_indices)

                if st.button("Calcular y mostrar resultados"):
                    st.session_state.clear()  # Limpiar sesión previa
                    with st.spinner("🔎 Buscando imágenes disponibles..."):
                        available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)
                        if not available_dates:
                            st.warning("⚠️ No se han encontrado imágenes disponibles.")
                            return

                    st.success(f"✅ Se encontraron {len(available_dates)} fechas con imágenes útiles.")
                    st.session_state["available_dates"] = available_dates
                    st.session_state["selected_indices"] = selected_indices
                    st.session_state["aoi"] = aoi
                    st.session_state["reservoir_name"] = reservoir_name
                    st.session_state["gdf"] = gdf
                    st.session_state["start_date"] = start_date
                    st.session_state["end_date"] = end_date
                    st.session_state["max_cloud_percentage"] = max_cloud_percentage
                    st.session_state["puntos_interes"] = puntos_interes
