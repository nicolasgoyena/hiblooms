# encoding: utf-8

import ee
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed", page_title="HIBLOOMS – Visor de embalses", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }

        /* Ocultar completamente los enlaces automáticos de encabezados (ícono + funcionalidad) */
        h1 a, h2 a, h3 a {
            display: none !important;
            pointer-events: none !important;
            text-decoration: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Bloquear acceso si no está logueado
if not st.session_state.get("logged_in", False):
    switch_page("login")
# Inicialización segura del estado
default_keys = {
    "cloud_results": [],
    "used_cloud_results": [],
    "data_time": [],
    "urls_exportacion": [],
    "available_dates": [],
    "selected_indices": [],
}

for key, default in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default
# Inicialización de las variables 'image_list' y 'selected_dates'
if "image_list" not in st.session_state:
    st.session_state["image_list"] = []

if "selected_dates" not in st.session_state:
    st.session_state["selected_dates"] = []
import geemap.foliumap as geemap
from streamlit_folium import folium_static
from datetime import datetime
import pandas as pd
import altair as alt
from dateutil.relativedelta import relativedelta  # Para calcular ±3 meses
import folium
import geopandas as gpd
import os
import time
from datetime import timedelta
import json
import numpy as np

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

# URL pública del archivo CSV en S3
url_csv = "https://hibloomsbucket.s3.eu-south-2.amazonaws.com/puntos_interes.csv"

try:
    df_poi = pd.read_csv(url_csv)
    puntos_interes = {}
    for _, row in df_poi.iterrows():
        embalse = row["embalse"]
        if embalse not in puntos_interes:
            puntos_interes[embalse] = {}
        puntos_interes[embalse][row["nombre"]] = (row["latitud"], row["longitud"])
except Exception as e:
    st.error(f"Error cargando puntos de interés desde S3: {e}")
    puntos_interes = {}


@st.cache_data
def cargar_fechas_csv(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)

        if 'fecha' in df.columns:
            df.rename(columns={'fecha': 'Fecha'}, inplace=True)
        else:
            st.warning("❌ El CSV no contiene la columna esperada 'fecha'.")
            return pd.DataFrame()

        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)

        return df
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el CSV desde {url}: {e}")
        return pd.DataFrame()

@st.cache_data
def cargar_csv_desde_url(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)

        if 'Time' in df.columns:
            df.rename(columns={'Time': 'Fecha-hora'}, inplace=True)

        df['Fecha-hora'] = pd.to_datetime(df['Fecha-hora'], format='mixed')

        return df
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el CSV desde {url}: {e}")
        return pd.DataFrame()


def obtener_nombres_embalses(shapefile_path="shapefiles/embalses_hiblooms.shp"):
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)

        if "NOMBRE" in gdf.columns:
            nombres_embalses = sorted(gdf["NOMBRE"].dropna().unique())
            return nombres_embalses
        else:
            st.error("❌ El shapefile cargado no contiene una columna llamada 'NOMBRE'. No se pueden mostrar embalses.")
            return []
    else:
        st.error(f"No se encontró el archivo {shapefile_path}.")
        return []

# Función combinada para cargar el shapefile, ajustar el zoom y mostrar los embalses con tooltip
def cargar_y_mostrar_embalses(map_object, shapefile_path="shapefiles/embalses_hiblooms.shp", nombre_columna="NOMBRE"):
    if os.path.exists(shapefile_path):
        gdf_embalses = gpd.read_file(shapefile_path).to_crs(epsg=4326)  # Convertir a WGS84

        # Ajustar el zoom automáticamente a la extensión de los embalses
        bounds = gdf_embalses.total_bounds  # (minx, miny, maxx, maxy)
        map_object.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        for _, row in gdf_embalses.iterrows():
            nombre_embalse = row.get(nombre_columna, "Embalse desconocido")  # Obtener el nombre real

            if row.geometry.geom_type == 'Point':
                folium.Marker(                    location=[row.geometry.y, row.geometry.x],
                    popup=nombre_embalse,
                    tooltip=nombre_embalse,  # Muestra el nombre al hacer hover
                    icon=folium.Icon(color="blue", icon="tint")
                ).add_to(map_object)

            elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                folium.GeoJson(
                    row.geometry,
                    name=nombre_embalse,
                    tooltip=folium.Tooltip(nombre_embalse),  # Muestra el nombre al hacer hover
                    style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.4}
                ).add_to(map_object)

    else:
        st.error(f"No se encontró el archivo {shapefile_path}.")

def get_available_dates(aoi, start_date, end_date, max_cloud_percentage):
    inicio_total = time.time()

    sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date)

    if sentinel2.size().getInfo() == 0:
        st.warning("❌ No se encontraron imágenes de Sentinel-2 para este embalse y rango de fechas.")
        return []

    images = sentinel2.toList(sentinel2.size())
    results_list = []

    for i in range(images.size().getInfo()):
        image = ee.Image(images.get(i)).clip(aoi)
        image_date = image.get('system:time_start').getInfo()
        formatted_date = datetime.utcfromtimestamp(image_date / 1000).strftime('%Y-%m-%d')
        image_time = datetime.utcfromtimestamp(image_date / 1000).strftime('%H:%M')

        # Evitar duplicados
        if any(r["Fecha"] == formatted_date for r in results_list):
            continue

        with st.spinner(f"🕒 Analizando imagen del {formatted_date}..."):
            cloud_obj = calculate_cloud_percentage(image, aoi)
            if cloud_obj is None:
                continue
            
            try:
                cloud_percentage = cloud_obj.getInfo()
            except Exception as e:
                continue

            coverage = calculate_coverage_percentage(image, aoi)
            if coverage is None:
                continue

            if (max_cloud_percentage == 100 or cloud_percentage <= max_cloud_percentage) and coverage >= 50:
                results_list.append({
                    "Fecha": formatted_date,
                    "Hora": image_time,
                    "Nubosidad aproximada (%)": round(cloud_percentage, 2),
                    "Cobertura (%)": round(coverage, 2)
                })

    st.session_state["cloud_results"] = results_list
    return sorted([r["Fecha"] for r in results_list])

def calcular_distribucion_area_por_clases(indices_image, index_name, aoi, bins):
    scl = indices_image.select("SCL")
    year = datetime.utcfromtimestamp(indices_image.get('system:time_start').getInfo() / 1000).year

    if year == 2018:
        mask_agua = scl.eq(6).Or(scl.eq(2))
    else:
        mask_agua = scl.eq(6)

    imagen_indice = indices_image.select(index_name).updateMask(mask_agua)
    pixel_area = ee.Image.pixelArea().updateMask(imagen_indice.mask())

    results = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]

        bin_mask = imagen_indice.gte(lower).And(imagen_indice.lt(upper))
        bin_area = pixel_area.updateMask(bin_mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=20,
            maxPixels=1e13
        ).get("area")

        results.append({
            "rango": f"{lower}–{upper}",
            "area_ha": ee.Number(bin_area).divide(10000)  # m² → ha
        })

    total_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=20,
        maxPixels=1e13
    ).get("area")

    total_area_ha = ee.Number(total_area).divide(10000).getInfo()

    resultados_finales = []
    for r in results:
        area_ha = r["area_ha"].getInfo()
        porcentaje = (area_ha / total_area_ha) * 100
        resultados_finales.append({
            "rango": r["rango"],
            "area_ha": area_ha,
            "porcentaje": porcentaje
        })

    return resultados_finales

def load_reservoir_shapefile(reservoir_name, shapefile_path="shapefiles/embalses_hiblooms.shp"):
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)

        # Verificar existencia del campo 'NOMBRE'
        if "NOMBRE" not in gdf.columns:
            st.error("❌ El shapefile cargado no contiene una columna llamada 'NOMBRE'. Añádela para poder seleccionar embalses.")
            return None

        # Reproyectar automáticamente a EPSG:4326 si no lo está
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Normalizar nombres
        gdf["NOMBRE"] = gdf["NOMBRE"].str.lower().str.replace(" ", "_")
        normalized_name = reservoir_name.lower().replace(" ", "_")

        gdf_filtered = gdf[gdf["NOMBRE"] == normalized_name]

        if gdf_filtered.empty:
            st.error(f"No se encontró el embalse {reservoir_name} en el shapefile.")
            return None

        return gdf_filtered
    else:
        st.error(f"No se encontró el archivo {shapefile_path}.")
        return None

def gdf_to_ee_geometry(gdf):
    if gdf.empty:
        raise ValueError("❌ El shapefile está vacío o no contiene geometrías.")
    
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("❌ El GeoDataFrame debe estar en EPSG:4326.")

    geometry = gdf.geometry.iloc[0]

    if geometry.geom_type == "MultiPolygon":
        geometry = list(geometry.geoms)[0]  # Extrae el primer polígono

    ee_coordinates = list(geometry.exterior.coords)

    ee_geometry = ee.Geometry.Polygon(
        [ee_coordinates],
        geodesic=False  # Suele ser preferible para polígonos pequeños
    )

    return ee_geometry


def calcular_media_diaria_embalse(indices_image, index_name, aoi):
    """Calcula la media del índice dado sobre el embalse solo en píxeles de agua SCL == 6 (o SCL == 2 también para el año 2018)."""
    scl = indices_image.select('SCL')

    # Extraer la fecha de la imagen
    fecha_millis = indices_image.get('system:time_start').getInfo()
    fecha_dt = datetime.utcfromtimestamp(fecha_millis / 1000)
    year = fecha_dt.year

    if year == 2018:
        # Considerar SCL 6 (agua) y 2 (área oscura)
        mask_agua = scl.eq(6).Or(scl.eq(2))
    else:
        mask_agua = scl.eq(6)

    indice_filtrado = indices_image.select(index_name).updateMask(mask_agua)

    media = indice_filtrado.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e13
    ).get(index_name)

    return media.getInfo() if media is not None else None

def calculate_cloud_percentage(image, aoi):
    scl = image.select('SCL')
    
    # Crear máscara de nubes (SCL 7, 8, 9, 10) pero excluir vegetación (SCL 4) y suelo desnudo (SCL 5)
    cloud_mask_scl = scl.eq(7).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    non_valid_mask = scl.eq(4).Or(scl.eq(5))  # Excluir vegetación y suelo desnudo
    
    # Aplicar máscara para excluir píxeles no válidos
    valid_pixels_mask = scl.mask().And(non_valid_mask.Not())

    # Calcular la fracción de nubes solo sobre píxeles válidos
    cloud_fraction_scl = cloud_mask_scl.updateMask(valid_pixels_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e13
    ).get('SCL')

    cloud_mask_prob = image.select('MSK_CLDPRB').gte(10).updateMask(valid_pixels_mask)
    cloud_fraction_prob = cloud_mask_prob.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e13
    ).get('MSK_CLDPRB')

    scl_ok = cloud_fraction_scl is not None
    prob_ok = cloud_fraction_prob is not None

    if not scl_ok and not prob_ok:
        return None  # ❌ No se puede calcular nubosidad con ninguna de las bandas

    if scl_ok and prob_ok:
        return (
            ee.Number(cloud_fraction_scl).multiply(0.95)
            .add(ee.Number(cloud_fraction_prob).multiply(0.05))
            .multiply(100)
        )

    elif scl_ok:
        return ee.Number(cloud_fraction_scl).multiply(100)

    else:  # prob_ok
        return ee.Number(cloud_fraction_prob).multiply(100)

def calculate_coverage_percentage(image, aoi):
    """Devuelve el % del embalse cubierto por la imagen (basado en la banda B4)."""
    try:
        # Capa constante para total de píxeles dentro del embalse
        total_pixels = ee.Image(1).clip(aoi).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=20,
            maxPixels=1e13
        ).get("constant")

        # Píxeles con datos válidos (usamos máscara de B4 como indicador)
        valid_mask = image.select("B4").mask()
        valid_pixels = ee.Image(1).updateMask(valid_mask).clip(aoi).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=20,
            maxPixels=1e13
        ).get("constant")

        if total_pixels is None or valid_pixels is None:
            return 0

        coverage = ee.Number(valid_pixels).divide(ee.Number(total_pixels)).multiply(100)
        return coverage.getInfo()

    except Exception as e:
        print(f"Error al calcular cobertura de imagen: {e}")
        return 0

def process_sentinel2(aoi, selected_date, max_cloud_percentage, selected_indices):
    with st.spinner("Procesando imágenes de Sentinel-2 para " + selected_date + "..."):
        selected_date_ee = ee.Date(selected_date)
        end_date_ee = selected_date_ee.advance(1, 'day')

        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(aoi) \
            .filterDate(selected_date_ee, end_date_ee)

        num_images = sentinel2.size().getInfo()
        if num_images == 0:
            st.warning(f"No hay imágenes disponibles para la fecha {selected_date}")
            return None, None, None

        images = sentinel2.toList(num_images)
        best_image = None

        # Construir un diccionario con los datos precalculados de nubosidad y cobertura
        cloud_results_dict = {r["Fecha"]: r for r in st.session_state.get("cloud_results", [])}

        for i in range(num_images):
            image = ee.Image(images.get(i))
            image_time_millis = image.get('system:time_start').getInfo()
            formatted_date = datetime.utcfromtimestamp(image_time_millis / 1000).strftime('%Y-%m-%d')
            hora = datetime.utcfromtimestamp(image_time_millis / 1000).strftime('%H:%M')

            if formatted_date not in cloud_results_dict:
                continue  # Solo procesar imágenes útiles

            cloud_score = cloud_results_dict[formatted_date]["Nubosidad aproximada (%)"]
            coverage = cloud_results_dict[formatted_date]["Cobertura (%)"]

            if coverage < 50 or cloud_score > max_cloud_percentage:
                continue

            best_image = image
            st.session_state.setdefault("used_cloud_results", []).append({
                "Fecha": formatted_date,
                "Hora": hora,
                "Nubosidad aproximada (%)": round(cloud_score, 2)
            })
            break  # Usamos la mejor disponible y salimos

        if best_image is None:
            st.warning(f"No se encontró ninguna imagen útil para la fecha {selected_date}")
            return None, None, None


        sentinel2_image = best_image
        image_date = sentinel2_image.get('system:time_start').getInfo()
        image_date = datetime.utcfromtimestamp(image_date / 1000).strftime('%Y-%m-%d %H:%M:%S')

        # Aplicar máscara de nubes SOLO a las bandas de índices seleccionados
        scl = sentinel2_image.select('SCL')
        cloud_mask = scl.neq(8).And(scl.neq(9)).And(scl.neq(10))

        bandas_requeridas = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8A']
        bandas_disponibles = sentinel2_image.bandNames().getInfo()

        for banda in bandas_requeridas:
            if banda not in bandas_disponibles:
                st.warning(f"La banda {banda} no está disponible en la imagen del {selected_date}.")
                return None, None, None

        clipped_image = sentinel2_image.clip(aoi)
        optical_bands = clipped_image.select(bandas_requeridas).divide(10000)
        scaled_image = clipped_image.addBands(optical_bands, overwrite=True)

        b3 = scaled_image.select('B3')
        b4 = scaled_image.select('B4')
        b5 = scaled_image.select('B5')
        b6 = scaled_image.select('B6')
        b8A = scaled_image.select('B8A') 

        indices_functions = {
            "MCI": lambda: b5.subtract(b4).subtract((b6.subtract(b4).multiply(705 - 665).divide(740 - 665))).updateMask(cloud_mask).rename('MCI'),
            "PCI_B5/B4": lambda: b5.divide(b4).updateMask(cloud_mask).rename('PCI_B5/B4'),
            "NDCI_ind": lambda: b5.subtract(b4).divide(b5.add(b4)).updateMask(cloud_mask).rename('NDCI_ind'),
            "PC_Val_cal": lambda: (
                ee.Image(100)
                .divide(
                    ee.Image(1).add(
                        (b5.divide(b4).subtract(1.9895)).multiply(-4.6755).exp()
                    )
                )
                .max(0)
                .updateMask(cloud_mask)
                .rename("PC_Val_cal")
            ),

            "Chla_Val_cal": lambda: (
                ee.Image(450)  
                .divide(
                    ee.Image(1).add(
                        (b5.subtract(b4).divide(b5.add(b4)).subtract(0.46))  
                        .multiply(-7.14)  
                        .exp()
                    )
                )
                .max(0)
                .updateMask(cloud_mask)
                .rename("Chla_Val_cal")
            ),
            "PC_Bellus_cal": lambda: (
                ee.Image(16957)
                .multiply(
                    b6.subtract(
                        b8A.multiply(0.96).add(
                            (b3.subtract(b8A)).multiply(0.51)
                        )
                    )
                )
                .add(571)
                .max(0)
                .updateMask(cloud_mask)
                .rename("PC_Bellus_cal")
            ),
            "Chla_Bellus_cal": lambda: (
                ee.Image(112.78)
                .multiply(
                    b5.subtract(b4).divide(b5.add(b4))  # NDI
                )
                .add(10.779)
                .max(0)
                .updateMask(cloud_mask)
                .rename("Chla_Bellus_cal")
            ),
            "UV_PC_Gral_cal": lambda: (
                ee.Image(24.665)
                .multiply(
                    b5.divide(b4).pow(3.4607)
                )
                .max(0)
                .updateMask(cloud_mask)
                .rename("UV_PC_Gral_cal")
            )
        }

        indices_to_add = []
        for index in selected_indices:
            try:
                if index in indices_functions:
                    indices_to_add.append(indices_functions[index]())
            except Exception as e:
                st.warning(f"⚠️ No se pudo calcular el índice {index} en {selected_date}: {e}")

        if not indices_to_add:
            st.warning(f"⚠️ No se generó ningún índice válido para la fecha {selected_date}.")
            return scaled_image, None, image_date

        indices_image = scaled_image.addBands(indices_to_add)
        return scaled_image, indices_image, image_date

def get_values_at_point(lat, lon, indices_image, selected_indices):
    if indices_image is None:
        return None

    # Sentinel-2 tiene 20 m de resolución para estas bandas -> 3x3 píxeles ~ 60x60 m
    buffer_radius_meters = 30  # Radio que cubre un área de 60x60 m
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_radius_meters)

    values = {}
    for index in selected_indices:
        try:
            mean_value = indices_image.select(index).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=20,
                maxPixels=1e13
            ).get(index)
            values[index] = mean_value.getInfo() if mean_value is not None else None
        except Exception as e:
            print(f"⚠️ Error al obtener valor para {index} en punto ({lat}, {lon}): {e}")
            values[index] = None
    return values

def generar_leyenda(indices_seleccionados):
    # Parámetros de visualización para cada índice
    parametros = {
        "MCI": {"min": -0.1, "max": 0.4, "palette": ['blue', 'green', 'yellow', 'red']},
        "PCI_B5/B4": {"min": 0.5, "max": 1.5, "palette": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]},
        "NDCI_ind": {"min": -0.1, "max": 0.4, "palette": ['blue', 'green', 'yellow', 'red']},
        "PC_Val_cal": {"min": 0, "max": 10, "palette": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]},
        "Chla_Val_cal": {"min": 0,"max": 150,"palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']},
        "Chla_Bellus_cal": {"min": 5,"max": 100,"palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']},
        "PC_Bellus_cal": {"min": 25,"max": 500,"palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']},
        "UV_PC_Gral_cal": {"min": 0,"max": 100,"palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']}
    }

    leyenda_html = "<div style='border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: white;'>"
    leyenda_html += "<h4 style='text-align: center;'>📌 Leyenda de Índices y Capas</h4>"

    # Leyenda para la capa SCL (Scene Classification Layer)
    scl_palette = {
        1: ('#ff0004', 'Píxeles saturados/defectuosos'),
        2: ('#000000', 'Píxeles de área oscura'),
        3: ('#8B4513', 'Sombras de nube'),
        4: ('#00FF00', 'Vegetación'),
        5: ('#FFD700', 'Suelo desnudo'),
        6: ('#0000FF', 'Agua'),
        7: ('#F4EEEC', 'Probabilidad baja de nubes / No clasificada'),
        8: ('#C8C2C0', 'Probabilidad media de nubes'),
        9: ('#706C6B', 'Probabilidad alta de nubes'),
        10: ('#87CEFA', 'Cirro'),
        11: ('#00FFFF', 'Nieve o hielo')
    }

    leyenda_html += "<b>Capa SCL (Clasificación de Escena):</b><br>"
    for val, (color, desc) in scl_palette.items():
        leyenda_html += f"<div style='display: flex; align-items: center;'><div style='width: 15px; height: 15px; background-color: {color}; border: 1px solid black; margin-right: 5px;'></div> {desc}</div>"

    leyenda_html += "<br>"

    # Leyenda para la capa MSK_CLDPRB (Probabilidad de nubes)
    msk_palette = ["blue", "green", "yellow", "red", "black"]
    leyenda_html += "<b>Capa MSK_CLDPRB (Probabilidad de Nubes):</b><br>"
    leyenda_html += f"<div style='background: linear-gradient(to right, {', '.join(msk_palette)}); height: 20px; border: 1px solid #000;'></div>"
    leyenda_html += "<div style='display: flex; justify-content: space-between; font-size: 12px;'><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>"
    leyenda_html += "<br>"

    # Leyenda para los índices seleccionados
    for indice in indices_seleccionados:
        if indice in parametros:
            min_val = parametros[indice]["min"]
            max_val = parametros[indice]["max"]
            palette = parametros[indice]["palette"]

            # Construcción del gradiente CSS
            gradient_colors = ", ".join(palette)
            gradient_style = f"background: linear-gradient(to right, {gradient_colors}); height: 20px; border: 1px solid #000;"

            leyenda_html += f"<b>{indice}:</b><br>"
            leyenda_html += f"<div style='{gradient_style}'></div>"

            # Crear marcadores intermedios
            markers_html = "<div style='display: flex; justify-content: space-between; margin-top: 5px;'>"
            num_colores = len(palette)
            for i in range(num_colores):
                valor = min_val + (max_val - min_val) * i / (num_colores - 1) if num_colores > 1 else min_val
                valor_formateado = f"{valor:.2f}" if isinstance(valor, float) else str(valor)

                markers_html += (
                    "<div style='display: flex; flex-direction: column; align-items: center;'>"
                    "<div style='width: 1px; height: 8px; background-color: black;'></div>"
                    f"<span style='font-size: 12px;'>{valor_formateado}</span>"
                    "</div>"
                )
            markers_html += "</div>"
            leyenda_html += markers_html + "<br>"

    leyenda_html += "</div>"

    # Mostrar la leyenda en Streamlit
    st.markdown(leyenda_html, unsafe_allow_html=True)

def generar_url_geotiff_multibanda(indices_image, selected_indices, region, scale=20):
    try:
        url = indices_image.select(selected_indices).getDownloadURL({
            'scale': scale,
            'region': region.getInfo()['coordinates'],
            'fileFormat': 'GeoTIFF'
        })
        return url
    except Exception as e:
        return None


# INTERFAZ DE STREAMLIT

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Raleway:wght@600;700&display=swap');

    /* Fondo beige solo detrás de todo */
    .stApp {
        background-color: #e8e3d9 !important;
    }

    /* Tipografía global en todo el contenido */
    html, body, .stApp, p, label, span, h4, h5, h6, li, ul, ol, th, td, div, button, input, textarea, select, .css-1cpxqw2, .css-ffhzg2 {
        font-family: 'Playfair Display', serif !important;
        color: #000000;
    }

    /* TITULARES en "Aquatico" (simulado con Raleway) */
    h1, h2, h3, .main-title, .important-header {
        font-family: 'Raleway', sans-serif !important;
        color: #475a23;
        text-transform: uppercase;
        font-weight: 700;
    }

    /* Evitamos que ciertos elementos pierdan sus estilos */
    .stAlert, .stFileUploader, .stDataFrame, .stTable, .stExpander, .stButton, .stSelectbox, .stSlider {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
    }

    /* Tabs */
    div[role="tablist"] {
        font-family: 'Playfair Display', serif;
        display: flex;
        justify-content: center;
        font-size: 18px;
        font-weight: bold;
    }

    button[role="tab"] {
        background-color: #92c3ea33 !important;
        color: #475a23 !important;
        padding: 10px 25px;
        margin: 2px;
        border-radius: 0.5rem;
        font-size: 18px;
        font-family: 'Playfair Display', serif;
        border: 2px solid #92c3ea;
    }

    button[role="tab"][aria-selected="true"] {
        background-color: #5297d2 !important;
        color: white !important;
        border-color: #5297d2 !important;
    }

    /* Botones */
    .stButton>button {
        background-color: #5297d2;
        color: white;
        font-family: 'Playfair Display', serif;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #92c3ea;
        color: black;
    }

    /* Cuadros informativos personalizados */
    .info-box {
        background-color: white;
        border-left: 6px solid #babf0a;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 8px;
        font-family: 'Playfair Display', serif;
    }

    /* Enlaces */
    a {
        color: #5297d2 !important;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }
    
    /* Fondo beige completo para los textos encima de los inputs */
.stSelectbox label, 
.stNumberInput label, 
.stTextInput label, 
.stSlider label {
    background-color: #e8e3d9 !important;
    padding: 0.3rem 0.7rem;
    border-radius: 0.3rem;
    display: block;
    width: 100%;
}
    /* Elimina fondo blanco en contenedores del botón */
section[data-testid="stButton"] {
    background-color: transparent !important;
}

/* Forzar fondo beige en toda la app excepto en widgets */
div.block-container {
    background-color: #eee7dc !important;  /* Marrón claro */
}

/* Evita que botones, selects y otros elementos tengan fondo blanco lateral */
.stButton, .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
    background-color: transparent !important;
}

/* Fondo para el texto de los widgets como file_uploader */
label[data-testid="stWidgetLabel"] > div {
    background-color: #e8e3d9 !important;
    padding: 0.5rem 0.8rem;
    border-radius: 0.5rem;
    display: inline-block;
}

/* Forzar fondo beige del contenedor completo del file_uploader */
div[data-testid="stFileUploader"] {
    background-color: #e8e3d9 !important;
    padding: 0.5rem;
    border-radius: 0.5rem;
}

/* También aplicarlo al dropzone interno */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #e8e3d9 !important;
    border: none !important;
    border-radius: 0.5rem;
}

div[data-testid="element-container"] {
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    background-color: transparent !important;
}



</style>

""", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 4, 1.25])

with col1:
    st.image("images/logo_hiblooms.png", width=280)  # reducido ligeramente
    st.image("images/ministerio.png", width=280)

with col2:
    st.markdown(
        """
        <h1 style="text-align: center; line-height: 1.1em; font-size: 32px; margin: 0.3em 0;">
            Visor de indicadores de la calidad del agua en embalses españoles:
            <br><span style="display: block; text-align: center;">Proyecto HIBLOOMS</span>
        </h1>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.image("images/bioma.jpg", width=240)  # también reducido solo un poco
    col3a, col3b = st.columns([1, 1])
    with col3a:
        st.image("images/logo_ebro.png", width=120)
    with col3b:
        st.image("images/logo_jucar.png", width=120)




tab1, tab2, tab3, tab4 = st.tabs(["Introducción", "Visualización", "Tablas", "Modo rápido"])
with tab1:
    st.markdown("""
        <style>
            .header {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                padding: 10px;
                background-color: #1f77b4;
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .info-box {
                padding: 15px;
                border-radius: 10px;
                background-color: #f4f4f4;
                margin-bottom: 15px;
            }
            .highlight {
                font-weight: bold;
                color: #1f77b4;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="header">Reconstrucción histórica y estado actual de la proliferación de cianobacterias en embalses españoles (HIBLOOMS)</div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box"><b>Alineación con estrategias nacionales:</b><br>📌 Plan Nacional de Adaptación al Cambio Climático (PNACC 2021-2030)<br>📌 Directiva Marco del Agua 2000/60/EC<br>📌 Objetivo de Desarrollo Sostenible 6: Agua limpia y saneamiento</div>',
        unsafe_allow_html=True)

    st.subheader("Justificación")
    st.markdown("""
        La proliferación de cianobacterias en embalses es una preocupación ambiental y de salud pública.
        El proyecto **HIBLOOMS** busca evaluar la evolución histórica y actual de estos eventos en los embalses de España, contribuyendo a:
        - La monitorización de parámetros clave del cambio climático y sus efectos en los ecosistemas acuáticos.
        - La identificación de factores ambientales y de contaminación que influyen en la proliferación de cianobacterias.
        - La generación de información para mejorar la gestión y calidad del agua en España.
    """)

    st.subheader("Hipótesis y Relevancia del Proyecto")
    st.markdown("""
        Se estima que **40% de los embalses españoles** son susceptibles a episodios de proliferación de cianobacterias.
        En un contexto de cambio climático, donde las temperaturas y la eutrofización aumentan, el riesgo de proliferaciones tóxicas es mayor.

        🛰 **¿Cómo abordamos este desafío?**
        - Uso de **teledetección satelital** para monitoreo en tiempo real.
        - Implementación de **técnicas avanzadas de análisis ambiental** para evaluar las causas y patrones de proliferación.
        - Creación de modelos para predecir episodios de blooms y sus impactos en la salud y el medio ambiente.
    """)

    st.subheader("Impacto esperado")
    st.markdown("""
        El proyecto contribuirá significativamente a la gestión sostenible de embalses, proporcionando herramientas innovadoras para:
        - Evaluar la **calidad del agua** con técnicas avanzadas.
        - Diseñar estrategias de mitigación para **minimizar el riesgo de toxicidad**.
        - Colaborar con administraciones públicas y expertos para la **toma de decisiones basada en datos**.
    """)

    st.subheader("Equipo de Investigación")

    st.markdown("""
        <div class="info-box">
            <b>Equipo de Investigación:</b><br>
            🔬 <b>David Elustondo (DEV)</b> - BIOMA/UNAV, calidad del agua, QA/QC y biogeoquímica.<br>
            🔬 <b>Yasser Morera Gómez (YMG)</b> - BIOMA/UNAV, geoquímica isotópica y geocronología con <sup>210</sup>Pb.<br>
            🔬 <b>Esther Lasheras Adot (ELA)</b> - BIOMA/UNAV, técnicas analíticas y calidad del agua.<br>
            🔬 <b>Jesús Miguel Santamaría (JSU)</b> - BIOMA/UNAV, calidad del agua y técnicas analíticas.<br>
            🔬 <b>Carolina Santamaría Elola (CSE)</b> - BIOMA/UNAV, técnicas analíticas y calidad del agua.<br>
            🔬 <b>Adriana Rodríguez Garraus (ARG)</b> - MITOX/UNAV, análisis toxicológico.<br>
            🔬 <b>Sheila Izquieta Rojano (SIR)</b> - BIOMA/UNAV, SIG y teledetección, datos FAIR, digitalización.<br>
        </div>

        <div class="info-box">
            <b>Equipo de Trabajo:</b><br>
            🔬 <b>Aimee Valle Pombrol (AVP)</b> - BIOMA/UNAV, taxonomía de cianobacterias e identificación de toxinas.<br>
            🔬 <b>Carlos Manuel Alonso Hernández (CAH)</b> - Laboratorio de Radioecología/IAEA, geocronología con <sup>210</sup>Pb.<br>
            🔬 <b>David Widory (DWI)</b> - GEOTOP/UQAM, geoquímica isotópica y calidad del agua.<br>
            🔬 <b>Ángel Ramón Moreira González (AMG)</b> - CEAC, taxonomía de fitoplancton y algas.<br>
            🔬 <b>Augusto Abilio Comas González (ACG)</b> - CEAC, taxonomía de cianobacterias y ecología acuática.<br>
            🔬 <b>Lorea Pérez Babace (LPB)</b> - BIOMA/UNAV, técnicas analíticas y muestreo de campo.<br>
            🔬 <b>José Miguel Otano Calvente (JOC)</b> - BIOMA/UNAV, técnicas analíticas y muestreo de campo.<br>
            🔬 <b>Alain Suescun Santamaría (ASS)</b> - BIOMA/UNAV, técnicas analíticas.<br>
            🔬 <b>Leyre López Alonso (LLA)</b> - BIOMA/UNAV, análisis de datos.<br>
            🔬 <b>María José Rodríguez Pérez (MRP)</b> - Confederación Hidrográfica del Ebro, calidad del agua.<br>
            🔬 <b>María Concepción Durán Lalaguna (MDL)</b> - Confederación Hidrográfica del Júcar, calidad del agua.<br>
        </div>
    """, unsafe_allow_html=True)

    st.success(
        "🔬 HIBLOOMS no solo estudia el presente, sino que reconstruye el pasado para entender el futuro de la calidad del agua en España.")
with tab2:
    # 🔄 Cargar shapefile personalizado (fuera de las columnas para que esté disponible antes)
    st.subheader("🔄 Cargar shapefile propio con todos los embalses de tu interés (opcional)")
    st.info("📄 Asegúrate de que el shapefile contiene una columna llamada **'NOMBRE'** con el nombre de cada embalse.")

    uploaded_zip = st.file_uploader("Sube un archivo ZIP con tu shapefile de embalses", type=["zip"])

    custom_shapefile_path = None

    if uploaded_zip is not None:
        import zipfile
        import tempfile

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

    # ──────────────────────────────
    # Dividimos el contenido en dos columnas
    row1 = st.columns([2, 2])
    row2 = st.columns([2, 2])

    with row1[0]:
        st.subheader("Mapa de Embalses")
        map_embalses = geemap.Map(center=[42.0, 0.5], zoom=8)
        cargar_y_mostrar_embalses(
            map_embalses,
            shapefile_path=custom_shapefile_path if custom_shapefile_path else "shapefiles/embalses_hiblooms.shp",
            nombre_columna="NOMBRE"
        )
        folium_static(map_embalses, width=1000, height=600)

    with row1[1]:
        st.subheader("Selección de Embalse")

        nombres_embalses = obtener_nombres_embalses(custom_shapefile_path) if custom_shapefile_path else obtener_nombres_embalses()

        # Seleccionar embalse
        reservoir_name = st.selectbox("Selecciona un embalse:", nombres_embalses)

        if reservoir_name:
            gdf = load_reservoir_shapefile(reservoir_name, shapefile_path=custom_shapefile_path) if custom_shapefile_path else load_reservoir_shapefile(reservoir_name)
            if gdf is not None:
                aoi = gdf_to_ee_geometry(gdf)
                st.subheader("Puntos de interés")

                pois_embalse = {}
                
                if reservoir_name in puntos_interes:
                    st.success("Puntos de interés por defecto disponibles para este embalse.")
                    pois_embalse = puntos_interes[reservoir_name]
                else:
                    st.warning("Este embalse no tiene puntos de interés por defecto. Puedes subir un archivo CSV con columnas llamadas exactamente: 'nombre', 'latitud' y 'longitud'.")
                    archivo_pois = st.file_uploader("Sube un archivo CSV con los puntos de interés", type=["csv"])
                
                    if archivo_pois is not None:
                        try:
                            df_pois = pd.read_csv(archivo_pois)
                            columnas_esperadas = {"nombre", "latitud", "longitud"}
                            columnas_archivo = set(df_pois.columns.str.lower().str.strip())
                            
                            if columnas_esperadas.issubset(columnas_archivo):
                                # Renombrar columnas ignorando mayúsculas/minúsculas y espacios
                                columnas_mapeo = {col: col.lower().strip() for col in df_pois.columns}
                                df_pois = df_pois.rename(columns=columnas_mapeo)
                
                                pois_embalse = {
                                    row["nombre"]: (row["latitud"], row["longitud"]) for _, row in df_pois.iterrows()
                                }
                                puntos_interes[reservoir_name] = pois_embalse
                                st.success("Puntos cargados correctamente.")
                            else:
                                st.error("❌ El archivo debe tener columnas llamadas exactamente: 'nombre', 'latitud' y 'longitud'.")
                        except Exception as e:
                            st.error(f"❌ Error al leer el archivo: {e}")
                
                if pois_embalse:
                    st.markdown("**Puntos de interés activos:**")
                    st.dataframe(pd.DataFrame([
                        {"nombre": nombre, "latitud": lat, "longitud": lon} for nombre, (lat, lon) in pois_embalse.items()
                    ]))

                # Slider de nubosidad
                st.subheader("Selecciona un porcentaje máximo de nubosidad:")
                max_cloud_percentage = st.selectbox(
                    "Dado que las nubes pueden alterar los valores estimados de concentraciones, es importante definir un límite máximo de nubosidad permitida. Es recomendable elegir valores de hasta el 60%, aunque si se quiere ver más imágenes disponibles, se puede aumentar la tolerancia:",
                    options=[60, 80, 100],
                    index=0  # Valor por defecto: 50%
                )
                if max_cloud_percentage == 100:
                    st.info("🔁 Has seleccionado un 100 % de nubosidad permitida: se mostrarán todas las imágenes del periodo. Aun así, se estimará la nubosidad de cada imagen.")

                # Selección de intervalo de fechas
                st.subheader("Selecciona el intervalo de fechas:")
                date_range = st.date_input(
                    "Rango de fechas:",
                    value=(datetime.today() - timedelta(days=15), datetime.today()),  # Últimos 15 días hasta hoy
                    min_value=datetime(2017, 7, 1),  # Fecha mínima permitida
                    max_value=datetime.today(),  # Restringe la selección hasta el día actual
                    format="DD-MM-YYYY"
                )

                # Extraer fechas seleccionadas
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = datetime(2017, 7, 1), datetime.today()

                start_date = start_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')

                # Selección de índices
                st.subheader("Selecciona los índices a visualizar:")
                available_indices = ["MCI", "PCI_B5/B4", "NDCI_ind","UV_PC_Gral_cal", "PC_Val_cal", "Chla_Val_cal", "Chla_Bellus_cal","PC_Bellus_cal"]
                selected_indices = st.multiselect("Selecciona uno o varios índices para visualizar y analizar:", available_indices)
                with st.expander("ℹ️ ¿Qué significa cada índice?"):
                    st.markdown("""
                    - **MCI (Maximum Chlorophyll Index):** Detecta altas concentraciones de clorofila-a, útil para identificar blooms intensos.
                    - **PCI_B5/B4:** Relación espectral entre el infrarrojo cercano (B5) y el rojo (B4), es un buen indicador de ficocianina para todo tipo de embalses, pero no proporciona concentraciones directas.
                    - **NDCI_ind (Normalized Difference Chlorophyll Index):** Relación normalizada entre bandas del rojo e infrarrojo cercano. Se asocia a clorofila-a.
                    - **UV_PC_Gral_cal:** Estimación cuantitativa general de ficocianina basada en la relación espectral entre el infrarrojo cercano (B5) y el rojo (B4). Ajustada mediante una función exponencial, proporciona concentraciones aproximadas de ficocianina en µg/L. Desarrollado por la Universidad de Valencia a partir de datos del estudio en embalses de la cuenca del Ebro (Pérez-González et al., 2021).
                    - **PC_Val_cal (Ficocianina en El Val):** Estimador cuantitativo de ficocianina, un pigmento exclusivo de cianobacterias. Basado en la relación espectral entre el infrarrojo cercano y el rojo, ha sido ajustado a partir de mediciones de ficocianina en el Embalse de El Val.
                    - **Chla_Val_cal:** Calibración cuantitativa de clorofila-a derivada del NDCI mediante ajuste exponencial a partir de mediciones en el embalse de El Val.
                    - **Chla_Bellus_cal:** Estimación cuantitativa de clorofila-a específicamente calibrada para el embalse de Bellús.
                    - **PC_Bellus_cal (Ficocianina Bellús):** Ajuste específico para el embalse de Bellús, basado en la fórmula empírica derivada de la relación espectral MCI. Se estima la concentración de ficocianina en µg/L.
                    """)

                if st.button("Calcular y mostrar resultados"):
                    # Limpiar resultados anteriores
                    st.session_state["data_time"].clear()
                    st.session_state["urls_exportacion"].clear()
                    st.session_state["used_cloud_results"].clear()
                    st.session_state["cloud_results"].clear()        
                
                    with st.spinner("Calculando fechas disponibles..."):
                        usar_csv_val = reservoir_name.lower() == "val" and int(max_cloud_percentage) == 60
                        usar_csv_bellus = reservoir_name.lower() == "bellus" and int(max_cloud_percentage) == 60
                
                        if usar_csv_val or usar_csv_bellus:
                            if usar_csv_val:
                                url_csv = "https://hibloomsbucket.s3.eu-south-2.amazonaws.com/fechas_validas_el_val_historico.csv"
                            elif usar_csv_bellus:
                                url_csv = "https://hibloomsbucket.s3.eu-south-2.amazonaws.com/fechas_validas_bellus_historico.csv"
                
                            df_fechas = cargar_fechas_csv(url_csv)
                
                            if not df_fechas.empty and "Fecha" in df_fechas.columns:
                                try:
                                    start_dt = pd.to_datetime(start_date)
                                    end_dt = pd.to_datetime(end_date)
                
                                    fechas_filtradas = df_fechas[
                                        (df_fechas["Fecha"] >= start_dt) & (df_fechas["Fecha"] <= end_dt)
                                    ]["Fecha"].dt.strftime("%Y-%m-%d").tolist()
                
                                    available_dates = sorted(fechas_filtradas)
                
                                    if available_dates:
                                        df_fechas.set_index("Fecha", inplace=True)
                
                                        cloud_results = []
                                        for f in available_dates:
                                            try:
                                                nubosidad = df_fechas.loc[f, "nubosidad"]
                                            except Exception:
                                                nubosidad = None
                
                                            cloud_results.append({
                                                "Fecha": f,
                                                "Hora": "00:00",
                                                "Nubosidad aproximada (%)": round(nubosidad, 2) if nubosidad is not None else "Desconocida",
                                                "Cobertura (%)": 100
                                            })
                
                                        st.session_state["cloud_results"] = cloud_results
                                    else:
                                        available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)
                                except Exception:
                                    available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)
                            else:
                                available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)
                        else:
                            available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)

                    

                        if not available_dates:
                            st.warning("⚠️ No se han encontrado imágenes dentro del rango de fechas y porcentaje de nubosidad seleccionados.")
                            st.session_state["data_time"] = []
                            st.stop()

                        if available_dates:
                            st.session_state['available_dates'] = available_dates
                            st.session_state['selected_indices'] = selected_indices

                            st.subheader("📅 Fechas disponibles dentro del rango seleccionado:")

                            df_available = pd.DataFrame(available_dates, columns=["Fecha"])
                            df_available["Fecha"] = pd.to_datetime(df_available["Fecha"])
                            df_available["Fecha_str"] = df_available["Fecha"].dt.strftime("%d-%b")
                            
                            # Línea base ficticia para separar los ticks visualmente
                            df_available["y_base"] = 0
                            
                            timeline_chart = alt.Chart(df_available).mark_tick(thickness=2, size=20).encode(
                                x=alt.X("Fecha:T", title=None, axis=alt.Axis(labelAngle=0, format="%d-%b")),
                                y=alt.Y("y_base:Q", axis=None),
                                tooltip=alt.Tooltip("Fecha:T", title="Fecha")
                            ).properties(
                                height=100,
                                width="container"
                            )
                            
                            # Etiquetas más arriba
                            text_layer = alt.Chart(df_available).mark_text(
                                align="center",
                                baseline="bottom",
                                dy=-15,  # Más separación vertical
                                fontSize=11
                            ).encode(
                                x="Fecha:T",
                                y=alt.value(30),  # Coloca el texto más arriba que el tick
                                text="Fecha_str:N"
                            )
                            
                            # Combina y configura
                            final_chart = (timeline_chart + text_layer).configure_axis(
                                labelFontSize=12,
                                tickSize=5
                            )
                            st.altair_chart(final_chart, use_container_width=True)
                            # Procesar y visualizar resultados
                            data_time = []

                            # Determinar si se seleccionaron índices de clorofila o de ficocianina
                            clorofila_indices = {"MCI", "NDCI_ind", "Chla_Val_cal", "Chla_Bellus_cal"}
                            ficocianina_indices = {"UV_PC_Gral_cal","PC_Val_cal", "PCI_B5/B4","PC_Bellus_cal"}
                            
                            hay_clorofila = any(indice in selected_indices for indice in clorofila_indices)
                            hay_ficocianina = any(indice in selected_indices for indice in ficocianina_indices)
                            
                            # El Val: añadir datos de SAICA solo si se han seleccionado índices de ficocianina
                            if reservoir_name.lower() == "val" and hay_ficocianina:
                                urls_csv = [
                                    "https://drive.google.com/uc?id=1-FpLJpudQd69r9JxTbT1EhHG2swASEn-&export=download",
                                    "https://drive.google.com/uc?id=1w5vvpt1TnKf_FN8HaM9ZVi3WSf0ibxlV&export=download"
                                ]
                                df_list = [cargar_csv_desde_url(url) for url in urls_csv]
                                df_list = [df for df in df_list if not df.empty]
                            
                                if df_list:
                                    df_fico = pd.concat(df_list).sort_values('Fecha-hora')
                                    start_dt = pd.to_datetime(start_date)
                                    end_dt = pd.to_datetime(end_date)
                                    df_filtrado = df_fico[(df_fico['Fecha-hora'] >= start_dt) & (df_fico['Fecha-hora'] <= end_dt)]
                            
                                    for _, row in df_filtrado.iterrows():
                                        data_time.append({
                                            "Point": "SAICA_Val",
                                            "Date": row["Fecha-hora"],
                                            "Ficocianina (µg/L)": row["Ficocianina (µg/L)"],
                                            "Tipo": "Valor Real"
                                        })
                            
                 
                            # Guardar data_time solo después de añadir (o no) los datos SAICA
                            st.session_state['data_time'] = data_time

                            # Paleta de colores para SCL con una mejor diferenciación
                            scl_palette = {
                                1: '#ff0004', 2: '#000000', 3: '#8B4513', 4: '#00FF00',
                                5: '#FFD700', 6: '#0000FF', 7: '#F4EEEC', 8: '#C8C2C0',
                                9: '#706C6B', 10: '#87CEFA', 11: '#00FFFF'
                            }
                            scl_colors = [scl_palette[i] for i in sorted(scl_palette.keys())]

                            # Antes de la iteración, limpia las listas de imágenes y fechas
                            if "image_list" in st.session_state:
                                st.session_state["image_list"] = []
                            if "selected_dates" in st.session_state:
                                st.session_state["selected_dates"] = []
                            
                            # Proceso de las fechas
                            for day in available_dates:
                                # Procesar la imagen para cada fecha
                                scaled_image, indices_image, image_date = process_sentinel2(aoi, day, max_cloud_percentage, selected_indices)
                                if indices_image is not None:
                                    # Añadir solo las imágenes y fechas necesarias (una vez por fecha)
                                    st.session_state["image_list"].append(indices_image)
                                    st.session_state["selected_dates"].append(day)
                                    
                                    # Para cada índice, calcular la distribución por clases
                                    for index_name in selected_indices:
                                        # Verifica que los valores de min y max son adecuados
                                        min_val, max_val = -0.1, 0.4  # valores por defecto
                                        if index_name == "PC_Val_cal":
                                            min_val, max_val = 0, 7
                                        elif index_name == "Chla_Val_cal":
                                            min_val, max_val = 0, 150
                                        elif index_name == "Chla_Bellus_cal":
                                            min_val, max_val = 5, 100
                                        elif index_name == "PC_Bellus_cal":
                                            min_val, max_val = 25, 500
                                        elif index_name == "PCI_B5/B4":
                                            min_val, max_val = 0.5, 1.5
                                        elif index_name == "UV_PC_Gral_cal":
                                            min_val, max_val = 0, 100
                            
                                        bins = np.linspace(min_val, max_val, 6)  # Usamos 6 bins de forma estándar
                            
                                        # Llamar a la función para calcular la distribución por clases
                                        result = calcular_distribucion_area_por_clases(indices_image, index_name, aoi, bins)
                                if indices_image is not None:
                                    url = generar_url_geotiff_multibanda(indices_image, selected_indices, aoi)
                                
                                    if "urls_exportacion" not in st.session_state:
                                        st.session_state["urls_exportacion"] = []
                                
                                    if url:
                                        st.session_state["urls_exportacion"].append({
                                            "fecha": day,
                                            "url": url
                                        })

                                if indices_image is None:
                                    continue

                                if reservoir_name in puntos_interes and puntos_interes[reservoir_name]:
                                    for point_name, (lat_point, lon_point) in puntos_interes[reservoir_name].items():
                                        values = get_values_at_point(lat_point, lon_point, indices_image, selected_indices)
                                        registro = {"Point": point_name, "Date": day, "Tipo": "Valor Estimado"}
                                    
                                        if hay_clorofila:
                                            for indice in clorofila_indices:
                                                if indice in values and values[indice] is not None:
                                                    registro[indice] = values[indice]
                                        if hay_ficocianina:
                                            for indice in ficocianina_indices:
                                                if indice in values and values[indice] is not None:
                                                    registro[indice] = values[indice]
                                    
                                        if any(k in registro for k in clorofila_indices.union(ficocianina_indices)):
                                            data_time.append(registro)


                                # Añadir media diaria del embalse solo en píxeles con SCL == 6
                                for index in selected_indices:
                                    if (hay_clorofila and index in clorofila_indices) or (hay_ficocianina and index in ficocianina_indices):
                                        media_valor = calcular_media_diaria_embalse(indices_image, index, aoi)
                                        if media_valor is None:
                                            st.warning(f"📅 En el día {day} no se ha podido calcular la media del índice '{index}' porque el embalse estaba completamente cubierto de nubes.")
                                        if media_valor is not None:
                                            data_time.append({
                                                "Point": "Media_Embalse",
                                                "Date": day,
                                                index: media_valor,
                                                "Tipo": "Valor Estimado"
                                            })

                                index_palettes = {
                                    "MCI": ['blue', 'green', 'yellow', 'red'],
                                    "PCI_B5/B4": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"],  # PCI
                                    "NDCI_ind": ['blue', 'green', 'yellow', 'red'],
                                    "UV_PC_Gral_cal": ['#2171b5', '#c7e9c0', '#238b45', '#e31a1c'],
                                    "PC_Val_cal": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"],  # Paleta específica para PC
                                    "Chla_Val_cal": ['#2171b5', '#c7e9c0', '#238b45', '#e31a1c'],
                                    "Chla_Bellus_cal": ['#2171b5', '#c7e9c0', '#238b45', '#e31a1c'],
                                    "PC_Bellus_cal": ['#2171b5', '#c7e9c0', '#238b45', '#e31a1c']
                                }

                                with row2[0]:
                                    image_date_fmt = datetime.strptime(image_date, "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y %H:%M")
                                    with st.expander(f"📅 Mapa de Índices para {image_date_fmt}"):
                                        gdf_4326 = gdf.to_crs(epsg=4326)
                                        map_center = [gdf_4326.geometry.centroid.y.mean(),
                                                      gdf_4326.geometry.centroid.x.mean()]
                                        map_indices = geemap.Map(center=map_center, zoom=13)
                                        
                                        # Crear grupos de capas para permitir que solo una se active a la vez
                                        rgb_layer = folium.raster_layers.TileLayer(
                                            tiles=scaled_image.visualize(bands=['B4', 'B3', 'B2'], min=0, max=0.3,
                                                                         gamma=1.4).getMapId()[
                                                "tile_fetcher"].url_format,
                                            name="RGB",
                                            overlay=True,
                                            control=True,
                                            show=True,  # Mostrar esta por defecto
                                            attr="Copernicus Sentinel-2, processed by GEE"
                                        )

                                        scl_layer = folium.raster_layers.TileLayer(
                                            tiles=indices_image.select('SCL').visualize(min=1, max=11,
                                                                                        palette=scl_colors).getMapId()[
                                                "tile_fetcher"].url_format,
                                            name="SCL - Clasificación de Escena",
                                            overlay=True,
                                            control=True,
                                            show=False,
                                            attr="Copernicus Sentinel-2, processed by GEE"
                                        )

                                        cloud_layer = folium.raster_layers.TileLayer(
                                            tiles=indices_image.select('MSK_CLDPRB').visualize(min=0, max=100,
                                                                                               palette=['blue', 'green',
                                                                                                        'yellow', 'red',
                                                                                                        'black']).getMapId()[
                                                "tile_fetcher"].url_format,
                                            name="Probabilidad de Nubes (MSK_CLDPRB)",
                                            overlay=True,
                                            control=True,
                                            show=False,
                                            attr="Copernicus Sentinel-2, processed by GEE"
                                        )

                                        # Crear el grupo de puntos de interés (no activado por defecto)
                                        poi_group = folium.FeatureGroup(name="Puntos de Interés", show=False)
                                        tiene_puntos = False  # Variable de control
                                        
                                        # Añadir marcadores al grupo si existen
                                        if reservoir_name in puntos_interes:
                                            for point_name, (lat_point, lon_point) in puntos_interes[reservoir_name].items():
                                                folium.Marker(
                                                    location=[lat_point, lon_point],
                                                    popup=f"{point_name}",
                                                    tooltip=f"{point_name}",
                                                    icon=folium.Icon(color="red", icon="info-sign")
                                                ).add_to(poi_group)
                                                tiene_puntos = True  # Al menos un punto añadido
                                        

                                        # Agregar capas al mapa
                                        rgb_layer.add_to(map_indices)
                                        scl_layer.add_to(map_indices)
                                        cloud_layer.add_to(map_indices)
                                        if tiene_puntos:
                                            poi_group.add_to(map_indices)

                                        # Agregar los índices como capas opcionales
                                        for index in selected_indices:
                                            vis_params = {"min": -0.1, "max": 0.4, "palette": index_palettes.get(index,
                                                                                                                 [
                                                                                                                     'blue',
                                                                                                                     'green',
                                                                                                                     'yellow',
                                                                                                                     'red'])}
                                            if index == "PC_Val_cal":
                                                vis_params["min"] = 0
                                                vis_params["max"] = 10
                                                vis_params["palette"] = ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]
                                            elif index == "PCI_B5/B4":
                                                vis_params["min"] = 0.5
                                                vis_params["max"] = 1.5
                                                vis_params["palette"] = ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]
                                            elif index == "Chla_Val_cal":
                                                vis_params["min"] = 0
                                                vis_params["max"] = 150
                                                vis_params["palette"] = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']
                                            elif index == "Chla_Bellus_cal":
                                                vis_params["min"] = 5
                                                vis_params["max"] = 100
                                                vis_params["palette"] = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']
                                            elif index == "PC_Bellus_cal":
                                                vis_params["min"] = 25
                                                vis_params["max"] = 500
                                                vis_params["palette"] = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']
                                            elif index == "UV_PC_Gral_cal":
                                                vis_params["min"] = 0
                                                vis_params["max"] = 100
                                                vis_params["palette"] = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']


                                            index_layer = folium.raster_layers.TileLayer(
                                                tiles=indices_image.select(index).visualize(**vis_params).getMapId()[
                                                    "tile_fetcher"].url_format,
                                                name=index,
                                                overlay=True,
                                                control=True,
                                                show=False,
                                                attr="Copernicus Sentinel-2, processed by GEE"
                                            )
                                            index_layer.add_to(map_indices)

                                        # Agregar el control de capas con opción exclusiva
                                        folium.LayerControl(collapsed=False, position='topright').add_to(map_indices)

                                        # Mostrar el mapa en Streamlit
                                        folium_static(map_indices)

                            st.session_state['data_time'] = data_time

                        df_time = pd.DataFrame(data_time)

                        with row2[1]:
                            # Leyenda de índices y capas
                            with st.expander("🗺️ Leyenda de índices y capas", expanded=False):
                                generar_leyenda(selected_indices)
                        
                            # Tabla de nubosidad estimada por imagen
                            if "used_cloud_results" in st.session_state and st.session_state["used_cloud_results"]:
                                with st.expander("☁️ Nubosidad estimada por imagen", expanded=False):
                                    df_results = pd.DataFrame(st.session_state["used_cloud_results"])
                                    df_results["Fecha"] = pd.to_datetime(df_results["Fecha"], errors='coerce').dt.strftime("%d-%m-%Y")
                                    st.dataframe(df_results)

                            with st.expander("📊 Evolución de la media diaria de concentraciones del embalse", expanded=False):
                                df_media = df_time[df_time["Point"] == "Media_Embalse"].copy()
                                df_media["Date"] = pd.to_datetime(df_media["Date"], errors='coerce')
                            
                                for indice in selected_indices:
                                    if indice in df_media.columns:
                                        df_indice = df_media[["Date", indice]].dropna()
                            
                                        # Determinar si es un índice calibrado según si contiene 'cal' (insensible a mayúsculas)
                                        if 'cal' in indice.lower():
                                            y_axis_title = 'Concentración (μg/L)'
                                        else:
                                            y_axis_title = 'Valor del índice'
                            
                                        chart = alt.Chart(df_indice).mark_bar().encode(
                                            x=alt.X('Date:T', title='Fecha', axis=alt.Axis(format="%d-%b", labelAngle=0)),
                                            y=alt.Y(f'{indice}:Q', title=y_axis_title),
                                            tooltip=[
                                                alt.Tooltip('Date:T', title='Fecha'),
                                                alt.Tooltip(f'{indice}:Q', title=f'{indice}')
                                            ]
                                        ).properties(
                                            title=f"🧪 Índice: {indice}",
                                            width=500,
                                            height=300
                                        )
                            
                                        st.altair_chart(chart, use_container_width=True)


                            # Dentro de tu código de interfaz para visualizar las distribuciones
                            if "image_list" in st.session_state and "selected_dates" in st.session_state:
                                # Un único expander para toda la sección de distribución
                                with st.expander("📊 Distribución diaria por clases del índice en el embalse", expanded=False):
                            
                                    # Inicializar la lista de datos
                                    data = []  # Guardamos los datos para todas las fechas
                            
                                    # Recorremos las imágenes y las fechas almacenadas en session_state
                                    for i, (img, fecha_str) in enumerate(zip(st.session_state["image_list"], st.session_state["selected_dates"])):
                                        fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()
                            
                                        for index_name in st.session_state["selected_indices"]:
                                            # Obtener min/max según vis_params (utilizando la paleta exacta para cada índice)
                                            min_val, max_val, palette = -0.1, 0.4, ['blue', 'green', 'yellow', 'red'] 
                            
                                            if index_name == "PC_Val_cal":
                                                min_val, max_val = 0, 25
                                                palette = ["#ADD8E6", "#008000", "#FFFF00", "#FF8000", "#FF0000"]  # Azul, verde, amarillo,naranja, rojo
                                            elif index_name == "Chla_Val_cal":
                                                min_val, max_val = 0, 150
                                                palette = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']  # Azul, verde, amarillo, rojo
                                            elif index_name == "Chla_Bellus_cal":
                                                min_val, max_val = 5, 100
                                                palette = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']  # Azul, verde, amarillo, rojo
                                            elif index_name == "PC_Bellus_cal":
                                                min_val, max_val = 25, 500
                                                palette = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']  # Azul, verde, amarillo, rojo
                                            elif index_name == "UV_PC_Gral_cal":
                                                min_val, max_val = 0, 100
                                                palette = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']  # Azul, verde, amarillo, rojo
                            
                                            if index_name == "PC_Val_cal":
                                                bins = np.linspace(min_val, max_val, 6)
                                            else: 
                                                bins = np.linspace(min_val, max_val, 5)# 4 categorías, por lo tanto, 5 puntos
                                            # Llamar a la función para calcular la distribución por clases
                                            result = calcular_distribucion_area_por_clases(img, index_name, aoi, bins)
                            
                                            if result:
                                                # Convertir el resultado en un DataFrame para graficarlo
                                                df_distribution = pd.DataFrame(result)
                            
                                                # Añadir los resultados al dataframe final, incluyendo la fecha para cada entrada
                                                for row in df_distribution.itertuples():
                                                    data.append({
                                                        "Fecha": fecha,
                                                        "Rango": row.rango,
                                                        "Porcentaje": row.porcentaje
                                                    })
                            
                                    # Si se ha recogido algún dato, generar el gráfico
                                    if data:
                                        df_final = pd.DataFrame(data)
                            
                                        # Invertir el orden de las categorías en la barra (se apilarán de abajo hacia arriba)
                                        df_final['Rango'] = pd.Categorical(df_final['Rango'], categories=df_final['Rango'].unique(), ordered=True)
                                        df_final['Rango'] = df_final['Rango'].cat.reorder_categories(df_final['Rango'].cat.categories[::-1])  # Invertir el orden
                            
                                        # Graficar la distribución como un gráfico de barras apiladas
                                        chart = alt.Chart(df_final).mark_bar(size=25).encode(  # Aumentar el tamaño de las barras
                                            x=alt.X('Fecha:T', title='Fecha'),
                                            y=alt.Y('Porcentaje:Q', title='Porcentaje de área (%)', stack='zero'),
                                            color=alt.Color('Rango:N', scale=alt.Scale(domain=df_final['Rango'].unique().tolist(), range=palette), legend=alt.Legend(title="Rango de valores"))
                                        ).properties(
                                            title=f"Distribución de las clases del índice por fecha",
                                            width=800,
                                            height=400
                                        )
                            
                                        # Mostrar el gráfico
                                        st.altair_chart(chart, use_container_width=True)
                                        
                            # Serie temporal real de ficocianina (solo si embalse es VAL)
                            if reservoir_name.lower() == "val" and "PC_Val_cal" in selected_indices:
                                with st.expander("📈 Serie temporal real de ficocianina (sonda SAICA)", expanded=False):
                                    urls_csv = [
                                        "https://drive.google.com/uc?id=1-FpLJpudQd69r9JxTbT1EhHG2swASEn-&export=download",
                                        "https://drive.google.com/uc?id=1w5vvpt1TnKf_FN8HaM9ZVi3WSf0ibxlV&export=download"
                                    ]
                                    df_list = [cargar_csv_desde_url(url) for url in urls_csv]
                                    df_list = [df for df in df_list if not df.empty]
                        
                                    if df_list:
                                        df_fico = pd.concat(df_list).sort_values('Fecha-hora')
                                        start_dt = pd.to_datetime(start_date)
                                        end_dt = pd.to_datetime(end_date)
                                        df_filtrado = df_fico[(df_fico['Fecha-hora'] >= start_dt) & (df_fico['Fecha-hora'] <= end_dt)]
                        
                                        if df_filtrado.empty:
                                            st.warning("⚠️ No hay datos de ficocianina en el rango de fechas seleccionado.")
                                        else:
                                            max_puntos_grafico = 500
                                            step = max(1, len(df_filtrado) // max_puntos_grafico)
                                            df_subsample = df_filtrado.iloc[::step]
                                            df_subsample["Fecha_formateada"] = df_subsample["Fecha-hora"].dt.strftime("%d-%m-%Y %H:%M")
                        
                                            chart_fico = alt.Chart(df_subsample).mark_line().encode(
                                                x=alt.X('Fecha_formateada:N', title='Fecha y hora', axis=alt.Axis(labelAngle=45)),
                                                y=alt.Y('Ficocianina (µg/L):Q', title='Concentración (µg/L)'),
                                                tooltip=[
                                                    alt.Tooltip('Fecha_formateada:N', title='Fecha y hora'),
                                                    alt.Tooltip('Ficocianina (µg/L):Q', title='Ficocianina (µg/L)', format=".2f")
                                                ]
                                            ).properties(
                                                title="Evolución de la concentración de ficocianina (µg/L)"
                                            )
                        
                                            st.altair_chart(chart_fico, use_container_width=True)
                                    else:
                                        st.warning("⚠️ No se pudo cargar ningún archivo de ficocianina.")                       

                            if reservoir_name.lower() == "val" and hay_clorofila:
                                with st.expander("📈 Valores reales de clorofila-a (valores de sonda Aquadam en 41.8761,-1.7883)", expanded=False):
                                    url_cloro_val = "https://hibloomsbucket.s3.eu-south-2.amazonaws.com/clorofila_val_entero.csv"
                                    try:
                                        df_cloro = pd.read_csv(url_cloro_val)
                            
                                        # Asegurar que están las columnas necesarias
                                        if "Fecha" in df_cloro.columns and "Clorofila (µg/L)" in df_cloro.columns:
                                            df_cloro["Fecha-hora"] = pd.to_datetime(df_cloro["Fecha"])
                                            df_cloro = df_cloro[["Fecha-hora", "Clorofila (µg/L)"]]
                            
                                            start_dt = pd.to_datetime(start_date)
                                            end_dt = pd.to_datetime(end_date)
                            
                                            df_filtrado = df_cloro[(df_cloro["Fecha-hora"] >= start_dt) & (df_cloro["Fecha-hora"] <= end_dt)]
                            
                                            if df_filtrado.empty:
                                                st.warning("⚠️ No hay datos de clorofila en el rango de fechas seleccionado.")
                                            else:
                                                max_puntos_grafico = 500
                                                step = max(1, len(df_filtrado) // max_puntos_grafico)
                                                df_subsample = df_filtrado.iloc[::step]
                                                df_subsample["Fecha_formateada"] = df_subsample["Fecha-hora"].dt.strftime("%d-%m-%Y %H:%M")
                            
                                                chart_cloro = alt.Chart(df_subsample).mark_line().encode(
                                                    x=alt.X('Fecha_formateada:N', title='Fecha y hora', axis=alt.Axis(labelAngle=45)),
                                                    y=alt.Y('Clorofila (µg/L):Q', title='Concentración (µg/L)'),
                                                    tooltip=[
                                                        alt.Tooltip('Fecha_formateada:N', title='Fecha y hora'),
                                                        alt.Tooltip('Clorofila (µg/L):Q', title='Clorofila (µg/L)', format=".2f")
                                                    ]
                                                ).properties(
                                                    title="Evolución de la concentración de clorofila (µg/L)"
                                                )
                            
                                                st.altair_chart(chart_cloro, use_container_width=True)
                                        else:
                                            st.error("❌ El archivo no contiene las columnas necesarias: 'Fecha' y 'Clorofila (µg/L)'")
                                    except Exception as e:
                                        st.error(f"❌ Error al cargar el archivo de clorofila desde S3: {e}")

                        
                            if not df_time.empty:
                                with st.expander("📉 Gráficos de valores por punto de interés", expanded=False):
                                    df_time["Fecha_dt"] = pd.to_datetime(df_time["Date"], errors='coerce')
                                
                                    for point in df_time["Point"].unique():
                                        if point != "Media_Embalse":
                                            df_point = df_time[df_time["Point"] == point]
                                            df_melted = df_point.melt(id_vars=["Point", "Fecha_dt"],
                                                                      value_vars=selected_indices,
                                                                      var_name="Índice", value_name="Valor")
                                
                                            chart = alt.Chart(df_melted).mark_line(point=True).encode(
                                                x=alt.X('Fecha_dt:T', title='Fecha y hora',
                                                        axis=alt.Axis(format="%d-%b %H:%M", labelAngle=45)),
                                                y=alt.Y('Valor:Q', title='Concentración (µg/L)'),
                                                color=alt.Color('Índice:N', title='Índice'),
                                                tooltip=[
                                                    alt.Tooltip('Fecha_dt:T', title='Fecha y hora', format="%d-%m-%Y %H:%M"),
                                                    'Índice:N', 'Valor:Q'
                                                ]
                                            ).properties(
                                                title=f"Valores de índices en {point}"
                                            )
                                
                                            st.altair_chart(chart, use_container_width=True)

                        with tab3:
                            st.subheader("Tablas de Índices Calculados")
                        
                            if not df_time.empty:
                                df_time = df_time.copy()
                        
                                # Renombrar la columna 'Point' a 'Ubicación'
                                df_time.rename(columns={"Point": "Ubicación"}, inplace=True)
                        
                                # Crear una única columna 'Fecha' en formato datetime para ordenar
                                if "Fecha" not in df_time.columns:
                                    posibles_fechas = ["Date", "Fecha-hora", "Fecha_dt"]
                                    for col in posibles_fechas:
                                        if col in df_time.columns:
                                            df_time["Fecha"] = pd.to_datetime(df_time[col], errors='coerce')
                                            break
                        
                                # Verificar que 'Fecha' existe y eliminar duplicados
                                if "Fecha" not in df_time.columns:
                                    st.error("❌ No se encontró ninguna columna de fecha válida.")
                                    st.stop()
                        
                                # Ordenar por 'Ubicación' y 'Fecha' (orden cronológico)
                                df_time = df_time.dropna(subset=["Fecha"]).sort_values(by=["Ubicación", "Fecha"])
                        
                                # Convertir la fecha a texto para visualización
                                df_time["Fecha"] = df_time["Fecha"].dt.strftime("%d-%m-%Y %H:%M")
                        
                                # Eliminar columnas de fecha duplicadas si existen
                                columnas_fecha = ["Date", "Fecha-hora", "Fecha_dt"]
                                df_time.drop(columns=[col for col in columnas_fecha if col in df_time.columns], errors='ignore', inplace=True)
                        
                                # Ordenar las columnas
                                columnas = list(df_time.columns)
                                orden = ["Ubicación", "Fecha", "Tipo"]
                                otras = [col for col in columnas if col not in orden]
                                columnas_ordenadas = orden + otras
                                df_time = df_time[columnas_ordenadas]
                        
                                # Dividir en puntos de interés y medias del embalse
                                df_medias = df_time[df_time["Ubicación"] == "Media_Embalse"]
                                df_puntos = df_time[df_time["Ubicación"] != "Media_Embalse"]
                        
                                # Mostrar las tablas corregidas
                                if not df_puntos.empty:
                                    st.markdown("### 📌 Datos en los puntos de interés")
                                    st.dataframe(df_puntos.reset_index(drop=True))
                        
                                if not df_medias.empty:
                                    st.markdown("### 💧 Datos de medias del embalse")
                                    st.dataframe(df_medias.reset_index(drop=True))
                            else:
                                st.warning("No hay datos disponibles. Primero realiza el cálculo en la pestaña de Visualización.")                                                   
                                                            
with tab4:
                            st.subheader("📈 Modo rápido: generación de gráficas")
                        
                            st.info("Este modo solo genera gráficas a partir de los parámetros seleccionados, sin mapas ni exportaciones.")
                        
                            # Selección de embalse
                            nombres_embalses = obtener_nombres_embalses()
                            reservoir_name = st.selectbox("Selecciona un embalse:", nombres_embalses, key="graficas_embalse")
                        
                            if reservoir_name:
                                gdf = load_reservoir_shapefile(reservoir_name)
                                if gdf is not None:
                                    aoi = gdf_to_ee_geometry(gdf)
                        
                                    max_cloud_percentage = st.slider("Porcentaje máximo de nubosidad permitido:", 0, 100, 10, key="graficas_nubosidad")
                        
                                    date_range = st.date_input(
                                        "Selecciona el rango de fechas:",
                                        value=(datetime.today() - timedelta(days=15), datetime.today()),
                                        min_value=datetime(2017, 7, 1),
                                        max_value=datetime.today(),
                                        key="graficas_fecha"
                                    )
                        
                                    if isinstance(date_range, tuple) and len(date_range) == 2:
                                        start_date, end_date = date_range
                                    else:
                                        start_date, end_date = datetime(2017, 7, 1), datetime.today()
                        
                                    start_date = start_date.strftime('%Y-%m-%d')
                                    end_date = end_date.strftime('%Y-%m-%d')
                        
                                    available_indices = ["MCI", "PCI_B5/B4", "NDCI_ind","UV_PC_Gral_cal", "PC_Val_cal", "Chla_Val_cal", "Chla_Bellus_cal", "PC_Bellus_cal"]
                                    selected_indices = st.multiselect("Selecciona los índices a visualizar:", available_indices, key="graficas_indices")
                        
                                    if st.button("Ejecutar modo rápido"):
                                        st.session_state["data_time"] = []
                                        # Mapeo de nombres para los CSV precalculados
                                        csv_name_map = {
                                                "El Val": "val",
                                                "Bellús": "bellus"
                                        }
                                        reservoir_key = csv_name_map.get(reservoir_name, None)
                                            
                                        # Usar CSV precalculado si procede
                                        if reservoir_key in ["val", "bellus"] and max_cloud_percentage == 60:
                                                url_csv = f"https://{BUCKET_NAME}.s3.amazonaws.com/fechas_validas/{reservoir_key}_60.csv"
                                                try:
                                                    df_csv = pd.read_csv(url_csv)
                                                    available_dates = pd.to_datetime(df_csv["fechas"])
                                                    available_dates = available_dates[
                                                        (available_dates >= pd.to_datetime(start_date)) &
                                                        (available_dates <= pd.to_datetime(end_date))
                                                    ]
                                                except Exception as e:
                                                    st.warning(f"No se pudo cargar el CSV precalculado para {reservoir_name}: {e}")
                                                    available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)
                                        else:
                                                available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)

                                        if not available_dates:
                                                st.warning("No se encontraron imágenes en ese rango de fechas.")
                                                st.stop()
                        
                                        data_time = []
                                        clorofila_indices = {"MCI", "NDCI_ind", "Chla_Val_cal", "Chla_Bellus_cal"}
                                        ficocianina_indices = {"UV_PC_Gral_cal""PC_Val_cal", "PCI_B5/B4","PC_Bellus_cal"}
                        
                                        hay_clorofila = any(i in selected_indices for i in clorofila_indices)
                                        hay_ficocianina = any(i in selected_indices for i in ficocianina_indices)
                        
                                        if reservoir_name.lower() == "val" and hay_ficocianina:
                                            urls = [
                                                "https://drive.google.com/uc?id=1-FpLJpudQd69r9JxTbT1EhHG2swASEn-&export=download",
                                                "https://drive.google.com/uc?id=1w5vvpt1TnKf_FN8HaM9ZVi3WSf0ibxlV&export=download"
                                            ]
                                            df_list = [cargar_csv_desde_url(u) for u in urls]
                                            df_list = [df for df in df_list if not df.empty]
                                            if df_list:
                                                df_fico = pd.concat(df_list).sort_values('Fecha-hora')
                                                start_dt = pd.to_datetime(start_date)
                                                end_dt = pd.to_datetime(end_date)
                                                df_filtrado = df_fico[(df_fico['Fecha-hora'] >= start_dt) & (df_fico['Fecha-hora'] <= end_dt)]
                                                for _, row in df_filtrado.iterrows():
                                                    data_time.append({
                                                        "Point": "SAICA_Val",
                                                        "Date": row["Fecha-hora"],
                                                        "Ficocianina (µg/L)": row["Ficocianina (µg/L)"],
                                                        "Tipo": "Valor Real"
                                                    })
                        
                                        if reservoir_name.lower() == "bellus" and (hay_clorofila or hay_ficocianina):
                                            url_fico = "https://drive.google.com/uc?id=1jeTpJfPTTKORN3iIprh6P_RPXPu16uDa&export=download"
                                            url_cloro = "https://drive.google.com/uc?id=17-jtO6mbjfj_CMnsMo_UX2RQ7IM_0hQ4&export=download"
                                            df_fico = cargar_csv_desde_url(url_fico)
                                            df_cloro = cargar_csv_desde_url(url_cloro)
                        
                                            for col in df_fico.columns:
                                                if "pc_ivf" in col.lower():
                                                    df_fico.rename(columns={col: "Ficocianina (µg/L)"}, inplace=True)
                                            for col in df_cloro.columns:
                                                if "chla_ivf" in col.lower():
                                                    df_cloro.rename(columns={col: "Clorofila (µg/L)"}, inplace=True)
                        
                                            if not df_fico.empty and not df_cloro.empty:
                                                df_bellus = pd.merge(df_fico, df_cloro, on="Fecha-hora", how="outer")
                                                df_bellus = df_bellus.sort_values("Fecha-hora")
                                                start_dt = pd.to_datetime(start_date)
                                                end_dt = pd.to_datetime(end_date)
                                                df_bellus_filtrado = df_bellus[(df_bellus["Fecha-hora"] >= start_dt) & (df_bellus["Fecha-hora"] <= end_dt)]
                                                for _, row in df_bellus_filtrado.iterrows():
                                                    entry = {"Point": "Sonda-Bellús", "Date": row["Fecha-hora"], "Tipo": "Real"}
                                                    if hay_ficocianina and pd.notna(row.get("Ficocianina (µg/L)")):
                                                        entry["Ficocianina (µg/L)"] = row["Ficocianina (µg/L)"]
                                                    if hay_clorofila and pd.notna(row.get("Clorofila (µg/L)")):
                                                        entry["Clorofila (µg/L)"] = row["Clorofila (µg/L)"]
                                                    if "Ficocianina (µg/L)" in entry or "Clorofila (µg/L)" in entry:
                                                        data_time.append(entry)
                        
                                        for day in available_dates:
                                            _, indices_image, _ = process_sentinel2(aoi, day, max_cloud_percentage, selected_indices)
                                            if indices_image is None:
                                                continue

                                            if reservoir_name in puntos_interes and puntos_interes[reservoir_name]: 
                                                for point_name, (lat, lon) in puntos_interes[reservoir_name].items():
                                                    values = get_values_at_point(lat, lon, indices_image, selected_indices)
                                                    registro = {"Point": point_name, "Date": day, "Tipo": "Valor Estimado"}
                                                    for i in selected_indices:
                                                        if i in values and values[i] is not None:
                                                            registro[i] = values[i]
                                                    if any(i in registro for i in selected_indices):
                                                        data_time.append(registro)
                            
                                                for i in selected_indices:
                                                    media_valor = calcular_media_diaria_embalse(indices_image, i, aoi)
                                                    if media_valor is not None:
                                                        data_time.append({
                                                            "Point": "Media_Embalse",
                                                            "Date": day,
                                                            i: media_valor,
                                                            "Tipo": "Valor Estimado"
                                                        })
                                        df_time = pd.DataFrame(data_time)
                                        if df_time.empty:
                                            st.warning("No se generaron datos válidos.")
                                            st.stop()
                        
                                        st.session_state["data_time"] = data_time
                                        st.success("✅ Datos procesados correctamente. Mostrando gráficas:")
                                        df_time["Fecha_dt"] = pd.to_datetime(df_time["Date"], errors='coerce')
                        
                                        with st.expander("📊 Evolución de la media diaria del embalse", expanded=True):
                                            df_media = df_time[df_time["Point"] == "Media_Embalse"]
                                            for i in selected_indices:
                                                if i in df_media.columns:
                                                    df_ind = df_media[["Fecha_dt", i]].dropna()
                                                    chart = alt.Chart(df_ind).mark_bar().encode(
                                                        x=alt.X("Fecha_dt:T", title="Fecha"),
                                                        y=alt.Y(f"{i}:Q", title="Concentración"),
                                                        tooltip=["Fecha_dt", i]
                                                    ).properties(title=f"{i} – Media embalse")
                                                    st.altair_chart(chart, use_container_width=True)
                        
                                        with st.expander("📍 Valores por punto de interés", expanded=True):
                                            for point in df_time["Point"].unique():
                                                if point != "Media_Embalse":
                                                    df_p = df_time[df_time["Point"] == point]
                                                    df_melt = df_p.melt(id_vars=["Point", "Fecha_dt"],
                                                                        value_vars=selected_indices,
                                                                        var_name="Índice", value_name="Valor")
                                                    chart = alt.Chart(df_melt).mark_line(point=True).encode(
                                                        x=alt.X("Fecha_dt:T", title="Fecha"),
                                                        y=alt.Y("Valor:Q", title="Valor"),
                                                        color="Índice:N",
                                                        tooltip=["Fecha_dt", "Índice", "Valor"]
                                                    ).properties(title=f"{point} – evolución de índices")
                                                    st.altair_chart(chart, use_container_width=True)
                                        # Mostrar tablas de resultados igual que en la pestaña "Tablas"
                                        st.subheader("📄 Resultados en tabla")
                                        
                                        # Copia del DataFrame y limpieza básica
                                        df_tabla = df_time.copy()
                                        df_tabla.rename(columns={"Point": "Ubicación"}, inplace=True)
                                        df_tabla["Fecha"] = pd.to_datetime(df_tabla["Date"], errors='coerce').dt.strftime("%d-%m-%Y %H:%M")
                                        df_tabla.drop(columns=["Date", "Fecha_formateada", "Fecha_dt", "Fecha-hora"], errors='ignore', inplace=True)
                                        
                                        # Agrupar valores medios si hay duplicados
                                        df_medias = df_tabla[df_tabla["Ubicación"] == "Media_Embalse"]
                                        df_otros = df_tabla[df_tabla["Ubicación"] != "Media_Embalse"]
                                        
                                        if not df_medias.empty:
                                            columnas_valor = [col for col in df_medias.columns if col not in ["Ubicación", "Fecha", "Tipo"]]
                                            df_medias = df_medias.groupby(["Ubicación", "Fecha", "Tipo"], as_index=False).agg({col: "max" for col in columnas_valor})
                                        
                                        df_tabla = pd.concat([df_medias, df_otros], ignore_index=True)
                                        
                                        # Ordenar columnas
                                        columnas = list(df_tabla.columns)
                                        orden = ["Ubicación", "Fecha", "Tipo"]
                                        otras = [col for col in columnas if col not in orden]
                                        columnas_ordenadas = orden + otras
                                        df_tabla = df_tabla[columnas_ordenadas]
                                        
                                        # Separar y mostrar
                                        df_puntos = df_tabla[df_tabla["Ubicación"] != "Media_Embalse"]
                                        df_medias = df_tabla[df_tabla["Ubicación"] == "Media_Embalse"]
                                        
                                        if not df_puntos.empty:
                                            st.markdown("### 📌 Datos en los puntos de interés")
                                            st.dataframe(df_puntos.reset_index(drop=True))
                                        
                                        if not df_medias.empty:
                                            st.markdown("### 💧 Datos de medias del embalse")
                                            st.dataframe(df_medias.reset_index(drop=True))



























