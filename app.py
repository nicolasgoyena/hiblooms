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

puntos_interes = {
    "EUGUI": {
        "EU-1": (42.972554580588245, -1.5153927772290117),
        "EU-2": (42.979363533124015, -1.5133104333288598),
        "EU-3": (42.99334922635716, -1.5164716652855823)
    },
    "ALLOZ": {
        "All-1": (42.7094165624961, -1.94727650210385),
        "All-2": (42.72287216, -1.94302458),
        "All-3": (42.72821464, -1.93702794),
        "All-4": (42.73680851, -1.930444394)
    },
    "VAL": {
        "VAL-1": (41.87977480764112, -1.8052571871777034),
        "VAL-2": (41.8765336, -1.792419161),
        "Sonda-SAICA": (41.8761, -1.7883)
    },
    "ITOIZ": {
        "IT-1": (42.81359078720846, -1.3675024704573298),
        "IT-2": (42.8133203, -1.365784146),
        "IT-3": (42.80699069, -1.36347659)
    },
    "GONZALEZ LACASA": {
        "G-1": (42.18173801436588, -2.6810543962707656),
        "G-2": (42.17967448, -2.687254697),
        "G-3": (42.1855258, -2.681117803)
    },
    "URRUNAGA": {
        "Urr-1": (42.960690852708, -2.6501586877888),
        "Urr-2": (42.983643076227516, -2.6477366856877382),
        "Urr-3": (42.980077967705235, -2.674754078553591)
    },
    "BELLUS": {
    "Sonda-Bellús": (38.936974, -0.479160) 
}
}

def reproject_coords_to_epsg(coords, target_crs='EPSG:32630'):
    reprojected_coords = {}
    for place, points in coords.items():
        if place in puntos_interes:  # Solo reproyectar si el embalse tiene puntos de interés
            reprojected_coords[place] = {}
            for point_id, (lat, lon) in points.items():
                point = ee.Geometry.Point([lon, lat])
                reprojected_point = point.transform(target_crs)
                reprojected_coords[place][point_id] = reprojected_point.coordinates().getInfo()
    return reprojected_coords

# Reproyectar las coordenadas
reprojected_puntos_interes = reproject_coords_to_epsg(puntos_interes)

@st.cache_data
def cargar_csv_desde_url(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)

        # Renombrar si la columna de fecha es 'Time'
        if 'Time' in df.columns:
            df.rename(columns={'Time': 'Fecha-hora'}, inplace=True)

        # Parsear fecha sin dayfirst para evitar errores con formato ISO
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

    if "cloud_results" not in st.session_state:
        st.session_state["cloud_results"] = []

    sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date)

    if sentinel2.size().getInfo() == 0:
        st.warning("❌ No se encontraron imágenes de Sentinel-2 para este embalse y rango de fechas.")
        return []

    images = sentinel2.toList(sentinel2.size())
    available_dates = set()
    results_list = []

    for i in range(images.size().getInfo()):
        inicio_iter = time.time()

        image = ee.Image(images.get(i)).clip(aoi)
        image_date = image.get('system:time_start').getInfo()
        formatted_date = datetime.utcfromtimestamp(image_date / 1000).strftime('%Y-%m-%d')
        image_time = datetime.utcfromtimestamp(image_date / 1000).strftime('%H:%M')

        # Evitar duplicados
        if formatted_date in available_dates:
            continue

        with st.spinner(f"**🕒 Analizando imagen del {formatted_date}...**"):
            cloud_obj = calculate_cloud_percentage(image, aoi)
            if cloud_obj is None:
                print(f"⚠️ Imagen del {formatted_date} descartada: no tiene SCL ni MSK_CLDPRB.")
                continue
            
            try:
                cloud_percentage = cloud_obj.getInfo()
            except Exception as e:
                print(f"⚠️ Error al obtener cloud_percentage en {formatted_date}: {e}")
                continue
            
            coverage = calculate_coverage_percentage(image, aoi)
            


            # Solo conservar fechas que pasen los filtros
            if (max_cloud_percentage == 100 or cloud_percentage <= max_cloud_percentage) and coverage >= 50:
                available_dates.add(formatted_date)
                results_list.append({
                    "Fecha": formatted_date,
                    "Hora": image_time,
                    "Nubosidad aproximada (%)": round(cloud_percentage, 2),
                    "Cobertura (%)": round(coverage, 2)
                })

        fin_iter = time.time()
        print(f"Tiempo en procesar imagen {formatted_date}: {fin_iter - inicio_iter:.2f} seg")

    fin_total = time.time()
    print(f"Tiempo total en get_available_dates: {fin_total - inicio_total:.2f} seg")

    st.session_state["cloud_results"] = results_list

    return sorted(available_dates)

def load_reservoir_shapefile(reservoir_name, shapefile_path="shapefiles/embalses_hiblooms.shp"):
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)

        # Verificar existencia del campo 'NOMBRE'
        if "NOMBRE" not in gdf.columns:
            st.error("❌ El shapefile cargado no contiene una columna llamada 'NOMBRE'. Añádela para poder seleccionar embalses.")
            return None

        # Reproyectar automáticamente si no está en EPSG:32630
        if gdf.crs is None or gdf.crs.to_epsg() != 32630:
            st.warning("🔄 El shapefile no está en EPSG:32630. Se reproyectará automáticamente.")
            gdf = gdf.to_crs(epsg=32630)

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
        
    if gdf.crs is None or gdf.crs.to_epsg() != 32630:
        raise ValueError("❌ El shapefile debe estar en EPSG:32630.")
        
    geometry = gdf.geometry.iloc[0]
    
    if geometry.geom_type == "MultiPolygon":
        geometry = list(geometry.geoms)[0]  # Extrae el primer polígono
        
    ee_coordinates = list(geometry.exterior.coords)
    ee_geometry = ee.Geometry.Polygon(
        ee_coordinates,
        proj=ee.Projection("EPSG:32630"),  # Especifica la proyección UTM
        geodesic=False # Evita errores con geometrías con huecos
    )

    return ee_geometry

def calcular_media_diaria_embalse(indices_image, index_name, aoi):
    """Calcula la media del índice dado sobre el embalse solo en píxeles de agua (SCL == 6 o SCL == 2 para el año 2018)."""
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
    cloud_mask_scl = scl.eq(7).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))

    cloud_fraction_scl = cloud_mask_scl.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e13
    ).get('SCL')

    cloud_mask_prob = image.select('MSK_CLDPRB').gte(10)
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
        best_score = None  # Combinación de baja nubosidad y alta cobertura

        for i in range(num_images):
            image = ee.Image(images.get(i))
            try:
                cloud_score = calculate_cloud_percentage(image, aoi).getInfo()
                coverage = calculate_coverage_percentage(image, aoi)
                
                if coverage < 50 or cloud_score > max_cloud_percentage:
                    continue  # ❌ Rechazar si cubre menos del 50% o supera nubosidad máxima
   
                # Escoger imagen con MENOR nubosidad
                if best_score is None or cloud_score < best_score:
                    best_score = cloud_score
                    best_image = image
                
                    # Registrar el resultado de nubosidad solo para la mejor imagen
                    image_time = image.get('system:time_start').getInfo()
                    hora = datetime.utcfromtimestamp(image_time / 1000).strftime('%H:%M')
                    if "used_cloud_results" not in st.session_state:
                        st.session_state["used_cloud_results"] = []
                    st.session_state["used_cloud_results"].append({
                        "Fecha": selected_date,
                        "Hora": hora,
                        "Nubosidad aproximada (%)": round(cloud_score, 2)
                    })
                

            except Exception as e:
                st.warning(f"Error al procesar imagen {i}: {e}")
                continue

        if best_image is None:
            st.warning(f"No se encontró ninguna imagen útil para la fecha {selected_date}")
            return None, None, None

        sentinel2_image = best_image
        image_date = sentinel2_image.get('system:time_start').getInfo()
        image_date = datetime.utcfromtimestamp(image_date / 1000).strftime('%Y-%m-%d %H:%M:%S')

        bandas_requeridas = ['B2', 'B3', 'B4', 'B5', 'B6']
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

        indices_functions = {
            "MCI": lambda: b5.subtract(b4).subtract((b6.subtract(b4).multiply(705 - 665).divide(740 - 665))).rename('MCI'),
            "B5_div_B4": lambda: b5.divide(b4).rename('B5_div_B4'),
            "NDCI_ind": lambda: b5.subtract(b4).divide(b5.add(b4)).rename('NDCI_ind'),
            "PC_Val_cal": lambda: b5.divide(b4).subtract(1.41).multiply(-3.97).exp().add(1).pow(-1).multiply(9.04).rename("PC_Val_cal"),
            "Chla_Val_cal": lambda: b5.subtract(b4).divide(b5.add(b4)).multiply(5.05).exp().multiply(23.16).rename("Chla_Val_cal"),
            "Chla_Bellus_cal": lambda: (
                b5.divide(b3)  # B5 / B3
                .add(ee.Image(0.995).divide(b3.add(0.395)))  # 0.995 / (B3 + 0.395)
                .multiply(-22)  # Multiplicamos por -22
                .subtract(0.1)  # Restamos 0.1
                .exp()  # Aplicamos la exponencial
                .add(1)  # Sumamos 1
                .pow(0.25)  # Potenciamos a la -0.25
                .multiply(45)  # Multiplicamos por 45
                .rename("Chla_Bellus_cal")  # Renombramos el resultado
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



def get_index_value(lon, lat, index_name, indices_image):
    """Función para obtener el valor del índice en un punto específico."""
    point = ee.Geometry.Point(lon, lat)
    value = indices_image.select(index_name).sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point)]),
        scale=20  # Resolución de Sentinel-2
    ).first().get(index_name)

    return value.getInfo() if value is not None else None


def generar_leyenda(indices_seleccionados):
    # Parámetros de visualización para cada índice
    parametros = {
        "MCI": {"min": -0.1, "max": 0.4, "palette": ['blue', 'green', 'yellow', 'red']},
        "B5_div_B4": {"min": 0.5, "max": 1.5, "palette": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]},
        "NDCI_ind": {"min": -0.1, "max": 0.4, "palette": ['blue', 'green', 'yellow', 'red']},
        "PC_Val_cal": {"min": 0, "max": 7, "palette": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]},
        "Chla_Val_cal": {"min": 0,"max": 150,"palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']},
        "Chla_Bellus_cal": {"min": 5,"max": 55,"palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']}
    }

    leyenda_html = "<div style='border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: white;'>"
    leyenda_html += "<h4 style='text-align: center;'>📌 Leyenda de Índices y Capas</h4>"

    # 🔹 Leyenda para la capa SCL (Scene Classification Layer)
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

    # 🔹 Leyenda para la capa MSK_CLDPRB (Probabilidad de nubes)
    msk_palette = ["blue", "green", "yellow", "red", "black"]
    leyenda_html += "<b>Capa MSK_CLDPRB (Probabilidad de Nubes):</b><br>"
    leyenda_html += f"<div style='background: linear-gradient(to right, {', '.join(msk_palette)}); height: 20px; border: 1px solid #000;'></div>"
    leyenda_html += "<div style='display: flex; justify-content: space-between; font-size: 12px;'><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>"
    leyenda_html += "<br>"

    # 🔹 Leyenda para los índices seleccionados
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
    .block-container {
        padding-top: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100% !important;
    }
    .header-container {
        margin-top: 30px;  /* Ajusta este valor para bajar más */
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px; /* Ajusta la altura según necesites */
    }
    div[role="tablist"] {
        display: flex;
        justify-content: center;
        font-size: 20px !important;
        font-weight: bold !important;
    }
    button[role="tab"] {
        padding: 15px 30px !important;
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 4, 1.25])  # Ajustamos la proporción para más espacio en col3
with col1:
    st.image("images/logo_hiblooms.png", width=350)
    st.image("images/ministerio.png", width=350)
with col2:
    st.markdown(
        """
        <h1 style="text-align: center; line-height: 1.2em;">
            Visor de indicadores de la calidad del agua en embalses españoles:
            <br> <span style="display: block; text-align: center;">Proyecto HIBLOOMS</span>
        </h1>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.image("images/bioma.jpg", width=300)  # Nueva imagen que ocupa el ancho de las dos anteriores
    col3a, col3b = st.columns([1, 1])  # Dividimos col3 en dos partes iguales debajo de la nueva imagen
    with col3a:
        st.image("images/logo_ebro.png", width=150)
    with col3b:
        st.image("images/logo_jucar.png", width=150)


tab1, tab2, tab3, tab4 = st.tabs(["Introducción", "Visualización", "Tablas", "Gráficas"])
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

    uploaded_zip = st.file_uploader("Sube un archivo ZIP con tu shapefile de embalses (proyección EPSG:32630)", type=["zip"])

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
    # 🔳 Dividimos el contenido en dos columnas
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

                # Slider de nubosidad
                st.subheader("Selecciona un porcentaje máximo de nubosidad:")
                max_cloud_percentage = st.slider("Dado que las nubes pueden alterar los valores estimados de concentraciones, es importante definir un límite máximo de nubosidad permitida. Es recomendable elegir valores de hasta el 25%, aunque si se quiere ver más imágenes disponibles, se puede aumentar la tolerancia:", 0, 100, 10)
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
                available_indices = ["MCI", "B5_div_B4", "NDCI_ind", "PC_Val_cal", "Chla_Val_cal", "Chla_Bellus_cal"]
                selected_indices = st.multiselect("Selecciona uno o varios índices para visualizar y analizar:", available_indices)
                with st.expander("ℹ️ ¿Qué significa cada índice?"):
                    st.markdown("""
                    - **MCI (Maximum Chlorophyll Index):** Detecta altas concentraciones de clorofila-a, útil para identificar blooms intensos.
                    - **NDCI_ind (Normalized Difference Chlorophyll Index):** Relación normalizada entre bandas del rojo e infrarrojo cercano. Se asocia a clorofila-a.
                    - **PC_Val_cal (Phycocyanin Estimator):** Estimador empírico de ficocianina, un pigmento exclusivo de cianobacterias. Basado en la relación espectral entre el infrarrojo cercano y el rojo, ha sido ajustado a partir de mediciones de ficocianina en el Embalse de El Val.
                    - **B5/B4:** Relación espectral entre el infrarrojo cercano (B5) y el rojo (B4), útil como indicador de biomasa y ficocianina.
                    - **Chla_Val_cal:** Estimación cuantitativa de clorofila-a derivada del NDCI mediante ajuste exponencial a partir de mediciones en el embalse de El Val.
                    - **Chla_Bellus_cal:** Estimación cuantitativa de clorofila-a específicamente calibrada para el embalse de Bellús.
                    """)

                if st.button("Calcular y mostrar resultados"):
                    # 🔁 Limpiar resultados anteriores
                    st.session_state["data_time"] = []
                    st.session_state["urls_exportacion"] = []
                    st.session_state["used_cloud_results"] = []
                    st.session_state["cloud_results"] = []
                
                    with st.spinner("Calculando fechas disponibles..."):
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
                            
                            # Ticks más bajos
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
                            ficocianina_indices = {"PC_Val_cal", "B5_div_B4"}
                            
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
                            
                            # Bellús: cargar datos solo si se ha seleccionado algún índice relacionado
                            if reservoir_name.lower() == "bellus" and (hay_clorofila or hay_ficocianina):
                                # 🔽 Cargar los CSV de Bellús
                                url_fico_bellus = "https://drive.google.com/uc?id=1jeTpJfPTTKORN3iIprh6P_RPXPu16uDa&export=download"
                                url_cloro_bellus = "https://drive.google.com/uc?id=17-jtO6mbjfj_CMnsMo_UX2RQ7IM_0hQ4&export=download"
                                
                                df_fico_bellus = cargar_csv_desde_url(url_fico_bellus)
                                df_cloro_bellus = cargar_csv_desde_url(url_cloro_bellus)
                                
                                # 🔁 Renombrado flexible
                                for col in df_fico_bellus.columns:
                                    if "pc_ivf" in col.lower():
                                        df_fico_bellus.rename(columns={col: "Ficocianina (µg/L)"}, inplace=True)
                                
                                for col in df_cloro_bellus.columns:
                                    if "chla_ivf" in col.lower():
                                        df_cloro_bellus.rename(columns={col: "Clorofila (µg/L)"}, inplace=True)
                                
                                # 🔗 Fusionar y filtrar por fechas
                                if not df_fico_bellus.empty and not df_cloro_bellus.empty:
                                    df_bellus = pd.merge(df_fico_bellus, df_cloro_bellus, on="Fecha-hora", how="outer")
                                    df_bellus = df_bellus.sort_values("Fecha-hora")
                                
                                    # Filtrado por fechas
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




                            
                            # ✅ Guardar data_time *solo después* de añadir (o no) los datos SAICA
                            st.session_state['data_time'] = data_time



                            # Paleta de colores para SCL con una mejor diferenciación
                            scl_palette = {
                                1: '#ff0004', 2: '#000000', 3: '#8B4513', 4: '#00FF00',
                                5: '#FFD700', 6: '#0000FF', 7: '#F4EEEC', 8: '#C8C2C0',
                                9: '#706C6B', 10: '#87CEFA', 11: '#00FFFF'
                            }
                            scl_colors = [scl_palette[i] for i in sorted(scl_palette.keys())]

                            for day in available_dates:
                                scaled_image, indices_image, image_date = process_sentinel2(aoi, day, max_cloud_percentage, selected_indices)
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


                                # 2️⃣ Añadir media diaria del embalse solo en píxeles con SCL == 6
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
                                    "B5_div_B4": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"],  # PCI
                                    "NDCI_ind": ['blue', 'green', 'yellow', 'red'],
                                    "PC_Val_cal": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"],  # Paleta específica para PC
                                    "Simbolic_Index": ['blue', 'green', 'yellow', 'red'],
                                    "Chla_Val_cal": ['#2171b5', '#c7e9c0', '#238b45', '#e31a1c'],
                                    "Chla_Bellus_cal": ['#2171b5', '#c7e9c0', '#238b45', '#e31a1c']
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
                                                vis_params["max"] = 7
                                                vis_params["palette"] = ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]
                                            elif index == "B5_div_B4":
                                                vis_params["min"] = 0.5
                                                vis_params["max"] = 1.5
                                                vis_params["palette"] = ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]
                                            elif index == "Chla_Val_cal":
                                                vis_params["min"] = 0
                                                vis_params["max"] = 150
                                                vis_params["palette"] = ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']
                                            elif index == "Chla_Bellus_cal":
                                                vis_params["min"] = 5
                                                vis_params["max"] = 55
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
                        if "urls_exportacion" in st.session_state and st.session_state["urls_exportacion"]:
                            st.markdown("## 📦 Descarga de imágenes multibanda por fecha")
                        
                            for item in st.session_state["urls_exportacion"]:
                                st.markdown(f"- 🗓️ **{item['fecha']}**: [Descargar GeoTIFF multibanda]({item['url']})")
                        
                            st.info("🔧 Puedes descargar todos los archivos y luego comprimirlos en ZIP en tu ordenador.")

                        with row2[1]:
                            # 🔽 Leyenda de índices y capas
                            with st.expander("🗺️ Leyenda de índices y capas", expanded=False):
                                generar_leyenda(selected_indices)
                        
                            # 🔽 Tabla de nubosidad estimada por imagen
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
                            
                                        chart = alt.Chart(df_indice).mark_bar().encode(
                                            x=alt.X('Date:T', title='Fecha', axis=alt.Axis(format="%d-%b", labelAngle=0)),
                                            y=alt.Y(f'{indice}:Q', title='Concentración (µg/L)'),
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
                            


                            # 🔽 Serie temporal real de ficocianina (solo si embalse es VAL)
                            if reservoir_name.lower() == "val":
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
                        
                                # ✅ Renombrar la columna 'Point' a 'Ubicación'
                                df_time.rename(columns={"Point": "Ubicación"}, inplace=True)
                        
                                # ✅ Crear una única columna 'Fecha' en formato datetime para ordenar
                                if "Fecha" not in df_time.columns:
                                    posibles_fechas = ["Date", "Fecha-hora", "Fecha_dt"]
                                    for col in posibles_fechas:
                                        if col in df_time.columns:
                                            df_time["Fecha"] = pd.to_datetime(df_time[col], errors='coerce')
                                            break
                        
                                # ✅ Verificar que 'Fecha' existe y eliminar duplicados
                                if "Fecha" not in df_time.columns:
                                    st.error("❌ No se encontró ninguna columna de fecha válida.")
                                    st.stop()
                        
                                # ✅ Ordenar por 'Ubicación' y 'Fecha' (orden cronológico)
                                df_time = df_time.dropna(subset=["Fecha"]).sort_values(by=["Ubicación", "Fecha"])
                        
                                # ✅ Convertir la fecha a texto para visualización
                                df_time["Fecha"] = df_time["Fecha"].dt.strftime("%d-%m-%Y %H:%M")
                        
                                # ✅ Eliminar columnas de fecha duplicadas si existen
                                columnas_fecha = ["Date", "Fecha-hora", "Fecha_dt"]
                                df_time.drop(columns=[col for col in columnas_fecha if col in df_time.columns], errors='ignore', inplace=True)
                        
                                # 🔧 Ordenar las columnas
                                columnas = list(df_time.columns)
                                orden = ["Ubicación", "Fecha", "Tipo"]
                                otras = [col for col in columnas if col not in orden]
                                columnas_ordenadas = orden + otras
                                df_time = df_time[columnas_ordenadas]
                        
                                # 🔧 Dividir en puntos de interés y medias del embalse
                                df_medias = df_time[df_time["Ubicación"] == "Media_Embalse"]
                                df_puntos = df_time[df_time["Ubicación"] != "Media_Embalse"]
                        
                                # ✅ Mostrar las tablas corregidas
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
                        
                                    available_indices = ["MCI", "B5_div_B4", "NDCI_ind", "PC_Val_cal", "Chla_Val_cal", "Chla_Bellus_cal"]
                                    selected_indices = st.multiselect("Selecciona los índices a visualizar:", available_indices, key="graficas_indices")
                        
                                    if st.button("Ejecutar modo rápido"):
                                        st.session_state["data_time"] = []
                        
                                        with st.spinner("Obteniendo fechas disponibles..."):
                                            available_dates = get_available_dates(aoi, start_date, end_date, max_cloud_percentage)
                                            if not available_dates:
                                                st.warning("No se encontraron imágenes en ese rango de fechas.")
                                                st.stop()
                        
                                        data_time = []
                                        clorofila_indices = {"MCI", "NDCI_ind", "Chla_Val_cal", "Chla_Bellus_cal"}
                                        ficocianina_indices = {"PC_Val_cal", "B5_div_B4"}
                        
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

