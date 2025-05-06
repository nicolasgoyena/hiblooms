# data/reservoirs.py
import os
import ee
import geopandas as gpd
import streamlit as st
import folium

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
        reprojected_coords[place] = {}
        for point_id, (lat, lon) in points.items():
            point = ee.Geometry.Point([lon, lat])
            reprojected_point = point.transform(target_crs)
            reprojected_coords[place][point_id] = reprojected_point.coordinates().getInfo()
    return reprojected_coords

def obtener_nombres_embalses(shapefile_path="shapefiles/embalses_hiblooms.shp"):
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        if "NOMBRE" in gdf.columns:
            return sorted(gdf["NOMBRE"].dropna().unique())
        else:
            st.error("❌ El shapefile cargado no contiene una columna llamada 'NOMBRE'.")
    else:
        st.error(f"No se encontró el archivo {shapefile_path}.")
    return []

def cargar_y_mostrar_embalses(map_object, shapefile_path="shapefiles/embalses_hiblooms.shp", nombre_columna="NOMBRE"):
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        bounds = gdf.total_bounds
        map_object.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        for _, row in gdf.iterrows():
            nombre_embalse = row.get(nombre_columna, "Embalse desconocido")
            if row.geometry.geom_type == 'Point':
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=nombre_embalse,
                    tooltip=nombre_embalse,
                    icon=folium.Icon(color="blue", icon="tint")
                ).add_to(map_object)
            elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                folium.GeoJson(
                    row.geometry,
                    name=nombre_embalse,
                    tooltip=folium.Tooltip(nombre_embalse),
                    style_function=lambda x: {
                        "fillColor": "blue",
                        "color": "blue",
                        "weight": 2,
                        "fillOpacity": 0.4
                    }
                ).add_to(map_object)
    else:
        st.error(f"No se encontró el archivo {shapefile_path}.")

def load_reservoir_shapefile(reservoir_name, shapefile_path="shapefiles/embalses_hiblooms.shp"):
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        if "NOMBRE" not in gdf.columns:
            st.error("❌ El shapefile cargado no contiene una columna llamada 'NOMBRE'.")
            return None
        if gdf.crs is None or gdf.crs.to_epsg() != 32630:
            gdf = gdf.to_crs(epsg=32630)
        gdf["NOMBRE"] = gdf["NOMBRE"].str.lower().str.replace(" ", "_")
        normalized_name = reservoir_name.lower().replace(" ", "_")
        gdf_filtered = gdf[gdf["NOMBRE"] == normalized_name]
        if gdf_filtered.empty:
            st.error(f"No se encontró el embalse {reservoir_name}.")
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
        geometry = list(geometry.geoms)[0]
    ee_coordinates = list(geometry.exterior.coords)
    return ee.Geometry.Polygon(ee_coordinates, proj=ee.Projection("EPSG:32630"), geodesic=False)
