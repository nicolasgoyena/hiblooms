# logic/sentinel_processing.py

import ee
import streamlit as st
from datetime import datetime

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

    if cloud_fraction_scl is None and cloud_fraction_prob is None:
        return None

    if cloud_fraction_scl is not None and cloud_fraction_prob is not None:
        return (
            ee.Number(cloud_fraction_scl).multiply(0.95)
            .add(ee.Number(cloud_fraction_prob).multiply(0.05))
            .multiply(100)
        )

    if cloud_fraction_scl is not None:
        return ee.Number(cloud_fraction_scl).multiply(100)

    return ee.Number(cloud_fraction_prob).multiply(100)


def calculate_coverage_percentage(image, aoi):
    try:
        total_pixels = ee.Image(1).clip(aoi).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=20,
            maxPixels=1e13
        ).get("constant")

        valid_mask = image.select("B4").mask()
        valid_pixels = ee.Image(1).updateMask(valid_mask).clip(aoi).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=20,
            maxPixels=1e13
        ).get("constant")

        if total_pixels is None or valid_pixels is None:
            return 0

        return ee.Number(valid_pixels).divide(ee.Number(total_pixels)).multiply(100).getInfo()
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
        best_image, best_score = None, None

        for i in range(num_images):
            image = ee.Image(images.get(i))
            try:
                cloud_score = calculate_cloud_percentage(image, aoi).getInfo()
                coverage = calculate_coverage_percentage(image, aoi)

                if coverage < 50 or cloud_score > max_cloud_percentage:
                    continue

                if best_score is None or cloud_score < best_score:
                    best_score = cloud_score
                    best_image = image

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

        b4, b5, b6 = scaled_image.select('B4'), scaled_image.select('B5'), scaled_image.select('B6')

        indices_functions = {
            "MCI": lambda: b5.subtract(b4).subtract((b6.subtract(b4).multiply(705 - 665).divide(740 - 665))).rename('MCI'),
            "B5_div_B4": lambda: b5.divide(b4).rename('B5_div_B4'),
            "NDCI_ind": lambda: b5.subtract(b4).divide(b5.add(b4)).rename('NDCI_ind'),
            "PC_Val_cal": lambda: b5.divide(b4).subtract(1.41).multiply(-3.97).exp().add(1).pow(-1).multiply(9.04).rename("PC_Val_cal"),
            "Chla_Val_cal": lambda: b5.subtract(b4).divide(b5.add(b4)).multiply(5.05).exp().multiply(23.16).rename("Chla_Val_cal"),
            "Chla_Bellus_cal": lambda: (
                b5.subtract(b4).divide(b5.add(b4))
                .multiply(-22).multiply(-1)
                .subtract(22 * 0.1)
                .exp()
                .add(1)
                .pow(-0.25)
                .multiply(45)
                .rename("Chla_Bellus_cal")
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


def calcular_media_diaria_embalse(indices_image, index_name, aoi):
    scl = indices_image.select('SCL')
    fecha_millis = indices_image.get('system:time_start').getInfo()
    year = datetime.utcfromtimestamp(fecha_millis / 1000).year

    if year == 2018:
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
