# logic/point_values.py

import ee

def get_values_at_point(lat, lon, indices_image, selected_indices):
    if indices_image is None:
        return None

    buffer_radius_meters = 30  # Aprox. 3x3 píxeles de 20m
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
    point = ee.Geometry.Point(lon, lat)
    value = indices_image.select(index_name).sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point)]),
        scale=20
    ).first().get(index_name)

    return value.getInfo() if value is not None else None
