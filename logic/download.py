# logic/download.py

def generar_url_geotiff_multibanda(indices_image, selected_indices, region, scale=20):
    """
    Genera una URL de descarga en formato GeoTIFF multibanda a partir de la imagen de índices seleccionada.
    
    Args:
        indices_image: ee.Image con las bandas calculadas.
        selected_indices: lista de nombres de índices (bandas) a incluir.
        region: ee.Geometry representando el área de interés.
        scale: resolución espacial en metros (por defecto: 20 m).

    Returns:
        URL de descarga si tiene éxito, None si ocurre algún error.
    """
    try:
        url = indices_image.select(selected_indices).getDownloadURL({
            'scale': scale,
            'region': region.getInfo()['coordinates'],
            'fileFormat': 'GeoTIFF'
        })
        return url
    except Exception as e:
        return None
