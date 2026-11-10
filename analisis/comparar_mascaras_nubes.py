"""
Compara máscaras de nubes en El Val con la sonda SAICA de la CHE como referencia.

Pregunta: ¿cambiar la máscara actual (clasificación SCL de Sen2Cor) por Cloud Score+ de Google
mejora la serie de la plataforma, o borra floraciones (natas brillantes que se confunden con nubes)?

Máscaras que se comparan, todas dentro del polígono del embalse:
  A_SCL      la actual: píxeles que Sen2Cor clasifica como agua (SCL = 6)
  B_SCL_CS   la actual + Cloud Score+ (cs_cdf ≥ 0,60): más estricta, nunca deja más píxeles que A
  C_AGUA_CS  sin SCL: agua detectada por MNDWI/NDWI (la máscara de lámina de agua de la web,
             que mantiene las natas) + Cloud Score+ (cs_cdf ≥ 0,60)

Para cada imagen de Sentinel-2 con dato diario de la sonda se guarda, por máscara: nº de píxeles,
media y mediana del embalse y media en un círculo de 60 m alrededor de la sonda, para NDCI y la
ficocianina calibrada de El Val (PC_Val_cal). Al final imprime un resumen y deja un CSV.

USO (desde la raíz del repo, con las mismas credenciales que el backend: GEE_SERVICE_ACCOUNT_JSON
o .streamlit/secrets.toml, y DATABASE_URL para leer la sonda):

    python analisis/comparar_mascaras_nubes.py

Tarda unos 5–10 minutos. Resultado:
    analisis/resultados/mascaras_nubes_val.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "web" / "backend")]
os.chdir(ROOT)

import server  # noqa: E402  (inicializa Earth Engine con las credenciales del backend)
import sonda_val  # noqa: E402

if server.MOCK:
    sys.exit(f"Earth Engine no está disponible ({server.GEE_ERROR}). Revisa las credenciales del backend.")
if sonda_val.pdb.is_mock():
    sys.exit("No hay conexión con la base de datos (DATABASE_URL): hace falta para leer la sonda SAICA.")

import ee  # noqa: E402

EMBALSE = "VAL"
SONDA = (-1.7883, 41.8761)          # Sonda-SAICA (data/puntos_interes.csv)
CS_UMBRAL = 0.60                    # recomendado por Google para cs_cdf
INDICES = ["NDCI_ind", "PC_Val_cal"]
MASCARAS = ["A_SCL", "B_SCL_CS", "C_AGUA_CS"]
SALIDA = ROOT / "analisis" / "resultados" / "mascaras_nubes_val.csv"


def por_imagen(img, aoi, sonda):
    clipped = img.clip(aoi)
    ind = server.core._build_indices_image(clipped, aoi, INDICES)
    scl = img.select("SCL")
    cs = img.select("cs_cdf").gte(CS_UMBRAL)
    agua = server._water_mask(ind)
    masks = {
        "A_SCL": scl.eq(6),
        "B_SCL_CS": scl.eq(6).And(cs),
        "C_AGUA_CS": agua.And(cs),
    }
    red = ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True).combine(ee.Reducer.count(), sharedInputs=True)
    props = {}
    for m, mk in masks.items():
        b = ind.select(INDICES).updateMask(mk)
        st = b.reduceRegion(red, aoi, 20, maxPixels=1e13, tileScale=4)
        ps = b.reduceRegion(ee.Reducer.mean(), sonda, 20, maxPixels=1e9)
        for i in INDICES:
            props[f"{m}|{i}|media"] = st.get(f"{i}_mean")
            props[f"{m}|{i}|mediana"] = st.get(f"{i}_median")
            props[f"{m}|{i}|sonda"] = ps.get(i)
        props[f"{m}|npx"] = st.get(f"{INDICES[0]}_count")
    nube = scl.eq(3).Or(scl.gte(7).And(scl.lte(10))).reduceRegion(ee.Reducer.mean(), aoi, 20, maxPixels=1e13).get("SCL")
    props["nubes_scl_pct"] = ee.Number(ee.Algorithms.If(nube, nube, 0)).multiply(100)
    props["fecha"] = img.date().format("YYYY-MM-dd")
    return ee.Feature(None, props)


def hampel(s: pd.Series, k: int = 3, n: float = 4.0) -> pd.Series:
    med = s.rolling(2 * k + 1, center=True, min_periods=4).median()
    mad = (s - med).abs().rolling(2 * k + 1, center=True, min_periods=4).median()
    return (s - med).abs() > n * 1.4826 * mad.clip(lower=1e-6)


def main():
    print("Leyendo la sonda SAICA…")
    d = sonda_val._daily()["pc"].dropna()
    print(f"  {len(d)} días con dato ({d.index.min():%Y-%m-%d} – {d.index.max():%Y-%m-%d})")

    aoi = server._aoi(EMBALSE)
    sonda = ee.Geometry.Point(list(SONDA)).buffer(60)
    s2 = (ee.ImageCollection(server.S2).filterBounds(aoi)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = s2.linkCollection(csp, ["cs_cdf"]).filter(ee.Filter.listContains("system:band_names", "cs_cdf"))

    filas = []
    for y in range(d.index.min().year, d.index.max().year + 1):
        ini, fin = f"{y}-01-01", f"{y + 1}-01-01"
        # _build_indices_image comprueba las bandas en el cliente (getInfo), así que no puede ir
        # dentro de un .map(): se construye cada imagen aquí y se piden todas juntas en lotes.
        ids = s2.filterDate(ini, fin).aggregate_array("system:index").getInfo()
        print(f"Earth Engine {y}: {len(ids)} imágenes…", flush=True)
        feats = []
        for k, iid in enumerate(ids, 1):
            img = ee.Image(f"{server.S2}/{iid}").addBands(
                ee.Image(f"GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED/{iid}").select("cs_cdf"))
            feats.append(por_imagen(img, aoi, sonda))
            if k % 25 == 0 or k == len(ids):
                print(f"   preparadas {k}/{len(ids)}", flush=True)
        for j in range(0, len(feats), 30):
            print(f"   calculando lote {j // 30 + 1}/{(len(feats) + 29) // 30}…", flush=True)
            for f in ee.FeatureCollection(feats[j:j + 30]).getInfo()["features"]:
                filas.append(f["properties"])

    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    # varias teselas el mismo día: nos quedamos con la de más píxeles válidos en la máscara actual
    df = df.sort_values("A_SCL|npx", ascending=False).drop_duplicates("fecha").sort_values("fecha")
    df["sonda_pc"] = df["fecha"].map(d)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"\nCSV: {SALIDA}  ({len(df)} fechas)\n")

    con = df[df["sonda_pc"].notna()].copy()
    p90 = d.quantile(0.9)
    bloom = con["sonda_pc"] >= p90
    print(f"Fechas con imagen y sonda: {len(con)} · días de bloom (sonda ≥ p90 = {p90:.1f} µg/L): {int(bloom.sum())}\n")

    res = []
    for m in MASCARAS:
        for i in INDICES:
            x_emb = pd.to_numeric(con[f"{m}|{i}|media"], errors="coerce")
            x_med = pd.to_numeric(con[f"{m}|{i}|mediana"], errors="coerce")
            x_son = pd.to_numeric(con[f"{m}|{i}|sonda"], errors="coerce")
            y = con["sonda_pc"]
            npx = pd.to_numeric(con[f"{m}|npx"], errors="coerce").fillna(0)
            npx_a = pd.to_numeric(con["A_SCL|npx"], errors="coerce").replace(0, np.nan)
            serie = pd.to_numeric(df[f"{m}|{i}|media"], errors="coerce")
            res.append({
                "máscara": m, "índice": i,
                "fechas con dato": int(x_emb.notna().sum()),
                "rho media embalse": round(x_emb.corr(y, method="spearman"), 3),
                "rho mediana embalse": round(x_med.corr(y, method="spearman"), 3),
                "rho junto a la sonda": round(x_son.corr(y, method="spearman"), 3),
                "picos aislados (Hampel)": int(hampel(serie.dropna()).sum()),
                "% píxeles vs A en bloom": round(100 * (npx[bloom] / npx_a[bloom]).median(), 1) if bloom.any() else None,
                "fechas de bloom sin dato": int((x_emb[bloom].isna()).sum()),
            })
    r = pd.DataFrame(res)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r.to_string(index=False))
    print("\nCómo leerlo: rho más alto = la serie sigue mejor a la sonda; menos picos = serie más limpia;")
    print("«% píxeles vs A en bloom» bajo o «fechas de bloom sin dato» alto = la máscara borra floraciones.")


if __name__ == "__main__":
    main()
