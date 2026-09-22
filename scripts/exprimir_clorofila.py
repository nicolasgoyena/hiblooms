"""
¿Se puede subir el R² de la clorofila de El Val mejorando los DATOS DE ENTRADA?

Prueba, siempre con la misma validación (dejando fuera un año completo):
  · corrección atmosférica: L2A Sen2Cor (lo actual) frente a L1C sin corregir (TOA)
  · ventana alrededor de la boya: 1 píxel · 3×3 · 5×5
  · capa de la sonda: 0–1,5 m · 0–3 m · 0–5 m
  · filtros de calidad de imagen: brillo solar (B11 alto) y aerosoles (AOT alto)
  · añadir la estacionalidad (día del año) como variable
Modelo base: cuadrático sobre NDCI en escala log (el de la web). Para la mejor
configuración también se prueba un bosque aleatorio con todas las bandas.

Uso (Anaconda Prompt, carpeta del repo):
    set "DATABASE_URL=postgresql://postgres:CLAVE@localhost:5432/unav_water_sampling"
    set "GEE_SERVICE_ACCOUNT_JSON=C:\\Users\\ngoyenaserv\\gee.json"
    python scripts\\exprimir_clorofila.py
La descarga de GEE se guarda en resultados_exprimir/s2_variantes.csv: si vuelves a
ejecutarlo, no se repite.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validacion_sonda import BOYA, RES_ID, init_gee  # noqa: E402
from mejorar_clorofila import metricas  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "resultados_exprimir"
B = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11"]
VENTANAS = {"1px": 10, "3x3": 30, "5x5": 50}          # radio del buffer (m) a 20 m de píxel


# ── 1. Sentinel-2: L2A y L1C, tres ventanas ─────────────────────────────────

def extraer(ee) -> pd.DataFrame:
    pt = ee.Geometry.Point(list(BOYA))
    entorno = pt.buffer(300)

    def feat(img):
        # Todo se combina como diccionarios: si un valor sale enmascarado (nulo) no rompe
        toa = ee.Image(img.get("toa"))
        scl = img.select("SCL")
        props = ee.Dictionary({"t": img.get("system:time_start")})
        nub = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).rename("nubes300")
               .reduceRegion(ee.Reducer.mean(), entorno, 20))
        aot = img.select("AOT").multiply(0.001).rename("aot").reduceRegion(ee.Reducer.mean(), pt.buffer(30), 20)
        props = props.combine(nub).combine(aot)
        for w, r in VENTANAS.items():
            g = pt.buffer(r)
            agua = scl.eq(6).rename(f"agua_{w}").reduceRegion(ee.Reducer.mean(), g, 20)
            sr = img.select(B).divide(10000).rename([f"sr_{w}_{b}" for b in B]).reduceRegion(ee.Reducer.mean(), g, 20)
            tt = toa.select(B).divide(10000).rename([f"toa_{w}_{b}" for b in B]).reduceRegion(ee.Reducer.mean(), g, 20)
            props = props.combine(agua).combine(sr).combine(tt)
        return ee.Feature(None, props)

    filas = []
    for y in range(2018, date.today().year + 1):
        a, b = f"{y}-01-01", f"{y + 1}-01-01"
        sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate(a, b)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
        toa = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(pt).filterDate(a, b)
        j = ee.Join.saveFirst("toa").apply(sr, toa, ee.Filter.equals(leftField="system:index", rightField="system:index"))
        fc = ee.FeatureCollection(ee.ImageCollection(j).map(feat)).getInfo()["features"]
        n0 = len(filas)
        filas += [f["properties"] for f in fc if f["properties"].get("sr_3x3_B4") is not None]
        print(f"   {y}: {len(filas) - n0} imágenes")
    s = pd.DataFrame(filas)
    s["t"] = pd.to_datetime(s["t"], unit="ms", utc=True)
    for c in ["nubes300", "aot"] + [f"agua_{w}" for w in VENTANAS]:
        if c not in s:
            s[c] = np.nan
    s["nubes300"] = s["nubes300"].fillna(0)
    s["aot"] = s["aot"].fillna(0)
    return s


# ── 2. Sonda por capas ──────────────────────────────────────────────────────

def sonda(url: str, fechas: list[str]) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    q = text("""SELECT date_time::timestamptz AS t, depth, chlorophyll FROM sensor_data
                WHERE reservoir_id = :r AND COALESCE(qc_flag, 0) < 2 AND chlorophyll IS NOT NULL
                  AND (date_time::timestamptz AT TIME ZONE 'UTC')::date = ANY(CAST(:f AS date[]))""")
    with create_engine(url).connect() as c:
        s = pd.read_sql(q, c, params={"r": RES_ID, "f": fechas})
    s["t"] = pd.to_datetime(s["t"], utc=True)
    return s


def emparejar(s2: pd.DataFrame, so: pd.DataFrame, horas: float = 3) -> pd.DataFrame:
    out = []
    for r in s2.to_dict("records"):
        w = so[(so.t - r["t"]).abs() <= pd.Timedelta(hours=horas)]
        if w.empty:
            continue
        for capa, zmax in (("0-1.5", 1.5), ("0-3", 3.0), ("0-5", 5.0)):
            v = w[w.depth.isna() | (w.depth <= zmax)].chlorophyll
            r[f"chl_{capa}"] = v.mean() if len(v) else np.nan
        out.append(r)
    return pd.DataFrame(out)


# ── 3. Validación ───────────────────────────────────────────────────────────

def cv(X: np.ndarray, ly: np.ndarray, years: np.ndarray, tipo: str) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.ensemble import RandomForestRegressor
    p = np.full(len(ly), np.nan)
    for yr in np.unique(years):
        te, tr = years == yr, years != yr
        if te.sum() < 3 or tr.sum() < 20:
            continue
        if tipo == "rf":
            mdl = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=1)
        else:
            mdl = make_pipeline(PolynomialFeatures(2), LinearRegression())
        mdl.fit(X[tr], ly[tr])
        p[te] = mdl.predict(X[te])
    return 10 ** p


def evaluar(m: pd.DataFrame) -> pd.DataFrame:
    filas = []
    doy = m.t.dt.dayofyear.values
    S = np.c_[np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    for src in ("sr", "toa"):
        for w in VENTANAS:
            b4, b5 = m[f"{src}_{w}_B4"], m[f"{src}_{w}_B5"]
            ndci = ((b5 - b4) / (b5 + b4)).values
            base_ok = (m[f"agua_{w}"] >= 0.75) & (m.nubes300 <= 0.01) & np.isfinite(ndci)
            filtros = {"sin filtro": base_ok,
                       "sin brillo (B11<0,02)": base_ok & (m[f"sr_{w}_B11"] < 0.02),
                       "sin brillo ni aerosoles": base_ok & (m[f"sr_{w}_B11"] < 0.02) & (m.aot < 0.15)}
            for capa in ("0-1.5", "0-3", "0-5"):
                y = m[f"chl_{capa}"].values
                for fnom, ok in filtros.items():
                    k = (ok & np.isfinite(y) & (y > 0)).values
                    if k.sum() < 60:
                        continue
                    ly, yrs = np.log10(y[k]), m.t.dt.year.values[k]
                    for extra, X in (("NDCI", ndci[k, None]), ("NDCI + estacionalidad", np.c_[ndci[k], S[k]])):
                        p = cv(X, ly, yrs, "quad")
                        filas.append({"corrección": "L2A Sen2Cor" if src == "sr" else "L1C sin corregir",
                                      "ventana": w, "capa_sonda": capa, "filtro": fnom, "predictores": extra,
                                      **metricas(y[k], p)})
    return pd.DataFrame(filas).sort_values("R2_log", ascending=False)


def main():
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("Falta DATABASE_URL.")
    OUT.mkdir(exist_ok=True)
    f = OUT / "s2_variantes.csv"
    if f.exists():
        print("1/3 Sentinel-2: usando la descarga guardada")
        s2 = pd.read_csv(f); s2["t"] = pd.to_datetime(s2["t"], format="ISO8601", utc=True)
    else:
        print("1/3 Sentinel-2 L2A + L1C en tres ventanas (GEE, unos minutos)…")
        s2 = extraer(init_gee())
        s2.to_csv(f, index=False)
    print("2/3 Sonda por capas…")
    so = sonda(url, sorted(set(s2.t.dt.strftime("%Y-%m-%d"))))
    m = emparejar(s2, so)
    m.to_csv(OUT / "pares_variantes.csv", index=False)
    print(f"   {len(m)} imágenes con sonda en ±3 h")
    print("3/3 Validando todas las combinaciones (dejando fuera cada año)…")
    res = evaluar(m)
    res.to_csv(OUT / "comparacion.csv", index=False)
    cols = ["corrección", "ventana", "capa_sonda", "filtro", "predictores", "n", "R2_log", "MAE_log(×)", "AUC≥10", "clase_trófica_ok"]
    ref = res[(res["corrección"] == "L2A Sen2Cor") & (res.ventana == "3x3") & (res.capa_sonda == "0-1.5")
              & (res.filtro == "sin filtro") & (res.predictores == "NDCI")]
    print("\n== Referencia (lo que tiene ahora la web) ==")
    print(ref[cols].to_string(index=False))
    print("\n== Las 12 mejores combinaciones ==")
    print(res[cols].head(12).to_string(index=False))

    # Bosque aleatorio con todas las bandas en la mejor combinación
    best = res.iloc[0]
    src = "sr" if best["corrección"].startswith("L2A") else "toa"
    w, capa = best.ventana, best.capa_sonda
    y = m[f"chl_{capa}"].values
    X = m[[f"{src}_{w}_{b}" for b in B]].values
    k = np.isfinite(X).all(1) & np.isfinite(y) & (y > 0) & (m[f"agua_{w}"] >= 0.75).values & (m.nubes300 <= 0.01).values
    p = cv(X[k], np.log10(y[k]), m.t.dt.year.values[k], "rf")
    rf = metricas(y[k], p)
    print(f"\n== Bosque aleatorio, todas las bandas ({best['corrección']}, {w}, {capa}) ==")
    print({c: rf[c] for c in ("n", "R2_log", "MAE_log(×)", "AUC≥10", "clase_trófica_ok")})
    print(f"\nListo → {OUT}")


if __name__ == "__main__":
    main()
