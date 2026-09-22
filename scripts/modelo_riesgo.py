"""
Modelo de RIESGO de picos de ficocianina en El Val (HIBLOOMS) · prueba rápida.

Objetivo: ¿se puede avisar de los días con ficocianina alta usando SOLO datos que la web
tiene cada día en cualquier embalse? (época del año, meteorología ERA5-Land, NDCI de
Sentinel-2 y nivel del embalse). La sonda SAICA 945 solo se usa como verdad de campo.

Diseño honesto:
  · Variable objetivo: media diaria de ficocianina en superficie (SAICA 945, sin datos erróneos).
    "Pico" = día por encima del percentil 90 de la serie.
  · Predictores del día anterior hacia atrás (lo que sabríamos esa mañana): nada del mismo día.
  · Validación dejando fuera un año completo (2024 / 2025 / 2026).
  · Se compara siempre con "solo estacionalidad": el modelo solo vale si la mejora.

Uso (Anaconda Prompt, carpeta del repo; tras haber ejecutado validacion_sonda.py):
    set "DATABASE_URL=postgresql://postgres:CLAVE@localhost:5432/unav_water_sampling"
    set "GEE_SERVICE_ACCOUNT_JSON=C:\\Users\\ngoyenaserv\\gee.json"
    python scripts\\modelo_riesgo.py
Resultados en resultados_riesgo/.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validacion_sonda import BOYA, RES_ID, auc, init_gee  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados_riesgo"
SOURCE = "che_saica_945"


# ── Datos ───────────────────────────────────────────────────────────────────

def ficocianina_diaria(url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    q = text("""
      SELECT ((date_time::timestamptz) AT TIME ZONE 'Europe/Madrid')::date AS fecha,
             AVG(phycocyanin) AS pc, MAX(phycocyanin) AS pc_max, AVG(water_temp) AS temp_agua, COUNT(*) AS n
      FROM sensor_data
      WHERE reservoir_id = :r AND source_code = :s AND COALESCE(qc_flag, 0) < 2 AND phycocyanin IS NOT NULL
      GROUP BY 1 ORDER BY 1""")
    with create_engine(url).connect() as c:
        d = pd.read_sql(q, c, params={"r": RES_ID, "s": SOURCE})
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d[d.n >= 12]          # días con al menos medio día de registros


def meteo_gee(ee, lon, lat, desde: str, hasta: str) -> pd.DataFrame:
    """ERA5-Land diario en el punto: temperatura, radiación, viento, precipitación."""
    pt = ee.Geometry.Point([lon, lat])
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(desde, hasta)
           .select(["temperature_2m", "surface_solar_radiation_downwards_sum",
                    "u_component_of_wind_10m", "v_component_of_wind_10m", "total_precipitation_sum"]))

    def f(img):
        v = img.reduceRegion(ee.Reducer.first(), pt, 11132)
        return ee.Feature(None, v).set("fecha", img.date().format("YYYY-MM-dd"))
    rows = []
    y0, y1 = int(desde[:4]), int(hasta[:4])
    for y in range(y0, y1 + 1):
        a, b = max(desde, f"{y}-01-01"), min(hasta, f"{y + 1}-01-01")
        if a < b:
            rows += [x["properties"] for x in ee.FeatureCollection(col.filterDate(a, b).map(f)).getInfo()["features"]]
    m = pd.DataFrame(rows).dropna()
    m["fecha"] = pd.to_datetime(m["fecha"])
    m["t_aire"] = m.temperature_2m - 273.15
    m["rad"] = m.surface_solar_radiation_downwards_sum / 1e6       # MJ/m²
    m["viento"] = np.hypot(m.u_component_of_wind_10m, m.v_component_of_wind_10m)
    m["lluvia"] = m.total_precipitation_sum * 1000                  # mm
    return m[["fecha", "t_aire", "rad", "viento", "lluvia"]].sort_values("fecha")


def ndci_diario(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Último NDCI válido de Sentinel-2 (≤ 15 días de antigüedad), del CSV de validacion_sonda.py."""
    f = ROOT / "resultados_validacion" / "s2_boya_todas.csv"
    if not f.exists():
        print("   (no hay resultados_validacion/s2_boya_todas.csv: sin NDCI)")
        return pd.DataFrame({"fecha": fechas})
    s = pd.read_csv(f)
    s = s[(s.agua >= 0.99) & (s.nubes300 <= 0.01) & (s.B4 > 0)].copy()
    s["fecha"] = pd.to_datetime(s["t"], format="ISO8601", utc=True).dt.tz_localize(None).dt.normalize()
    s["ndci"] = (s.B5 - s.B4) / (s.B5 + s.B4)
    s = s.groupby("fecha", as_index=False)["ndci"].mean().sort_values("fecha")
    s["fecha_img"] = s["fecha"]
    base = pd.DataFrame({"fecha": fechas}).sort_values("fecha")
    # imagen de como muy tarde el día anterior (lo que sabríamos esa mañana)
    base["fecha_ref"] = base["fecha"] - pd.Timedelta(days=1)
    j = pd.merge_asof(base, s.rename(columns={"fecha": "fecha_ref"}), on="fecha_ref", direction="backward")
    j["edad_ndci"] = (j["fecha"] - j["fecha_img"]).dt.days
    j.loc[j["edad_ndci"] > 15, "ndci"] = np.nan
    return j[["fecha", "ndci", "edad_ndci"]]


def nivel_diario(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    f = ROOT / "data" / "boletin_historico_hiblooms.csv"
    base = pd.DataFrame({"fecha": fechas})
    if not f.exists():
        return base
    b = pd.read_csv(f)
    b = b[b.hiblooms_id.astype(str).str.upper() == "VAL"].copy()
    if b.empty:
        return base
    b["fecha"] = pd.to_datetime(b["fecha"])
    b = b.dropna(subset=["pct"]).sort_values("fecha")[["fecha", "pct"]]
    return pd.merge_asof(base.sort_values("fecha"), b.rename(columns={"pct": "nivel_pct"}), on="fecha", direction="backward")


def construir(pc, met, ndci, niv) -> pd.DataFrame:
    d = pc.merge(met, on="fecha", how="left").merge(ndci, on="fecha", how="left").merge(niv, on="fecha", how="left")
    d = d.sort_values("fecha").set_index("fecha")
    m = met.set_index("fecha").sort_index()
    # meteorología de los días ANTERIORES (ventanas que terminan ayer)
    for w in (3, 7, 14, 30):
        d[f"t_aire_{w}d"] = m["t_aire"].shift(1).rolling(w).mean().reindex(d.index)
    for w in (7, 14):
        d[f"rad_{w}d"] = m["rad"].shift(1).rolling(w).mean().reindex(d.index)
        d[f"viento_{w}d"] = m["viento"].shift(1).rolling(w).mean().reindex(d.index)
        d[f"lluvia_{w}d"] = m["lluvia"].shift(1).rolling(w).sum().reindex(d.index)
    d["calentamiento_7d"] = d["t_aire_7d"] - d["t_aire_30d"]
    doy = d.index.dayofyear
    d["sin_doy"], d["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["temp_agua_ayer"] = d["temp_agua"].shift(1)       # solo referencia: la web no la tiene en otros embalses
    return d.reset_index()


# ── Modelos ─────────────────────────────────────────────────────────────────

CONJUNTOS = {
    "1 · solo estacionalidad": ["sin_doy", "cos_doy"],
    "2 · + meteorología": ["sin_doy", "cos_doy", "t_aire_7d", "t_aire_14d", "t_aire_30d", "calentamiento_7d",
                           "rad_7d", "rad_14d", "viento_7d", "viento_14d", "lluvia_7d", "lluvia_14d"],
    "3 · + NDCI satélite": None,   # se rellena abajo (2 + NDCI)
    "4 · + nivel embalse": None,   # 3 + nivel
    "ref · + temperatura del agua (sonda)": None,
}
CONJUNTOS["3 · + NDCI satélite"] = CONJUNTOS["2 · + meteorología"] + ["ndci", "edad_ndci"]
CONJUNTOS["4 · + nivel embalse"] = CONJUNTOS["3 · + NDCI satélite"] + ["nivel_pct"]
CONJUNTOS["ref · + temperatura del agua (sonda)"] = CONJUNTOS["4 · + nivel embalse"] + ["temp_agua_ayer"]


def validar(d: pd.DataFrame, umbral: float):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from scipy import stats

    y = (d["pc"] >= umbral).values
    years = d["fecha"].dt.year.values
    filas, preds = [], {}
    for nombre, cols in CONJUNTOS.items():
        cols = [c for c in cols if c in d and d[c].notna().mean() > 0.3]
        X = d[cols]
        for tipo in ("logística", "boosting"):
            p = np.full(len(d), np.nan)
            for yr in np.unique(years):
                te, tr = years == yr, years != yr
                if te.sum() < 30 or y[tr].sum() < 5:
                    continue
                if tipo == "logística":
                    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
                else:
                    mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                         min_samples_leaf=20, random_state=0)
                mdl.fit(X[tr], y[tr])
                p[te] = mdl.predict_proba(X[te])[:, 1]
            k = np.isfinite(p)
            if k.sum() < 50:
                continue
            # ¿cuántos picos caen en el 10 % de días con más riesgo?
            top = p[k] >= np.quantile(p[k], 0.9)
            filas.append({"predictores": nombre, "modelo": tipo, "n_dias": int(k.sum()), "picos": int(y[k].sum()),
                          "AUC": round(auc(y[k], p[k]), 3),
                          "rho_con_pc": round(stats.spearmanr(p[k], d["pc"].values[k])[0], 3),
                          "picos_en_top10%": f"{int((y[k] & top).sum())}/{int(y[k].sum())}",
                          "Brier": round(float(np.mean((p[k] - y[k]) ** 2)), 4)})
            preds[(nombre, tipo)] = p
    return pd.DataFrame(filas), preds


def figura(d, preds, clave, umbral):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(d.fecha, d.pc, color="#1D6FA8", lw=1, label="Ficocianina sonda (media diaria)")
    ax.axhline(umbral, color="#1D6FA8", ls=":", lw=1, label=f"umbral de pico (p90 = {umbral:.1f})")
    ax.set_ylabel("Ficocianina")
    a2 = ax.twinx()
    a2.fill_between(d.fecha, 0, preds[clave], color="#C8561B", alpha=.35, label=f"riesgo · {clave[0]} ({clave[1]})")
    a2.set_ylim(0, 1); a2.set_ylabel("probabilidad de pico (validada por años)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title("El Val · riesgo de pico de ficocianina con datos disponibles cada día (sin sonda)")
    fig.savefig(OUT / "riesgo_serie.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("Falta DATABASE_URL (base local unav_water_sampling).")
    OUT.mkdir(exist_ok=True)
    print("1/4 Ficocianina diaria de la SAICA…")
    pc = ficocianina_diaria(url)
    if pc.empty:
        sys.exit("No hay datos de ficocianina de che_saica_945.")
    ini = (pc.fecha.min() - timedelta(days=40)).strftime("%Y-%m-%d")
    fin = (pc.fecha.max() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"   {len(pc)} días, {pc.fecha.min():%Y-%m-%d} → {pc.fecha.max():%Y-%m-%d}")
    print("2/4 Meteorología ERA5-Land (GEE)…")
    met = meteo_gee(init_gee(), BOYA[0], BOYA[1], ini, fin)
    print(f"   {len(met)} días (ERA5-Land llega con ~1 semana de retraso)")
    print("3/4 NDCI y nivel…")
    d = construir(pc, met, ndci_diario(pd.DatetimeIndex(pc.fecha)), nivel_diario(pd.DatetimeIndex(pc.fecha)))
    d = d.dropna(subset=["t_aire_30d"])
    umbral = float(d["pc"].quantile(0.9))
    d.to_csv(OUT / "datos_diarios.csv", index=False)
    print(f"   {len(d)} días útiles · umbral de pico p90 = {umbral:.2f} · {int((d.pc >= umbral).sum())} días de pico")
    print(f"   NDCI disponible (≤15 días) en {int(d['ndci'].notna().sum()) if 'ndci' in d else 0} días; "
          f"nivel en {int(d['nivel_pct'].notna().sum()) if 'nivel_pct' in d else 0}")
    print("4/4 Validando modelos (dejando fuera cada año)…")
    res, preds = validar(d, umbral)
    res.to_csv(OUT / "resultados_riesgo.csv", index=False)
    if not res.empty:
        cand = res[~res.predictores.str.startswith("ref")].sort_values("AUC", ascending=False).iloc[0]
        figura(d, preds, (cand.predictores, cand.modelo), umbral)
    print(f"\nListo → {OUT}\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
