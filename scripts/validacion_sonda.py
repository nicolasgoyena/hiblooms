"""
Validación rápida Sentinel-2 ↔ sondas de El Val (HIBLOOMS).

Para cada imagen Sentinel-2 (2018–hoy) sobre la boya de El Val:
  1. extrae la reflectancia media en ~3×3 píxeles alrededor de la boya (GEE),
  2. descarta las que no son agua limpia (SCL) o tienen nubes cerca,
  3. empareja con la sonda en ±2 h: clorofila (Aquadam, 2018–2024, 0–1,5 m)
     y ficocianina (Aquadam y SAICA 945, 2024–hoy),
  4. calcula correlaciones y capacidad de detección (AUC) por índice,
  5. guarda tablas, figuras e informe en resultados_validacion/.

Uso (Anaconda Prompt, en la carpeta del repo):
    set DATABASE_URL=postgresql://postgres:CLAVE@localhost:5432/unav_water_sampling
    python scripts\\validacion_sonda.py
Opcional: --lon/--lat de la boya, --desde 2018-01-01, --ventana 2 (horas)

Credenciales GEE: variable GEE_SERVICE_ACCOUNT_JSON (JSON o ruta) o .streamlit/secrets.toml,
igual que la web.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados_validacion"
BOYA = (-1.7883, 41.8761)          # lon, lat de la boya de El Val (la de los scripts antiguos)
RES_ID = 2351

BANDAS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11"]
LAMBDA = {"B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740, "B7": 783,
          "B8": 842, "B8A": 865, "B11": 1610}

INDICES = {
    # nombre: (fórmula legible, función sobre un DataFrame de reflectancias)
    "NDCI": ("(B5−B4)/(B5+B4)", lambda d: (d.B5 - d.B4) / (d.B5 + d.B4)),
    "PCI": ("B5/B4", lambda d: d.B5 / d.B4),
    "MCI": ("B5 − B4 − (B6−B4)·0,53", lambda d: d.B5 - d.B4 - (d.B6 - d.B4) * (705 - 665) / (740 - 665)),
    "3BDA": ("(1/B4 − 1/B5)·B6", lambda d: (1 / d.B4 - 1 / d.B5) * d.B6),
    "B5-B4": ("B5 − B4", lambda d: d.B5 - d.B4),
    "Turbidez(B4)": ("B4", lambda d: d.B4),
    # índices de la bibliografía de cianobacterias/blooms adaptados a las bandas de S2
    "SABI": ("(B8−B4)/(B2+B3)", lambda d: (d.B8 - d.B4) / (d.B2 + d.B3)),
    "FAI": ("B8 − [B4 + (B11−B4)·0,19]", lambda d: d.B8 - (d.B4 + (d.B11 - d.B4) * (842 - 665) / (1610 - 665))),
    "CI_cyano": ("−[B5 − B4 − (B6−B4)·0,53] (Wynne)", lambda d: -(d.B5 - d.B4 - (d.B6 - d.B4) * (705 - 665) / (740 - 665))),
    "B3/B4": ("verde/rojo", lambda d: d.B3 / d.B4),
    "B5/B3": ("borde rojo/verde", lambda d: d.B5 / d.B3),
    "NDVI": ("(B8−B4)/(B8+B4)", lambda d: (d.B8 - d.B4) / (d.B8 + d.B4)),
    "Sombra 620 (B3,B4)": ("B4 − (B3+B5)/2  (hundimiento hacia 620–665)", lambda d: d.B4 - (d.B3 + d.B5) / 2),
}


# ── Google Earth Engine ─────────────────────────────────────────────────────

def init_gee():
    import ee
    sa = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if not sa:
        for cand in (ROOT / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"):
            if cand.exists():
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore
                sec = tomllib.loads(cand.read_text(encoding="utf-8"))
                sa = sec.get("GEE_SERVICE_ACCOUNT_JSON")
                if sa:
                    break
    if sa:
        info = sa if isinstance(sa, dict) else (json.loads(Path(sa).read_text()) if Path(str(sa)).exists() else json.loads(sa))
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info)))
    else:  # cuenta personal (tras "earthengine authenticate")
        ee.Initialize(project=os.getenv("GEE_PROJECT", "ee-nicolasgoyenaserveto"))
    return ee


def extraer_s2(ee, lon: float, lat: float, desde: str) -> pd.DataFrame:
    pt = ee.Geometry.Point([lon, lat])
    cerca, entorno = pt.buffer(30), pt.buffer(300)
    bands = BANDAS

    def feat(img):
        scl = img.select("SCL")
        refl = img.select(bands).divide(10000)
        m = refl.reduceRegion(ee.Reducer.mean(), cerca, 20, maxPixels=1e6)
        agua = scl.eq(6).reduceRegion(ee.Reducer.mean(), cerca, 20).get("SCL")
        # nubes, sombras y cirros en 300 m (8,9,10 nubes · 3 sombra)
        nub = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).reduceRegion(
            ee.Reducer.mean(), entorno, 20).get("SCL")
        return ee.Feature(None, m.set("agua", agua).set("nubes300", nub)
                          .set("t", img.get("system:time_start")).set("id", img.get("system:index")))

    fin = date.today().isoformat()
    filas = []
    for y in range(int(desde[:4]), date.today().year + 1):
        a, b = max(desde, f"{y}-01-01"), min(fin, f"{y + 1}-01-01")
        if a >= b:
            continue
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate(a, b)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
        fc = ee.FeatureCollection(col.map(feat)).getInfo()
        n = 0
        for f in fc["features"]:
            p = f["properties"]
            if p.get("B4") is None:
                continue
            filas.append(p); n += 1
        print(f"  S2 {y}: {n} imágenes")
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df


# ── Sonda ───────────────────────────────────────────────────────────────────

def leer_sonda(url: str, fechas: list[str]) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    eng = create_engine(url)
    q = text("""
      SELECT source_code, date_time::timestamptz AS t, depth, chlorophyll, phycocyanin, water_temp, turbidity
      FROM sensor_data
      WHERE reservoir_id = :r AND COALESCE(qc_flag, 0) < 2
        AND (date_time::timestamptz AT TIME ZONE 'UTC')::date = ANY(CAST(:f AS date[]))""")
    with eng.connect() as c:
        s = pd.read_sql(q, c, params={"r": RES_ID, "f": fechas})
    s["t"] = pd.to_datetime(s["t"], utc=True)
    return s


def emparejar(s2: pd.DataFrame, sonda: pd.DataFrame, horas: float) -> pd.DataFrame:
    out = []
    for row in s2.to_dict("records"):     # (itertuples renombraría "3BDA", "B5-B4"…)
        w = sonda[(sonda.t - row["t"]).abs() <= pd.Timedelta(hours=horas)]
        sup = w[w.depth.isna() | (w.depth <= 1.5)]          # superficie (SAICA no da profundidad)
        col3 = w[w.depth.isna() | (w.depth <= 3.0)]
        row["chl_sup"] = sup.chlorophyll.mean()
        row["chl_0_3"] = col3.chlorophyll.mean()
        row["pc_sup"] = sup.phycocyanin.mean()
        row["pc_0_3"] = col3.phycocyanin.mean()
        row["temp_sup"] = sup.water_temp.mean()
        row["turb_sup"] = sup.turbidity.mean()
        row["n_sonda"] = len(sup)
        row["fuente"] = ",".join(sorted(sup.source_code.dropna().unique()))
        out.append(row)
    return pd.DataFrame(out)


# ── Estadística ─────────────────────────────────────────────────────────────

def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC por Mann-Whitney (sin sklearn). >0,5 = el índice sube con el bloom."""
    pos, neg = score[y_true], score[~y_true]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluar(m: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    filas = []
    objetivos = [("chl_sup", "Clorofila sonda 0–1,5 m (µg/L)", [10, 20]),
                 ("pc_sup", "Ficocianina sonda superficie", None)]
    for col, nombre, umbrales in objetivos:
        d = m.dropna(subset=[col])
        if len(d) < 8:
            filas.append({"variable": nombre, "indice": "—", "n": len(d), "nota": "muy pocos pares"})
            continue
        if umbrales is None:   # ficocianina: sin umbral oficial → percentiles propios
            umbrales = [round(float(d[col].quantile(q)), 2) for q in (0.75, 0.9)]
        for ind in INDICES:
            x, y = d[ind].values, d[col].values
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 8:
                continue
            rho, p = stats.spearmanr(x, y)
            lr = stats.linregress(x, np.log10(np.clip(y, 0.01, None)))
            fila = {"variable": nombre, "indice": ind, "n": len(x), "rho_spearman": round(rho, 3),
                    "p": float(f"{p:.2g}"), "R2_log": round(lr.rvalue ** 2, 3)}
            for u in umbrales:
                yt = y >= u
                fila[f"AUC ≥{u}"] = round(auc(yt, x), 3)
                fila[f"n ≥{u}"] = int(yt.sum())
            filas.append(fila)
    return pd.DataFrame(filas)


def figuras(m: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for col, nombre, fn in (("chl_sup", "Clorofila sonda (µg/L)", "dispersion_clorofila.png"),
                            ("pc_sup", "Ficocianina sonda", "dispersion_ficocianina.png")):
        d = m.dropna(subset=[col])
        if len(d) < 5:
            continue
        fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
        for ax, ind in zip(axs.flat, INDICES):
            sc = ax.scatter(d[ind], d[col], c=d.t.dt.month, cmap="twilight", s=22, edgecolor="k", linewidth=.3)
            ax.set_xlabel(f"{ind}  {INDICES[ind][0]}", fontsize=9); ax.set_ylabel(nombre, fontsize=9)
            ax.set_yscale("log"); ax.grid(alpha=.3)
        fig.colorbar(sc, ax=axs, label="mes", shrink=.6)
        fig.suptitle(f"El Val · Sentinel-2 vs sonda (n={len(d)})")
        fig.savefig(OUT / fn, dpi=140, bbox_inches="tight"); plt.close(fig)
    # Serie temporal: sonda diaria + índice en fechas de satélite
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, col, ind, nombre in ((axs[0], "chl_sup", "NDCI", "Clorofila"), (axs[1], "pc_sup", "PCI", "Ficocianina")):
        d = m.dropna(subset=[col])
        ax.plot(d.t, d[col], "o", ms=4, color="#1D6FA8", label=f"{nombre} sonda (±2 h)")
        ax.set_ylabel(nombre); ax.set_yscale("log")
        a2 = ax.twinx(); a2.plot(d.t, d[ind], "d", ms=4, color="#C8561B", label=ind); a2.set_ylabel(ind)
        ax.legend(loc="upper left"); a2.legend(loc="upper right")
    fig.savefig(OUT / "serie_temporal.png", dpi=140, bbox_inches="tight"); plt.close(fig)


# ── Búsqueda exhaustiva de índices (con validación honesta) ─────────────────
# Probar cientos de combinaciones y quedarse con la mejor sobreajusta: alguna sale buena
# por azar. Por eso se valida "dejando fuera un año": se elige el mejor índice con los
# demás años y se evalúa en el año que no se ha visto. Lo que sobreviva a eso es real.

def combinaciones(d: pd.DataFrame) -> pd.DataFrame:
    b = [x for x in BANDAS if x in d]
    out = {}
    for i, x in enumerate(b):
        for y in b[i + 1:]:
            out[f"ND({x},{y})"] = (d[x] - d[y]) / (d[x] + d[y])
            out[f"{x}−{y}"] = d[x] - d[y]
    for x in b:
        for y in b:
            for z in b:
                if len({x, y, z}) == 3:
                    out[f"(1/{x}−1/{y})·{z}"] = (1 / d[x] - 1 / d[y]) * d[z]
    for x in b:
        for y in b:
            for z in b:
                if LAMBDA[x] < LAMBDA[y] < LAMBDA[z]:
                    f = (LAMBDA[y] - LAMBDA[x]) / (LAMBDA[z] - LAMBDA[x])
                    out[f"altura {y} sobre {x}–{z}"] = d[y] - (d[x] + (d[z] - d[x]) * f)
    for k, (_, fn) in INDICES.items():
        out[f"[{k}]"] = fn(d)
    c = pd.DataFrame(out, index=d.index).replace([np.inf, -np.inf], np.nan)
    return c.loc[:, c.notna().mean() > 0.95]


def _rho(C: pd.DataFrame, y: pd.Series) -> pd.Series:
    return C.rank().corrwith(y.rank())


def explorar(m: pd.DataFrame, col: str, nombre: str) -> tuple[pd.DataFrame, dict]:
    d = m.dropna(subset=[col]).copy()
    d = d[np.isfinite(d[col])]
    res = {"variable": nombre, "n": len(d)}
    if len(d) < 15:
        res["nota"] = "muy pocos pares para explorar"
        return pd.DataFrame(), res
    C = combinaciones(d)
    y = d[col]
    rho = _rho(C, y).dropna()
    top = rho.abs().sort_values(ascending=False).head(20)
    tabla = pd.DataFrame({"indice": top.index, "rho_en_muestra": rho[top.index].round(3).values})
    res["candidatos"] = C.shape[1]
    res["mejor_en_muestra"] = f"{top.index[0]} (rho {rho[top.index[0]]:.2f})"

    # Validación dejando fuera un año
    years = d["t"].dt.year
    thr = [float(y.quantile(q)) for q in (0.75, 0.9)]
    oof, elegidos = pd.Series(np.nan, index=d.index), []
    for yr in sorted(years.unique()):
        te, tr = years == yr, years != yr
        if te.sum() < 3 or tr.sum() < 12:
            continue
        r = _rho(C[tr], y[tr]).dropna()
        if r.empty:
            continue
        best = r.abs().idxmax(); sg = np.sign(r[best])
        elegidos.append(f"{yr}: {best}")
        trv = (sg * C.loc[tr, best]).dropna().sort_values().values
        tev = sg * C.loc[te, best]
        oof[te] = np.searchsorted(trv, tev.values) / max(len(trv), 1)   # percentil frente a entrenamiento
    ok = oof.notna()
    if ok.sum() >= 10:
        from scipy import stats
        res["rho_validado"] = float(round(stats.spearmanr(oof[ok], y[ok])[0], 3))
        for q, u in zip((75, 90), thr):
            res[f"AUC_validado_p{q}"] = float(round(auc((y[ok] >= u).values, oof[ok].values), 3))
    res["elegido_por_año"] = "; ".join(elegidos)

    # Techo de Sentinel-2: modelo con todas las bandas e índices (bosque aleatorio), mismo esquema
    try:
        from sklearn.ensemble import RandomForestRegressor
        from scipy import stats
        X = pd.concat([d[[b for b in BANDAS if b in d]],
                       pd.DataFrame({k: fn(d) for k, (_, fn) in INDICES.items()}, index=d.index)], axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        doy = d["t"].dt.dayofyear
        S = pd.DataFrame({"sin": np.sin(2 * np.pi * doy / 365), "cos": np.cos(2 * np.pi * doy / 365)}, index=d.index)
        ylog = np.log10(np.clip(y, 0.01, None))
        for nombre_m, XX in (("RF bandas S2", X), ("solo estacionalidad (mes)", S), ("RF bandas + estacionalidad", pd.concat([X, S], axis=1))):
            pred = pd.Series(np.nan, index=d.index)
            for yr in sorted(years.unique()):
                te, tr = years == yr, years != yr
                if te.sum() < 3 or tr.sum() < 12:
                    continue
                rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)
                rf.fit(XX[tr], ylog[tr]); pred[te] = rf.predict(XX[te])
            k = pred.notna()
            if k.sum() >= 10:
                res[f"{nombre_m}: rho"] = float(round(stats.spearmanr(pred[k], y[k])[0], 3))
                res[f"{nombre_m}: AUC p90"] = float(round(auc((y[k] >= thr[1]).values, pred[k].values), 3))
    except ImportError:
        res["nota_ml"] = "scikit-learn no instalado: sin modelo de bosque aleatorio"
    return tabla, res


def _tabla(res: pd.DataFrame) -> str:
    if res.empty:
        return "(sin resultados)"
    try:
        return res.to_markdown(index=False)
    except ImportError:          # sin el paquete tabulate
        return "```\n" + res.to_string(index=False) + "\n```"


def informe(m: pd.DataFrame, res: pd.DataFrame, n_img: int, n_ok: int, expl=()):
    L = [f"# Validación Sentinel-2 ↔ sonda · El Val", "",
         f"Generado {datetime.now():%Y-%m-%d %H:%M}. Imágenes S2 sobre la boya: {n_img}; "
         f"válidas (agua limpia, sin nubes a 300 m): {n_ok}; con sonda en ±2 h: {len(m)}.", "",
         f"- Pares con clorofila: {int(m.chl_sup.notna().sum())}",
         f"- Pares con ficocianina: {int(m.pc_sup.notna().sum())}", "",
         "## Resultados por índice", "",
         "rho = correlación de Spearman; R2_log = R² de log10(sonda) frente al índice; "
         "AUC = capacidad de separar muestreos por encima del umbral (0,5 = azar, >0,7 útil, >0,8 buena).", "",
         _tabla(res), "",
         "## Cómo leerlo", "",
         "- Si NDCI/MCI tienen rho ≥ 0,5 y AUC ≥ 0,75 para clorofila, Sentinel-2 sigue la biomasa algal en El Val.",
         "- Si PCI no mejora a NDCI para la ficocianina, el PCI no aporta información propia de cianobacterias "
         "(esperable: Sentinel-2 no tiene banda a 620 nm).",
         "- Mira las figuras de dispersión: la forma (umbral, saturación, nube sin estructura) dice más que un número."]
    for nombre, tabla, r in expl:
        L += ["", f"## Búsqueda exhaustiva · {nombre}", "",
              "Se prueban todas las diferencias normalizadas, diferencias, índices de tres bandas y alturas de línea "
              "entre las 10 bandas de S2, más los índices de la bibliografía. «En muestra» es optimista (se elige y se "
              "evalúa con los mismos datos); lo que vale es lo **validado dejando fuera un año**. "
              "Compara siempre con «solo estacionalidad»: si el satélite no la supera, no aporta información propia.", ""]
        L += [f"- **{k}**: {v}" for k, v in r.items()]
        if not tabla.empty:
            L += ["", "Top 20 en muestra:", "", _tabla(tabla)]
    (OUT / "informe.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, default=BOYA[0]); ap.add_argument("--lat", type=float, default=BOYA[1])
    ap.add_argument("--desde", default="2018-01-01"); ap.add_argument("--ventana", type=float, default=2)
    a = ap.parse_args()
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("Falta DATABASE_URL (la base local unav_water_sampling).")
    OUT.mkdir(exist_ok=True)

    print("1/4 Sentinel-2 en GEE…")
    ee = init_gee()
    s2 = extraer_s2(ee, a.lon, a.lat, a.desde)
    s2.to_csv(OUT / "s2_boya_todas.csv", index=False)
    n_img = len(s2)
    s2 = s2[(s2.agua >= 0.99) & (s2.nubes300 <= 0.01) & (s2.B4 > 0)].copy()
    print(f"   {n_img} imágenes, {len(s2)} válidas")
    for k, (_, f) in INDICES.items():
        s2[k] = f(s2)

    print("2/4 Sonda…")
    fechas = sorted(set((s2.t.dt.strftime("%Y-%m-%d"))))
    sonda = leer_sonda(url, fechas)
    print(f"   {len(sonda)} medidas de sonda en esas fechas")

    print("3/4 Emparejando…")
    m = emparejar(s2, sonda, a.ventana)
    m = m[m.n_sonda > 0]
    m.to_csv(OUT / "emparejamientos.csv", index=False)

    print("4/4 Estadística y figuras…")
    res = evaluar(m)
    res.to_csv(OUT / "resultados.csv", index=False)
    figuras(m)
    print("   búsqueda exhaustiva de índices…")
    expl = []
    for col, nombre in (("pc_sup", "Ficocianina"), ("chl_sup", "Clorofila")):
        tabla, r = explorar(m, col, nombre)
        if not tabla.empty:
            tabla.to_csv(OUT / f"exploracion_{col}.csv", index=False)
        expl.append((nombre, tabla, r))
    informe(m, res, n_img, len(s2), expl)
    print(f"\nListo → {OUT}")
    print(res.to_string(index=False))
    for nombre, _, r in expl:
        print(f"\n== {nombre} ==")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
