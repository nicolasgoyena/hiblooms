"""
Validación Sentinel-2 ↔ MUESTREOS DE CAMPO de HIBLOOMS (todos los embalses).

A diferencia de validacion_sonda.py (un embalse, serie temporal), aquí se usan los
muestreos del proyecto en todos los embalses: el caso del artículo de Water (2021), con
rangos muy distintos entre embalses.

Variables de campo (superficie, ≤ 1,5 m), las que existan en la base:
  · clorofila: chl_a (laboratorio/campo), chl_total_probe (FluoroProbe)
  · cianobacterias: chl_cyano (FluoroProbe), ficocianina (si hay), biovolumen y % de
    cianobacterias del recuento de fitoplancton
Satélite: la imagen Sentinel-2 válida más cercana en ±3 días a cada muestreo (3×3 píxeles).

Validación: dejando fuera un EMBALSE entero (¿funciona en un embalse que el modelo no ha
visto?), y comparación con "solo estacionalidad".

Uso (Anaconda Prompt, carpeta del repo):
    set "DATABASE_URL=postgresql://postgres:CLAVE@localhost:5432/unav_water_sampling"
    set "GEE_SERVICE_ACCOUNT_JSON=C:\\Users\\ngoyenaserv\\gee.json"
    python scripts\\validacion_muestreos.py
Resultados en resultados_muestreos/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "web" / "backend"))
from validacion_sonda import BANDAS, INDICES, auc, combinaciones, init_gee, _tabla  # noqa: E402

OUT = ROOT / "resultados_muestreos"
VENTANA_DIAS = 3


# ── Datos de campo ──────────────────────────────────────────────────────────

def es_objetivo(code: str, name: str) -> str | None:
    c, n = code.lower(), str(name).lower()
    if "sediment" in c:
        return None
    if "chl_cyano" in c or "cianobacteria" in n and "clorof" in n:
        return "Clorofila de cianobacterias (FluoroProbe)"
    if "phycocyan" in c or "ficocian" in n:
        return "Ficocianina (campo)"
    if c in ("chl_a", "chla") or c.startswith("chl_a") or n.strip() == "clorofila a":
        return "Clorofila a (laboratorio/campo)"
    if "chl_total" in c or ("clorofila total" in n):
        return "Clorofila total (FluoroProbe)"
    return None


def muestreos() -> pd.DataFrame:
    import projectdb as pdb
    d = pdb.data()
    o, cat, pts = d["obs"], d["catalog"], d["points"]
    cmap = {r.parameter_code: es_objetivo(r.parameter_code, r.name) for r in cat.itertuples()}
    o = o.assign(var=o["parameter_code"].map(cmap)).dropna(subset=["var"])
    o = o[o["depth_m"].isna() | (o["depth_m"] <= 1.5)]
    g = o.groupby(["site", "date", "var"])["value"].mean().unstack("var").reset_index()
    print("   variables de campo:", {k: int(g[k].notna().sum()) for k in g.columns if k not in ("site", "date")})
    # Fitoplancton: biovolumen y % de cianobacterias por muestra
    ph = pdb.phyto()["samples"]
    if ph:
        f = pd.DataFrame([{"site": s["site"], "date": pd.Timestamp(s["date"]),
                           "Biovolumen de cianobacterias (µm³/mL)": s["groups"].get("Cyanobacteria", 0.0),
                           "% cianobacterias (biovolumen)": s["cyano_pct"]}
                          for s in ph if s["depth_m"] is None or s["depth_m"] <= 1.5])
        f = f.groupby(["site", "date"], as_index=False).mean()
        g = g.merge(f, on=["site", "date"], how="outer")
        print(f"   muestras de fitoplancton: {len(f)}")
    p = pts.drop_duplicates("site").set_index("site")
    g["lat"], g["lon"] = g["site"].map(p["site_lat"]), g["site"].map(p["site_lon"])
    g["embalse"], g["punto"] = g["site"].map(p["water_body_name"]), g["site"].map(p["site_code"])
    g["date"] = pd.to_datetime(g["date"])
    return g.dropna(subset=["lat", "lon"])


# ── Sentinel-2 en cada punto ────────────────────────────────────────────────

def s2_por_punto(ee, g: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for site, gg in g.groupby("site"):
        lon, lat = float(gg.lon.iloc[0]), float(gg.lat.iloc[0])
        pt = ee.Geometry.Point([lon, lat])
        cerca, entorno = pt.buffer(30), pt.buffer(300)
        a = (gg.date.min() - pd.Timedelta(days=VENTANA_DIAS)).strftime("%Y-%m-%d")
        b = (gg.date.max() + pd.Timedelta(days=VENTANA_DIAS + 1)).strftime("%Y-%m-%d")

        def feat(img):
            scl = img.select("SCL")
            m = img.select(BANDAS).divide(10000).reduceRegion(ee.Reducer.mean(), cerca, 20)
            agua = scl.eq(6).reduceRegion(ee.Reducer.mean(), cerca, 20).get("SCL")
            nub = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).reduceRegion(ee.Reducer.mean(), entorno, 20).get("SCL")
            return ee.Feature(None, m.set("agua", agua).set("nubes300", nub).set("t", img.get("system:time_start")))
        # solo las fechas cercanas a algún muestreo (filtro por lista de ventanas)
        filt = None
        for dte in gg.date.drop_duplicates():
            f = ee.Filter.date((dte - pd.Timedelta(days=VENTANA_DIAS)).strftime("%Y-%m-%d"),
                               (dte + pd.Timedelta(days=VENTANA_DIAS + 1)).strftime("%Y-%m-%d"))
            filt = f if filt is None else ee.Filter.Or(filt, f)
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate(a, b).filter(filt)
        try:
            feats = ee.FeatureCollection(col.map(feat)).getInfo()["features"]
        except Exception as e:  # noqa: BLE001
            print(f"   ✗ {site}: {str(e)[:120]}")
            continue
        for x in feats:
            p = x["properties"]
            if p.get("B4") is None:
                continue
            p["site"] = site
            filas.append(p)
        print(f"   {gg['embalse'].iloc[0]} · {gg['punto'].iloc[0]}: {len(gg)} muestreos, {len(feats)} imágenes")
    s = pd.DataFrame(filas)
    if s.empty:
        return s
    s["t"] = pd.to_datetime(s["t"], unit="ms", utc=True).dt.tz_localize(None)
    s = s[(s.agua >= 0.99) & (s.nubes300 <= 0.01) & (s.B4 > 0)]
    return s


def emparejar(g: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in g.to_dict("records"):
        c = s[s.site == r["site"]]
        if c.empty:
            continue
        dt = (c.t.dt.normalize() - r["date"]).dt.days.abs()
        c = c[dt <= VENTANA_DIAS]
        if c.empty:
            continue
        best = c.loc[(c.t.dt.normalize() - r["date"]).dt.days.abs().idxmin()]
        r.update({b: best[b] for b in BANDAS})
        r["dias_desfase"] = int(abs((best.t.normalize() - r["date"]).days))
        out.append(r)
    m = pd.DataFrame(out)
    if not m.empty:
        for k, (_, fn) in INDICES.items():
            m[k] = fn(m)
    return m


# ── Evaluación ──────────────────────────────────────────────────────────────

def evaluar(m: pd.DataFrame, col: str) -> tuple[dict, pd.DataFrame]:
    from scipy import stats
    d = m.dropna(subset=[col]).copy()
    d = d[np.isfinite(d[col])]
    r = {"variable": col, "n": len(d), "embalses": int(d["embalse"].nunique()),
         "mismo_dia": int((d.dias_desfase == 0).sum())}
    if len(d) < 12 or d["embalse"].nunique() < 3:
        r["nota"] = "pocos datos o embalses para validar"
        return r, pd.DataFrame()
    y = d[col]
    r["rango"] = f"{y.min():.3g} – {y.max():.3g}"
    # índices con nombre, en todo el conjunto
    filas = []
    for k in INDICES:
        x = d[k]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 10:
            rho = stats.spearmanr(x[ok], y[ok])[0]
            filas.append({"indice": k, "rho": round(float(rho), 3)})
    tabla = pd.DataFrame(filas).sort_values("rho", key=abs, ascending=False)
    r["NDCI rho (todo)"] = float(tabla.set_index("indice").rho.get("NDCI", np.nan))
    # búsqueda exhaustiva validada dejando fuera un embalse
    C = combinaciones(d)
    thr = float(y.quantile(0.75))
    oof = pd.Series(np.nan, index=d.index)
    elegidos = []
    for emb in d["embalse"].unique():
        te, tr = d["embalse"] == emb, d["embalse"] != emb
        if tr.sum() < 10:
            continue
        rr = C[tr].rank().corrwith(y[tr].rank()).dropna()
        if rr.empty:
            continue
        best = rr.abs().idxmax(); sg = np.sign(rr[best]); elegidos.append(best)
        trv = (sg * C.loc[tr, best]).dropna().sort_values().values
        oof[te] = np.searchsorted(trv, (sg * C.loc[te, best]).values) / max(len(trv), 1)
    ok = oof.notna()
    if ok.sum() >= 10:
        r["mejor índice · rho validado (embalse fuera)"] = round(float(stats.spearmanr(oof[ok], y[ok])[0]), 3)
        r["mejor índice · AUC p75 validado"] = round(float(auc((y[ok] >= thr).values, oof[ok].values)), 3)
        r["índice más elegido"] = pd.Series(elegidos).value_counts().index[0]
    # NDCI tal cual, sin elegir nada (no puede sobreajustar)
    x = d["NDCI"]
    ok2 = np.isfinite(x)
    r["NDCI · AUC p75"] = round(float(auc((y[ok2] >= thr).values, x[ok2].values)), 3)
    # bosque aleatorio (bandas + índices) y solo estacionalidad, embalse fuera
    try:
        from sklearn.ensemble import RandomForestRegressor
        X = pd.concat([d[BANDAS], d[list(INDICES)]], axis=1).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        doy = d["date"].dt.dayofyear
        S = pd.DataFrame({"sin": np.sin(2 * np.pi * doy / 365), "cos": np.cos(2 * np.pi * doy / 365)}, index=d.index)
        ylog = np.log10(np.clip(y, 1e-3, None))
        for nombre, XX in (("RF satélite", X), ("solo estacionalidad", S)):
            pred = pd.Series(np.nan, index=d.index)
            for emb in d["embalse"].unique():
                te, tr = d["embalse"] == emb, d["embalse"] != emb
                if tr.sum() < 10:
                    continue
                rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1)
                rf.fit(XX[tr], ylog[tr]); pred[te] = rf.predict(XX[te])
            k = pred.notna()
            if k.sum() >= 10:
                r[f"{nombre} · rho validado"] = round(float(stats.spearmanr(pred[k], y[k])[0]), 3)
                r[f"{nombre} · AUC p75"] = round(float(auc((y[k] >= thr).values, pred[k].values)), 3)
    except ImportError:
        pass
    return r, tabla


def figura(m: pd.DataFrame, cols: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [c for c in cols if c in m and m[c].notna().sum() >= 5]
    if not cols:
        return
    fig, axs = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.2), squeeze=False)
    embs = sorted(m["embalse"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    for ax, c in zip(axs[0], cols):
        d = m.dropna(subset=[c])
        for i, e in enumerate(embs):
            dd = d[d.embalse == e]
            if len(dd):
                ax.scatter(dd["NDCI"], dd[c], s=26, color=cmap(i % 20), edgecolor="k", linewidth=.3, label=e)
        ax.set_yscale("log"); ax.set_xlabel("NDCI (B5−B4)/(B5+B4)"); ax.set_title(c, fontsize=9); ax.grid(alpha=.3)
    axs[0][-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Muestreos HIBLOOMS · Sentinel-2 (±3 días) — un color por embalse")
    fig.savefig(OUT / "dispersion_muestreos.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    if not os.getenv("DATABASE_URL"):
        sys.exit("Falta DATABASE_URL (base local unav_water_sampling).")
    OUT.mkdir(exist_ok=True)
    print("1/3 Muestreos de campo…")
    g = muestreos()
    g.to_csv(OUT / "muestreos.csv", index=False)
    print(f"   {len(g)} muestreos (punto+fecha) en {g['embalse'].nunique()} masas de agua")
    print("2/3 Sentinel-2 en cada punto (GEE)…")
    s = s2_por_punto(init_gee(), g)
    m = emparejar(g, s)
    m.to_csv(OUT / "emparejamientos.csv", index=False)
    print(f"   {len(m)} muestreos con imagen válida en ±{VENTANA_DIAS} días")
    print("3/3 Evaluación (dejando fuera cada embalse)…")
    vars_ = [c for c in m.columns if c.startswith(("Clorofila", "Ficocianina", "Biovolumen", "% ciano"))]
    res, L = [], ["# Validación Sentinel-2 ↔ muestreos HIBLOOMS", ""]
    for c in vars_:
        r, tabla = evaluar(m, c)
        res.append(r)
        L += [f"## {c}", ""] + [f"- **{k}**: {v}" for k, v in r.items()] + [""]
        if not tabla.empty:
            L += ["Índices con nombre (todo el conjunto):", "", _tabla(tabla), ""]
    pd.DataFrame(res).to_csv(OUT / "resultados.csv", index=False)
    (OUT / "informe.md").write_text("\n".join(L), encoding="utf-8")
    figura(m, vars_)
    print(f"\nListo → {OUT}\n")
    for r in res:
        print(f"== {r['variable']} ==")
        for k, v in r.items():
            if k != "variable":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
