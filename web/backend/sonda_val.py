"""
Sonda SAICA 945 de El Val (CHE): datos en tiempo real y los dos modelos que se apoyan en ella.

  · live()           últimas lecturas de ficocianina, temperatura, turbidez y pH (+ últimos 7 días)
  · riesgo()         riesgo estacional de pico de ficocianina (regresión logística sobre el día del año)
                     validado dejando fuera un año completo → ficha «validado» en la pestaña Modelos
  · tendencia()      persistencia + estacionalidad a 1, 3 y 7 días → ficha «experimental»

Todo se calcula en el servidor a partir de la base de datos (solo lectura) y se guarda en
caché: la ficha se actualiza sola cuando entran datos nuevos de la sonda.
"""
from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import projectdb as pdb

RES_ID = 2351
SOURCE = "che_saica_945"
VARS = {  # clave → (nombre, unidad, decimales)
    "phycocyanin": ("Ficocianina", "µg/L", 1),
    "water_temp": ("Temperatura del agua", "°C", 1),
    "turbidity": ("Turbidez", "NTU", 1),
    "ph": ("pH", "", 2),
}
HORIZONTES = [1, 3, 7]
_cache: dict = {}
_lock = threading.Lock()


def _cached(key: str, ttl: float, fn):
    with _lock:
        c = _cache.get(key)
        if c and time.time() - c[0] < ttl:
            return c[1]
    v = fn()
    with _lock:
        _cache[key] = (time.time(), v)
    return v


# ── Datos ───────────────────────────────────────────────────────────────────

def _raw(days: int) -> pd.DataFrame:
    """Lecturas crudas (sin datos marcados como erróneos) de los últimos `days` días de datos."""
    if pdb.is_mock():
        return _mock_raw(days)
    return pdb.q(f"""
      WITH ult AS (SELECT MAX(date_time::timestamptz) AS m FROM sensor_data
                   WHERE reservoir_id = {RES_ID} AND source_code = '{SOURCE}')
      SELECT (date_time::timestamptz AT TIME ZONE 'Europe/Madrid') AS t,
             phycocyanin, water_temp, turbidity, ph
      FROM sensor_data, ult
      WHERE reservoir_id = {RES_ID} AND source_code = '{SOURCE}' AND COALESCE(qc_flag, 0) < 2
        AND date_time::timestamptz > ult.m - INTERVAL '{int(days)} days'
      ORDER BY 1""")


def _daily() -> pd.DataFrame:
    """Media diaria de ficocianina (días con ≥ 12 lecturas), serie completa."""
    if pdb.is_mock():
        r = _mock_raw(3 * 365)
    else:
        r = pdb.q(f"""
          SELECT (date_time::timestamptz AT TIME ZONE 'Europe/Madrid') AS t, phycocyanin
          FROM sensor_data WHERE reservoir_id = {RES_ID} AND source_code = '{SOURCE}'
            AND COALESCE(qc_flag, 0) < 2 AND phycocyanin IS NOT NULL""")
    r["t"] = pd.to_datetime(r["t"])
    g = r.groupby(r["t"].dt.normalize())["phycocyanin"].agg(["mean", "count"])
    g = g[g["count"] >= 12]
    return pd.DataFrame({"pc": g["mean"]}).rename_axis("fecha").asfreq("D")


def _mock_raw(days: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    end = pd.Timestamp.now().floor("15min")
    t = pd.date_range(end - pd.Timedelta(days=days), end, freq="15min")
    doy = t.dayofyear.values
    season = np.exp(1.2 * np.cos(2 * np.pi * (doy - 250) / 365.25))
    noise = np.exp(0.5 * np.sin(t.dayofyear.values * 0.45 + t.year.values) + rng.normal(0, 0.05, len(t)))
    return pd.DataFrame({"t": t, "phycocyanin": 1.5 * season * noise,
                         "water_temp": 14 + 9 * np.cos(2 * np.pi * (doy - 210) / 365.25) + rng.normal(0, .3, len(t)),
                         "turbidity": 4 + rng.gamma(2, 1, len(t)), "ph": 8.1 + rng.normal(0, .05, len(t))})


# ── 1. Tiempo real ──────────────────────────────────────────────────────────

def live() -> dict:
    return _cached("live", 600, _live)


def _live() -> dict:
    r = _raw(8)
    if r is None or not len(r):
        return {"ok": False, "reservoir_id": RES_ID, "variables": []}
    r["t"] = pd.to_datetime(r["t"])
    ultimo = r["t"].max()
    out = []
    for k, (nombre, unidad, dec) in VARS.items():
        s = r[["t", k]].dropna()
        if not len(s):
            continue
        last = s.iloc[-1]
        prev = s[s["t"] <= last["t"] - pd.Timedelta(hours=24)]
        h = s.set_index("t")[k].resample("3h").mean().dropna()
        out.append({"key": k, "name": nombre, "unit": unidad, "decimals": dec,
                    "value": round(float(last[k]), 3), "time": last["t"].strftime("%Y-%m-%d %H:%M"),
                    "delta_24h": None if not len(prev) else round(float(last[k] - prev.iloc[-1][k]), 3),
                    "spark": [{"t": i.strftime("%Y-%m-%d %H:%M"), "v": round(float(v), 3)} for i, v in h.items()]})
    horas = (pd.Timestamp.now(tz="Europe/Madrid").tz_localize(None) - ultimo).total_seconds() / 3600
    res = {"ok": True, "reservoir_id": RES_ID, "station": "SAICA 945 · CHE", "last": ultimo.strftime("%Y-%m-%d %H:%M"),
           "hours_ago": round(horas, 1), "variables": out}
    try:
        res["risk_today"] = riesgo()["today"]
    except Exception:  # noqa: BLE001
        res["risk_today"] = None
    try:
        res["trend"] = tendencia()["forecast"]
    except Exception:  # noqa: BLE001
        res["trend"] = None
    return res


# ── 2. Riesgo estacional ────────────────────────────────────────────────────

def _X(doy) -> np.ndarray:
    doy = np.asarray(doy, float)
    return np.c_[np.ones_like(doy), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]


def _logit_fit(X, y, l2=1e-3, it=50):
    """Regresión logística por IRLS (sin sklearn en el servidor)."""
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _auc(y, s) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def riesgo() -> dict:
    return _cached("riesgo", 6 * 3600, _riesgo)


def _riesgo() -> dict:
    d = _daily().dropna()
    umbral = float(d["pc"].quantile(0.9))
    y = (d["pc"] >= umbral).values.astype(float)
    doy, yrs = d.index.dayofyear.values, d.index.year.values
    X = _X(doy)
    # validación: dejar fuera cada año
    p = np.full(len(d), np.nan)
    por_año = []
    for yr in np.unique(yrs):
        te, tr = yrs == yr, yrs != yr
        if tr.sum() < 100 or y[tr].sum() < 5:
            continue
        p[te] = 1 / (1 + np.exp(-X[te] @ _logit_fit(X[tr], y[tr])))
        por_año.append({"year": int(yr), "days": int(te.sum()), "peaks": int(y[te].sum()),
                        "auc": None if y[te].sum() < 3 else round(_auc(y[te], p[te]), 3)})
    ok = np.isfinite(p)
    auc = _auc(y[ok], p[ok])
    top = p[ok] >= np.quantile(p[ok], 0.75)
    captados = float(y[ok][top].sum() / max(y[ok].sum(), 1))
    w = _logit_fit(X, y)
    dias = np.arange(1, 367)
    curva = 1 / (1 + np.exp(-_X(dias) @ w))
    obs = pd.Series(y, index=doy).groupby(level=0).mean().reindex(dias)
    obs = obs.rolling(15, center=True, min_periods=3).mean()
    q = np.quantile(curva, [0.5, 0.8])
    nivel = lambda v: "alto" if v >= q[1] else "medio" if v >= q[0] else "bajo"  # noqa: E731
    hoy = date.today().timetuple().tm_yday
    alto = dias[curva >= q[1]]
    fmt = lambda x: f"{x:.2f}".replace(".", ",")  # noqa: E731
    fd = lambda n: (date(2025, 1, 1) + timedelta(days=int(n) - 1)).strftime("%d/%m")  # noqa: E731
    return {
        "id": "riesgo_pc_val", "kind": "seasonal", "index_id": "", "reservoir": "VAL", "reservoir_label": "El Val",
        "variable": "Riesgo de ficocianina alta", "unit": "probabilidad", "status": "validado",
        "version": pd.Timestamp.now().strftime("%Y-%m"),
        "description": "Probabilidad de que la ficocianina media del día supere su percentil 90 histórico, según la época del año. Aprendido de la sonda SAICA de la CHE; indica cuándo suele haber episodios, no si lo hay ahora (para eso, la sonda en tiempo real del Monitor).",
        "formula": f"P(pico) = 1 / (1 + e^−({w[0]:.3f} {w[1]:+.3f}·sen(día) {w[2]:+.3f}·cos(día)))".replace(".", ","),
        "index_formula": f"Pico = día con ficocianina media ≥ {fmt(umbral)} µg/L (percentil 90) · día = día del año",
        "training": {"pairs": int(len(d)), "period": f"{d.index.min():%Y-%m} – {d.index.max():%Y-%m}",
                     "ground_truth": "Sonda SAICA 945 de la CHE en El Val (ficocianina, media diaria)",
                     "matching": f"Días con al menos 12 lecturas válidas · {int(y.sum())} días de pico"},
        "validation": {
            "method": "Se ajusta con todos los años menos uno y se comprueba en el que queda fuera, año por año.",
            "metrics": [
                {"label": "AUC", "value": fmt(auc), "help": "Capacidad de ordenar los días de pico por encima del resto (0,5 = azar, 1 = perfecto)"},
                {"label": "Picos en el 25 % de días de más riesgo", "value": f"{captados * 100:.0f} %", "help": "Proporción de días de pico que caen en la cuarta parte del año marcada como de más riesgo"},
                {"label": "Umbral de pico", "value": f"{fmt(umbral)} µg/L", "help": "Percentil 90 de la ficocianina media diaria de la sonda"},
                {"label": "Años de datos", "value": str(len(np.unique(yrs))), "help": "Veranos distintos con los que se ha aprendido el calendario"},
            ],
            "compared": "Se probó a añadir meteorología (ERA5-Land: temperatura del aire, calentamiento, viento, radiación, lluvia), el NDCI de Sentinel-2 y el nivel del embalse: ninguno mejoró la validación con los años de sonda disponibles.",
            "by_year": por_año,
        },
        "limits": [
            "Es un calendario: dice cuándo suele haber picos en El Val, no si hay uno ahora.",
            "Aprendido con pocos veranos; un año atípico (p. ej. un pico fuera de época) no lo anticipa.",
            "Solo válido para El Val; en otros embalses la época de riesgo puede ser distinta.",
        ],
        "pairs": [],
        "curve": [{"doy": int(k), "p": round(float(v), 4), "obs": None if not np.isfinite(o) else round(float(o), 4)}
                  for k, v, o in zip(dias, curva, obs.values)],
        "levels": {"medio": round(float(q[0]), 4), "alto": round(float(q[1]), 4)},
        "high_season": f"{fd(alto.min())} – {fd(alto.max())}" if len(alto) else None,
        "today": {"doy": hoy, "p": round(float(curva[hoy - 1]), 4), "level": nivel(curva[hoy - 1])},
    }


# ── 3. Tendencia a corto plazo ──────────────────────────────────────────────

def _clim(d: pd.DataFrame, excluir=None) -> pd.Series:
    tr = d if excluir is None else d[d.index.year != excluir]
    c = np.log1p(tr["pc"]).groupby(tr.index.dayofyear).mean().reindex(range(1, 367))
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(31, center=True, min_periods=5).mean().iloc[15:-15]
    return c.interpolate(limit_direction="both")


def tendencia() -> dict:
    return _cached("tendencia", 3 * 3600, _tendencia)


def _tendencia() -> dict:
    d = _daily()
    lp = np.log1p(d["pc"])
    doy = d.index.dayofyear.values
    # backtest honesto: climatología sin el año que se predice
    filas, resid = [], {h: [] for h in HORIZONTES}
    for yr in sorted(set(d.index.year)):
        c = _clim(d, excluir=yr)
        for h in HORIZONTES:
            y = lp.shift(-h)
            dh = ((doy + h - 1) % 366) + 1
            pred = lp.values + c.loc[dh].values - c.loc[doy].values
            k = (d.index.year == yr) & np.isfinite(pred) & y.notna().values
            if k.sum() < 30:
                continue
            e = y.values[k] - pred[k]
            ep = y.values[k] - lp.values[k]
            resid[h] += list(e)
            yy = np.expm1(y.values[k])
            r2 = 1 - np.sum((np.expm1(pred[k]) - yy) ** 2) / np.sum((yy - yy.mean()) ** 2)
            skill = 1 - np.mean(e ** 2) / np.mean(ep ** 2)
            filas.append({"year": int(yr), "h": h, "n": int(k.sum()), "r2": round(float(r2), 3), "skill": round(float(skill), 3)})
    # pronóstico desde el último día con dato
    c = _clim(d)
    ult = d["pc"].dropna()
    t0 = ult.index.max()
    v0 = float(ult.iloc[-1])
    doy0 = t0.dayofyear
    fc = [{"h": 0, "date": t0.strftime("%Y-%m-%d"), "value": round(v0, 2), "lo": round(v0, 2), "hi": round(v0, 2)}]
    for h in HORIZONTES:
        dh = ((doy0 + h - 1) % 366) + 1
        m = np.log1p(v0) + c.loc[dh] - c.loc[doy0]
        lo, hi = (np.quantile(resid[h], [0.1, 0.9]) if len(resid[h]) > 30 else (0.0, 0.0))
        fc.append({"h": h, "date": (t0 + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                   "value": round(float(np.expm1(m)), 2), "lo": round(float(max(np.expm1(m + lo), 0)), 2),
                   "hi": round(float(np.expm1(m + hi)), 2)})
    hist = d["pc"].loc[t0 - pd.Timedelta(days=45):].dropna()
    bt = pd.DataFrame(filas)
    fmt = lambda x: f"{x:.2f}".replace(".", ",")  # noqa: E731

    def rango(h):
        s = bt[bt.h == h]["r2"] if len(bt) else pd.Series(dtype=float)
        return "–" if not len(s) else f"{fmt(s.min())} a {fmt(s.max())}"
    return {
        "id": "tendencia_pc_val", "kind": "trend", "index_id": "", "reservoir": "VAL", "reservoir_label": "El Val",
        "variable": "Tendencia de ficocianina a corto plazo", "unit": "µg/L", "status": "experimental",
        "version": pd.Timestamp.now().strftime("%Y-%m"),
        "description": "Valor esperado de ficocianina en 1, 3 y 7 días a partir del último dato de la sonda, corregido por cómo suele evolucionar en esa época del año. Método sencillo y transparente; sirve para ver hacia dónde va la serie, no para dar un valor exacto.",
        "formula": "log(1 + PC(t+h)) = log(1 + PC(t)) + clim(día + h) − clim(día)",
        "index_formula": "clim = media de log(1 + PC) de la sonda para cada día del año (suavizada ±15 días)",
        "training": {"pairs": int(d["pc"].notna().sum()), "period": f"{d.index.min():%Y-%m} – {d.index.max():%Y-%m}",
                     "ground_truth": "Sonda SAICA 945 de la CHE en El Val (ficocianina, media diaria)",
                     "matching": "Banda sombreada: rango en el que cayó el 80 % de los aciertos en años pasados"},
        "validation": {
            "method": "Para cada año, la climatología se calcula sin ese año y se comprueba cuánto se acerca la predicción al valor real de la sonda.",
            "metrics": [{"label": f"R² a {h} día{'s' if h > 1 else ''}", "value": rango(h),
                         "help": "Rango entre años del R² de la predicción (1 = perfecto, 0 = no mejor que la media)"} for h in HORIZONTES],
            "compared": "Se comparó con suponer «igual que hoy» (persistencia), con la climatología sola y con un modelo de gradient boosting con temperatura y turbidez de la sonda: ninguno fue mejor de forma estable entre años.",
            "by_year": filas,
        },
        "limits": [
            "Experimental: funciona razonablemente en años normales, pero en años atípicos falla a partir de 1–3 días.",
            "No anticipa el inicio de un episodio que aún no ha empezado; sigue la tendencia de la serie.",
            "Depende de que la sonda esté al día: si el último dato es antiguo, la predicción también.",
        ],
        "pairs": [],
        "history": [{"date": i.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for i, v in hist.items()],
        "forecast": fc,
    }


# ── 4. Laboratorio de calibración de ficocianina ────────────────────────────

def pares_pc() -> dict:
    """Pares índice de Sentinel-2 ↔ ficocianina de la sonda, para el laboratorio de la web."""
    return _cached("pares_pc", 6 * 3600, _pares_pc)


def _pares_pc() -> dict:
    if pdb.is_mock():
        rng = np.random.default_rng(3)
        t = pd.date_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=900), periods=120, freq="7D")
        doy = t.dayofyear.values
        pc = np.exp(1.1 * np.cos(2 * np.pi * (doy - 250) / 365.25) + rng.normal(0, .5, len(t)))
        d = pd.DataFrame({"date": t, "pci": 1.1 + rng.normal(0, .12, len(t)),
                          "tbda": rng.normal(1, .2, len(t)), "ci": rng.normal(0, .05, len(t)), "pc": pc})
    else:
        d = pdb.q(f"""
          SELECT c.date::date AS date,
                 COALESCE(c.pci, i.pci) AS pci, COALESCE(c.tbda, i.tbda) AS tbda, COALESCE(c.ci, i.ci) AS ci,
                 c.phycocyanin AS pc
          FROM calibration_pairs c
          LEFT JOIN indices_sentinel i ON i.date = c.date AND i.reservoir_id = c.reservoir_id
          WHERE c.reservoir_id = {RES_ID} AND c.phycocyanin IS NOT NULL
          ORDER BY 1""")
        if d is None:
            d = pd.DataFrame(columns=["date", "pci", "tbda", "ci", "pc"])
    d = d[d["pc"] > 0].copy()
    d["date"] = pd.to_datetime(d["date"])
    # NDCI es el PCI en otra escala: NDCI = (PCI - 1) / (PCI + 1)
    d["ndci"] = (d["pci"] - 1) / (d["pci"] + 1)
    filas = []
    for r in d.to_dict("records"):
        fila = {"date": r["date"].strftime("%Y-%m-%d"), "doy": int(r["date"].dayofyear),
                "year": int(r["date"].year), "pc": round(float(r["pc"]), 3)}
        for k in ("pci", "ndci", "tbda", "ci"):
            v = r.get(k)
            fila[k] = None if v is None or not np.isfinite(v) else round(float(v), 5)
        filas.append(fila)
    return {"reservoir_label": "El Val", "station": "SAICA 945 · CHE",
            "indices": [{"key": "pci", "name": "PCI (B5/B4)"}, {"key": "ndci", "name": "NDCI"},
                        {"key": "tbda", "name": "TBDA (tres bandas)"}, {"key": "ci", "name": "CI"}],
            "rows": filas}


def _ficha_lab() -> dict:
    p = pares_pc()
    r = p["rows"]
    años = sorted({x["year"] for x in r})
    return {
        "id": "lab_pc_val", "kind": "lab", "index_id": "", "reservoir": "VAL", "reservoir_label": "El Val",
        "variable": "Laboratorio: ficocianina y Sentinel-2", "unit": "µg/L", "status": "laboratorio",
        "version": pd.Timestamp.now().strftime("%Y-%m"),
        "description": "Espacio para probar por tu cuenta si algún índice de Sentinel-2 sirve para estimar la ficocianina en El Val. Cambia el índice, la forma del ajuste y los años, y compara el R² del ajuste con el R² validado dejando fuera un año.",
        "formula": "PC = f(índice) · la forma la eliges tú",
        "index_formula": "PCI = B5/B4 · NDCI = (B5 − B4)/(B5 + B4) = (PCI − 1)/(PCI + 1) · TBDA y CI, índices de tres bandas",
        "training": {"pairs": len(r), "period": f"{años[0]}–{años[-1]}" if años else "—",
                     "ground_truth": "Sonda SAICA 945 de la CHE (ficocianina a las 11:00, misma fecha que la imagen)",
                     "matching": "Un par por fecha con imagen despejada de Sentinel-2 y dato de sonda"},
        "validation": {
            "method": "El laboratorio calcula a la vez el R² del ajuste (con todos los datos) y el R² validado (ajustando sin un año y comprobando en ese año). Si el segundo baja al complicar el modelo, es sobreajuste.",
            "metrics": [], "compared": "",
        },
        "limits": [
            "No es un modelo publicado: es una herramienta para explorar.",
            "Sentinel-2 no tiene banda en 620 nm, donde absorbe la ficocianina; estos índices miden biomasa.",
            "Un R² de ajuste alto no significa nada si el validado no lo acompaña.",
        ],
        "pairs": [],
    }


def cards() -> list:
    out = []
    for fn in (riesgo, tendencia, _ficha_lab):
        try:
            out.append(fn())
        except Exception as e:  # noqa: BLE001
            print("[sonda_val] ficha no disponible:", e)
    return out
