"""
Modelo final de clorofila-a de El Val para la web (HIBLOOMS).

Configuración elegida con exprimir_clorofila.py:
  · Sentinel-2 L1C (sin corrección atmosférica) · media 5×5 píxeles (≈100 m)
  · clorofila de la sonda en 0–3 m
  · log10(Chl) = a + b·NDCI + c·NDCI² + d·sen(2π·día/365,25) + e·cos(2π·día/365,25)

Lee resultados_exprimir/pares_variantes.csv (no consulta GEE), valida dejando fuera cada
año, ajusta con todos los datos y actualiza la ficha data/modelos/chla_val.json, de donde
el visor lee los coeficientes (hiblooms_core.py).

Uso:  python scripts\\ajustar_clorofila_final.py
"""
from __future__ import annotations

import json
import sys
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
from mejorar_clorofila import metricas  # noqa: E402

ROOT = HERE.parent
FICHA = ROOT / "data" / "modelos" / "chla_val.json"


def rasgos(ndci, doy, estacional=True, cuadratico=True):
    cols = [ndci, ndci ** 2] if cuadratico else [ndci, np.zeros_like(ndci)]
    if estacional:
        cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.c_[tuple(cols)]


def main():
    f = ROOT / "resultados_exprimir" / "pares_variantes.csv"
    if not f.exists():
        sys.exit("Falta resultados_exprimir\\pares_variantes.csv: ejecuta antes exprimir_clorofila.py")
    m = pd.read_csv(f)
    m["t"] = pd.to_datetime(m["t"], format="ISO8601", utc=True)
    b4, b5 = m["toa_5x5_B4"], m["toa_5x5_B5"]
    m["ndci"] = (b5 - b4) / (b5 + b4)
    y = m["chl_0-3"]
    k = (m["agua_5x5"] >= 0.75) & (m["nubes300"] <= 0.01) & np.isfinite(m.ndci) & np.isfinite(y) & (y > 0)
    m = m[k].reset_index(drop=True)
    y, ly = m["chl_0-3"].values, np.log10(m["chl_0-3"].values)
    doy, yrs = m.t.dt.dayofyear.values, m.t.dt.year.values
    print(f"{len(m)} pares (L1C · 5×5 · sonda 0–3 m) · {yrs.min()}–{yrs.max()}")

    from sklearn.linear_model import LinearRegression
    lo, hi = float(m.ndci.min()), float(m.ndci.max())
    grid = np.linspace(lo, hi, 50)
    formas = {"cuadrático + estacionalidad": (True, True), "cuadrático": (True, False),
              "lineal + estacionalidad": (False, True), "lineal": (False, False)}
    res, preds, fits = {}, {}, {}
    for nombre, (cuad, est) in formas.items():
        X = rasgos(m.ndci.values, doy, est, cuad)
        p = np.full(len(m), np.nan)
        for yr in np.unique(yrs):
            te, tr = yrs == yr, yrs != yr
            if te.sum() < 3 or tr.sum() < 20:
                continue
            p[te] = LinearRegression().fit(X[tr], ly[tr]).predict(X[te])
        lr = LinearRegression().fit(X, ly)
        b, c = lr.coef_[0], lr.coef_[1]
        monotono = bool(np.all(b + 2 * c * grid > 0))       # la clorofila sube siempre con el NDCI
        preds[nombre], fits[nombre] = 10 ** p, (lr, est)
        res[nombre] = {**metricas(y, 10 ** p), "sube_siempre": monotono, "complejidad": int(cuad) + int(est)}
    t = pd.DataFrame(res).T
    print(t[["n", "R2_log", "MAE_log(×)", "AUC≥10", "AUC≥20", "clase_trófica_ok", "sube_siempre"]].to_string())
    val = t[t.sube_siempre]
    if val.empty:
        sys.exit("Ninguna forma es monótona en el rango de calibración: revisa los datos.")
    mejor = val.R2_log.astype(float).max()
    elegido = val[val.R2_log.astype(float) >= mejor - 0.01].sort_values("complejidad").index[0]
    print(f"\nElegido: {elegido} (sube siempre con el NDCI; entre las válidas, la más sencilla a ≤0,01 de R² de la mejor)")
    lr, est = fits[elegido]
    a = float(lr.intercept_); b, c = float(lr.coef_[0]), float(lr.coef_[1])
    d, e = (float(lr.coef_[2]), float(lr.coef_[3])) if est else (0.0, 0.0)
    es = lambda v, n=3: f"{v:+.{n}f}".replace(".", ",").replace("+", "+ ").replace("-", "− ")  # noqa: E731
    formula = f"log10(Chl-a) = {a:.3f}".replace(".", ",") + f" {es(b)}·NDCI"
    if abs(c) > 1e-9:
        formula += f" {es(c)}·NDCI²"
    if est:
        formula += f" {es(d)}·sen(día) {es(e)}·cos(día)"
    print("Modelo final:", formula)
    res = {"elegido": res[elegido]}
    preds = {"elegido": preds[elegido]}

    card = json.loads(FICHA.read_text(encoding="utf-8"))
    fila = res["elegido"]
    v = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")  # noqa: E731
    card.update({
        "version": pd.Timestamp.now().strftime("%Y-%m"),
        "description": "Estima la concentración de clorofila-a (biomasa algal total) en los primeros 3 m del embalse a partir del NDCI de Sentinel-2" + (" y la época del año." if est else "."),
        "formula": formula,
        "index_formula": "NDCI = (B5 − B4) / (B5 + B4) sobre Sentinel-2 L1C, media en 5×5 píxeles (≈100 m)" + (" · día = día del año de la imagen" if est else ""),
        "coef": {"intercept": a, "ndci": b, "ndci2": c, "sin_doy": d, "cos_doy": e,
                 "fuente": "L1C", "ventana_m": 50, "ndci_rango": [float(m.ndci.min()), float(m.ndci.max())]},
    })
    card["training"].update({
        "pairs": int(len(m)), "period": f"{yrs.min()}–{yrs.max()}",
        "ground_truth": "Sonda de perfiles Aquadam de la CHE (clorofila, media 0–3 m)",
        "matching": "Imagen Sentinel-2 L1C y sonda en ±3 h · media de 5×5 píxeles alrededor de la boya · solo agua sin nubes a 300 m",
    })
    card["validation"]["metrics"] = [
        {"label": "Error típico", "value": f"×{v(fila['MAE_log(×)'], 1)}", "help": "La predicción suele estar dentro de ese factor del valor real"},
        {"label": "R² (escala log)", "value": v(fila["R2_log"]), "help": "Proporción de la variación explicada"},
        {"label": "AUC ≥ 10 µg/L", "value": v(fila["AUC≥10"]), "help": "Capacidad de distinguir episodios de más de 10 µg/L (0,5 = azar, 1 = perfecto)"},
        {"label": "AUC ≥ 20 µg/L", "value": v(fila["AUC≥20"]), "help": "Capacidad de distinguir episodios de más de 20 µg/L"},
        {"label": "Clase trófica", "value": str(fila["clase_trófica_ok"]), "help": "Aciertos de clase OCDE (oligo · meso · eutro · hipereutrófico)"},
    ]
    card["validation"]["compared"] = (
        "Se compararon 108 configuraciones (corrección atmosférica L2A/L1C, ventanas de 1, 3×3 y 5×5 píxeles, "
        "capas de sonda, filtros de imagen) y modelos lineales, cuadráticos, Lasso, bosque aleatorio y gradient boosting. "
        "La reflectancia L1C sin corregir con media 5×5 fue la más robusta; ningún modelo de aprendizaje automático mejoró a esta fórmula.")
    card["limits"] = [
        "Estima la biomasa algal total: no distingue cianobacterias de otras algas.",
        "Representa la media de los primeros 3 m, no la clorofila de la superficie estricta.",
        "Calibrado en El Val: en otros embalses es orientativo hasta validarlo con sus datos.",
    ]
    p = preds["elegido"]; kk = np.isfinite(p)
    card["pairs"] = [{"date": t.strftime("%Y-%m-%d"), "obs": round(float(o), 2), "pred": round(float(q), 2), "ndci": round(float(n), 4)}
                     for t, o, q, n in zip(m.t[kk], y[kk], p[kk], m.ndci[kk])]
    FICHA.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nFicha actualizada: {FICHA.relative_to(ROOT)} — el visor usará estos coeficientes.")


if __name__ == "__main__":
    main()
