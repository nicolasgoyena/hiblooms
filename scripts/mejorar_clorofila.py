"""
Mejor modelo posible de CLOROFILA-a en El Val con Sentinel-2 (HIBLOOMS).

Parte de lo que ya descargó validacion_sonda.py (resultados_validacion/s2_boya_todas.csv),
así que NO vuelve a consultar GEE: solo lee la sonda de la base local.

Qué prueba:
  1. Cómo emparejar (ventana ±1/2/3 h · filtro de agua estricto o relajado) → más pares sin
     meter ruido.
  2. Modelos, todos sobre log10(clorofila) y validados dejando fuera un AÑO:
       · modelo actual de la web (logística sobre NDCI, sin reajustar)  ← referencia
       · lineal y cuadrático sobre NDCI
       · mejor índice de la búsqueda exhaustiva (elegido DENTRO de cada pliegue)
       · Lasso con bandas + índices
       · bosque aleatorio y gradient boosting
  3. Métricas en µg/L: RMSE, MAE, sesgo, R² (log), AUC ≥10 y ≥20 µg/L y acierto de clase
     trófica (OCDE: <2,5 · 2,5–8 · 8–25 · >25 µg/L).
  4. Ajusta el mejor modelo SENCILLO (fórmula) con todos los datos y lo guarda para la web.

Uso:
    set "DATABASE_URL=postgresql://postgres:CLAVE@localhost:5432/unav_water_sampling"
    python scripts\\mejorar_clorofila.py
Resultados en resultados_clorofila/.
"""
from __future__ import annotations

import json
import os
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
from validacion_sonda import BANDAS, INDICES, auc, combinaciones, emparejar, leer_sonda  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "resultados_clorofila"
CLASES = [0, 2.5, 8, 25, np.inf]            # OCDE: oligo · meso · eutro · hipereutrófico
NOMBRES_CLASE = ["oligotrófico", "mesotrófico", "eutrófico", "hipereutrófico"]


def modelo_actual(ndci):
    """Chla_Val_cal de la web: 450 / (1 + exp(−7,14·(NDCI − 0,46)))."""
    return 450 / (1 + np.exp(-7.14 * (ndci - 0.46)))


def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    k = np.isfinite(y) & np.isfinite(p) & (p > 0)
    y, p = y[k], p[k]
    ly, lp = np.log10(np.clip(y, .01, None)), np.log10(np.clip(p, .01, None))
    ss = ((ly - ly.mean()) ** 2).sum()
    cy, cp = np.digitize(y, CLASES[1:-1]), np.digitize(p, CLASES[1:-1])
    return {"n": int(len(y)), "R2_log": round(1 - ((ly - lp) ** 2).sum() / ss, 3) if ss else np.nan,
            "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 2), "MAE": round(float(np.mean(np.abs(y - p))), 2),
            "sesgo": round(float(np.mean(p - y)), 2),
            "MAE_log(×)": round(float(10 ** np.mean(np.abs(ly - lp))), 2),   # error típico multiplicativo
            "AUC≥10": round(float(auc(y >= 10, p)), 3), "AUC≥20": round(float(auc(y >= 20, p)), 3),
            "clase_trófica_ok": f"{100 * np.mean(cy == cp):.0f}%"}


def pares(url: str, horas: float, agua_min: float) -> pd.DataFrame:
    s2 = pd.read_csv(ROOT / "resultados_validacion" / "s2_boya_todas.csv")
    s2["t"] = pd.to_datetime(s2["t"], format="ISO8601", utc=True)
    s2 = s2[(s2.agua >= agua_min) & (s2.nubes300 <= 0.01) & (s2.B4 > 0)].copy()
    for k, (_, f) in INDICES.items():
        s2[k] = f(s2)
    son = leer_sonda(url, sorted(set(s2.t.dt.strftime("%Y-%m-%d"))))
    m = emparejar(s2, son, horas)
    m = m[m.chl_sup.notna() & (m.chl_sup > 0)].copy()
    m["year"] = m.t.dt.year
    return m.reset_index(drop=True)


def validar_modelos(m: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LassoCV, LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    y = m.chl_sup.values
    ly = np.log10(y)
    X_all = pd.concat([m[BANDAS], m[list(INDICES)]], axis=1).replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(X_all.median())
    C = combinaciones(m)
    preds = {k: np.full(len(m), np.nan) for k in
             ["actual web (sin reajustar)", "lineal NDCI", "cuadrático NDCI", "mejor índice (lineal)",
              "mejor índice (cuadrático)", "Lasso bandas+índices", "bosque aleatorio", "gradient boosting"]}
    elegidos = []
    for yr in sorted(m.year.unique()):
        te, tr = (m.year == yr).values, (m.year != yr).values
        if te.sum() < 3 or tr.sum() < 20:
            continue
        nd = m[["NDCI"]].values
        preds["actual web (sin reajustar)"][te] = modelo_actual(m.NDCI.values[te])
        preds["lineal NDCI"][te] = 10 ** LinearRegression().fit(nd[tr], ly[tr]).predict(nd[te])
        q = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(nd[tr], ly[tr])
        preds["cuadrático NDCI"][te] = 10 ** q.predict(nd[te])
        # mejor índice elegido solo con el entrenamiento
        r = C[tr].rank().corrwith(pd.Series(ly[tr], index=C.index[tr]).rank()).dropna()
        best = r.abs().idxmax(); elegidos.append(best)
        xb = C[[best]].values
        preds["mejor índice (lineal)"][te] = 10 ** LinearRegression().fit(xb[tr], ly[tr]).predict(xb[te])
        qb = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(xb[tr], ly[tr])
        preds["mejor índice (cuadrático)"][te] = 10 ** qb.predict(xb[te])
        la = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=20000)).fit(X_all[tr], ly[tr])
        preds["Lasso bandas+índices"][te] = 10 ** la.predict(X_all[te])
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=1).fit(X_all[tr], ly[tr])
        preds["bosque aleatorio"][te] = 10 ** rf.predict(X_all[te])
        gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                           min_samples_leaf=10, random_state=0).fit(X_all[tr], ly[tr])
        preds["gradient boosting"][te] = 10 ** gb.predict(X_all[te])
    tabla = pd.DataFrame([{"modelo": k, **metricas(y, p)} for k, p in preds.items()])
    return tabla.sort_values("R2_log", ascending=False), preds, elegidos


def ajustar_final(m: pd.DataFrame, indice: str, grado: int) -> dict:
    C = combinaciones(m)
    x, ly = C[indice].values, np.log10(m.chl_sup.values)
    k = np.isfinite(x)
    coef = np.polyfit(x[k], ly[k], grado)          # log10(chl) = polinomio(índice)
    return {"indice": indice, "grado": grado, "coef_log10": [float(c) for c in coef],
            "formula": f"log10(Chl-a µg/L) = " + " + ".join(
                f"{c:.4g}·x^{grado - i}" if grado - i > 1 else (f"{c:.4g}·x" if grado - i == 1 else f"{c:.4g}")
                for i, c in enumerate(coef)),
            "rango_indice_entrenamiento": [float(np.nanmin(x)), float(np.nanmax(x))],
            "rango_chl_entrenamiento": [float(m.chl_sup.min()), float(m.chl_sup.max())], "n": int(k.sum())}


def figura(m, preds, nombres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(nombres), figsize=(4.4 * len(nombres), 4.2), squeeze=False)
    for ax, n in zip(axs[0], nombres):
        p = preds[n]; k = np.isfinite(p)
        ax.scatter(m.chl_sup[k], p[k], c=m.t.dt.month[k], cmap="twilight", s=20, edgecolor="k", linewidth=.3)
        lim = [0.3, max(m.chl_sup.max(), np.nanmax(p)) * 1.3]
        ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        for u in (2.5, 8, 25):
            ax.axvline(u, color="grey", lw=.5, ls=":"); ax.axhline(u, color="grey", lw=.5, ls=":")
        ax.set_xlabel("Clorofila sonda (µg/L)"); ax.set_ylabel("Predicha (validada por años)"); ax.set_title(n, fontsize=10)
    fig.suptitle("El Val · clorofila-a Sentinel-2 · líneas punteadas: límites de estado trófico (OCDE)")
    fig.savefig(OUT / "pred_vs_obs.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("Falta DATABASE_URL.")
    if not (ROOT / "resultados_validacion" / "s2_boya_todas.csv").exists():
        sys.exit("Ejecuta antes scripts\\validacion_sonda.py (hace falta resultados_validacion\\s2_boya_todas.csv).")
    OUT.mkdir(exist_ok=True)

    print("1/3 ¿Cómo emparejar? (ventana y filtro de agua)")
    variantes = []
    for horas in (1, 2, 3):
        for agua in (0.99, 0.75):
            m = pares(url, horas, agua)
            if len(m) < 30:
                continue
            ly = np.log10(m.chl_sup)
            from scipy import stats
            rho = stats.spearmanr(m.NDCI, ly)[0]
            variantes.append({"ventana_h": horas, "agua_min": agua, "n": len(m), "rho_NDCI": round(rho, 3)})
            print(f"   ±{horas} h · agua ≥{agua:.0%}: {len(m)} pares · rho NDCI {rho:.3f}")
    v = pd.DataFrame(variantes)
    v.to_csv(OUT / "variantes_emparejado.csv", index=False)
    # la más grande cuya correlación no empeore más de 0,02 respecto a la mejor
    ok = v[v.rho_NDCI >= v.rho_NDCI.max() - 0.02].sort_values("n", ascending=False).iloc[0]
    print(f"   → elegido ±{int(ok.ventana_h)} h, agua ≥{ok.agua_min:.0%} ({int(ok.n)} pares)")
    m = pares(url, float(ok.ventana_h), float(ok.agua_min))
    m.to_csv(OUT / "pares.csv", index=False)

    print("2/3 Modelos (validación dejando fuera cada año)…")
    tabla, preds, elegidos = validar_modelos(m)
    tabla.to_csv(OUT / "comparacion_modelos.csv", index=False)
    top_idx = pd.Series(elegidos).value_counts()
    print(tabla.to_string(index=False))
    print("\n   índice elegido en cada año:", dict(top_idx))

    print("3/3 Modelo final sencillo (fórmula para la web)…")
    sencillos = tabla[tabla.modelo.str.contains("NDCI|mejor índice")].iloc[0]
    if "mejor índice" in sencillos.modelo:
        indice = top_idx.index[0]
    else:
        indice = "[NDCI]"
    grado = 2 if "cuadrático" in sencillos.modelo else 1
    final = ajustar_final(m, indice, grado)
    final["validacion"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in sencillos.items()}
    final["emparejado"] = {"ventana_h": float(ok.ventana_h), "agua_min": float(ok.agua_min)}
    (OUT / "modelo_final.json").write_text(json.dumps(final, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    figura(m, preds, ["actual web (sin reajustar)", sencillos.modelo, tabla.iloc[0].modelo]
           if tabla.iloc[0].modelo != sencillos.modelo else ["actual web (sin reajustar)", sencillos.modelo])
    print(f"\n   Modelo sencillo: {sencillos.modelo} · {final['formula']}  (x = {indice})")
    print(f"\nListo → {OUT}")


if __name__ == "__main__":
    main()
