"""
Satellite ↔ in situ calibration for the React viewer.

Satellite extraction: instead of 4–6 GEE calls per image, the 12 indices + cloud cover +
coverage of ALL images in the period are computed on the GEE server and collected with a
SINGLE call.
"""
from __future__ import annotations

import io
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Candidate indices used as predictors (same band names as hiblooms_core._add_calibration_feature_bands)
CANDIDATE_INDICES = [
    "NDCI_705_665", "MCI_705", "R705_R665", "R740_R665", "R783_R665", "TB_740",
    "TB_783", "NDRE_783_705", "NDRE_740_705", "B5_B4_diff", "B6_B5_diff", "B7_B5_diff",
]
MODEL_NAMES = ["linear", "ridge", "lasso", "elastic_net", "logistic", "poly2", "svr_rbf", "random_forest", "gradient_boosting"]
RASTERIZABLE = ["linear", "ridge", "lasso", "elastic_net", "logistic"]
S2 = "COPERNICUS/S2_SR_HARMONIZED"


# ── Reading the in situ CSV ─────────────────────────────────────────────────
def read_csv_text(text: str) -> pd.DataFrame:
    """Read a CSV with , ; or tab separator and . or , decimal mark."""
    best, best_score = None, -1
    for sep, dec in ((",", "."), (";", ","), (";", "."), ("\t", "."), ("\t", ","), (None, ".")):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, decimal=dec, engine="python")
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        score = df.shape[1] * 10 + sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        if score > best_score:
            best, best_score = df, score
    if best is None:
        raise ValueError("No se pudo leer el CSV (prueba con separador , o ;)")
    return best


def csv_preview(text: str) -> Dict[str, Any]:
    df = read_csv_text(text)
    cols = [str(c) for c in df.columns]
    numeric = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.6]
    guess = lambda keys: next((c for c in cols if c.strip().lower() in keys), None)
    return {
        "columns": cols,
        "numeric": numeric,
        "n_rows": int(len(df)),
        "guess": {
            "date": guess({"date", "fecha", "datetime", "fecha_hora", "date_time", "timestamp"}),
            "time": guess({"time", "hora"}),
            "site": guess({"punto", "estacion", "estación", "station", "site", "point", "punto_muestreo", "id_punto", "sample_point", "muestra_punto"}),
            "lat": guess({"lat", "latitud", "latitude", "y"}),
            "lon": guess({"lon", "lng", "long", "longitud", "longitude", "x"}),
        },
        "head": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }


def to_insitu_frame(text: str, date_col: str, time_col: Optional[str], value_col: str, default_time: str = "11:00",
                    tz: str = "Europe/Madrid", site_col: Optional[str] = None, lat_col: Optional[str] = None,
                    lon_col: Optional[str] = None, start_hour: int = 0, end_hour: int = 23) -> pd.DataFrame:
    """In situ CSV → DataFrame [datetime(UTC), date, value, site, lat, lon].
    Times are interpreted in `tz` (local time of the probe) and converted to UTC (Sentinel-2 reference).
    If there is a site column and/or lat-lon, each row keeps its location."""
    df = read_csv_text(text)
    for c in [date_col, value_col] + [x for x in (time_col, site_col, lat_col, lon_col) if x]:
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no está en el CSV")
    dt = pd.to_datetime(df[date_col].astype(str), errors="coerce", dayfirst=True)
    if time_col:
        # "11:00", "11:00:00", "11.00", "1100" or "2024-03-01 11:00" → HH:MM:SS (works with pandas 1.x and 2.x)
        hm = df[time_col].astype(str).str.extract(r"(\d{1,2})[:.h]?(\d{2})(?:[:.](\d{2}))?\s*$")
        hhmmss = hm[0].str.zfill(2) + ":" + hm[1] + ":" + hm[2].fillna("00")
        dt = pd.to_datetime(dt.dt.strftime("%Y-%m-%d") + " " + hhmmss, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    elif (dt.dt.hour + dt.dt.minute).fillna(0).sum() == 0:
        dt = pd.to_datetime(dt.dt.strftime("%Y-%m-%d") + " " + default_time, errors="coerce")
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    out = pd.DataFrame({"local": dt, "value": num(value_col)})
    out["site"] = df[site_col].astype(str).str.strip() if site_col else "__single"
    out["lat"] = num(lat_col) if lat_col else np.nan
    out["lon"] = num(lon_col) if lon_col else np.nan
    out = out.dropna(subset=["local", "value"])
    out = out[(out["local"].dt.hour >= start_hour) & (out["local"].dt.hour <= end_hour)]
    if tz and tz.upper() != "UTC":
        out["datetime"] = out["local"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    else:
        out["datetime"] = out["local"].dt.tz_localize("UTC")
    out = out.dropna(subset=["datetime"])
    if out.empty:
        ej = df[[c for c in (date_col, time_col, value_col) if c]].head(3).to_dict(orient="records")
        raise ValueError(f"No quedan filas válidas tras leer fecha/hora y valor. Primeras filas leídas: {ej}")
    out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    return out.drop(columns="local").reset_index(drop=True)


def resolve_sites(insitu: pd.DataFrame, pois: List[dict], single_point: Optional[dict]) -> tuple[pd.DataFrame, List[dict]]:
    """Assign coordinates to each row. Returns (insitu with a site column, list of sites {site, lat, lon})."""
    df = insitu.copy()
    has_ll = df["lat"].notna().any() and df["lon"].notna().any()
    multi = df["site"].nunique() > 1 or df["site"].iloc[0] != "__single"
    if has_ll:
        if not multi:
            df["site"] = df["lat"].round(5).astype(str) + "," + df["lon"].round(5).astype(str)
        coords = df.dropna(subset=["lat", "lon"]).groupby("site")[["lat", "lon"]].mean()
        df = df[df["site"].isin(coords.index)]
        sites = [{"site": s, "lat": float(r.lat), "lon": float(r.lon)} for s, r in coords.iterrows()]
    elif multi:
        lut = {str(p["name"]).strip().lower(): p for p in pois}
        unknown = sorted({s for s in df["site"].unique() if s.lower() not in lut})
        if unknown:
            raise ValueError(f"Puntos del CSV sin coordenadas: {', '.join(unknown[:8])}{'…' if len(unknown) > 8 else ''}. "
                             "Añádelos como puntos de interés del embalse o incluye columnas de latitud y longitud.")
        sites = [{"site": s, "lat": float(lut[s.lower()]["lat"]), "lon": float(lut[s.lower()]["lon"])} for s in sorted(df["site"].unique())]
    else:
        name = (single_point or {}).get("name") or "embalse"
        df["site"] = name
        sites = [{"site": name, "lat": single_point["lat"], "lon": single_point["lon"]}] if single_point else [{"site": name, "lat": None, "lon": None}]
    return df.reset_index(drop=True), sites


# ── Satellite extraction in ONE call ─────────────────────────────────────────
def _add_indices(img):
    """The 12 candidate indices (same formulas as hiblooms_core._add_calibration_feature_bands)."""
    b4 = img.select("B4").toFloat().divide(10000)
    b5 = img.select("B5").toFloat().divide(10000)
    b6 = img.select("B6").toFloat().divide(10000)
    b7 = img.select("B7").toFloat().divide(10000)
    return img.addBands([
        b5.subtract(b4).divide(b5.add(b4)).rename("NDCI_705_665"),
        b5.subtract(b4.add(b6.subtract(b4).multiply((705 - 665) / (740 - 665)))).rename("MCI_705"),
        b5.divide(b4).rename("R705_R665"),
        b6.divide(b4).rename("R740_R665"),
        b7.divide(b4).rename("R783_R665"),
        b6.subtract(b5).rename("TB_740"),
        b7.subtract(b5).rename("TB_783"),
        b7.subtract(b5).divide(b7.add(b5)).rename("NDRE_783_705"),
        b6.subtract(b5).divide(b6.add(b5)).rename("NDRE_740_705"),
        b5.subtract(b4).rename("B5_B4_diff"),
        b6.subtract(b5).rename("B6_B5_diff"),
        b7.subtract(b5).rename("B7_B5_diff"),
    ])


POINT_BUFFER_M = 45  # radius around the probe (≈ 3×3–4×4 pixels of 20 m)


SHORE_BUFFER_M = 20  # inward margin from the shoreline (mixed pixels / adjacency)


def clear_water_mask(scl, inner):
    """Water = inside the reservoir (>20 m from the shore) and clear pixel.
    SCL = 6 is NOT required: Sen2Cor often classifies intense blooms/scums as
    vegetation (4) or bare soil (5), which would remove exactly the peak pixels.
    Excluded: no data/defective (0,1), cloud shadows (3), clouds (8,9), cirrus (10), snow (11)."""
    import ee
    bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return bad.Not().And(ee.Image.constant(1).clip(inner).mask())


def satellite_features(aoi, sites: List[dict], dates: List[str], window_days: int, scale_m: int = 20) -> List[dict]:
    """ONE GEE call: for each image and site, the 12 indices averaged over clear water
    (shrunken reservoir mask, not SCL=6) within ±45 m of the point (or over the whole reservoir),
    the clear-water fraction in that area and the SCL cloud cover over the reservoir."""
    import ee
    inner = aoi.buffer(-SHORE_BUFFER_M)
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([s["lon"], s["lat"]]).buffer(POINT_BUFFER_M) if s["lat"] is not None else inner, {"site": s["site"]})
        for s in sites])
    d0 = (pd.to_datetime(min(dates)) - timedelta(days=window_days + 1)).strftime("%Y-%m-%d")
    d1 = (pd.to_datetime(max(dates)) + timedelta(days=window_days + 2)).strftime("%Y-%m-%d")

    def per_image(img):
        scl = img.select("SCL")
        water = clear_water_mask(scl, inner)
        cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        cloud = cloud_mask.multiply(100).clip(aoi).rename("c").reduceRegion(ee.Reducer.mean(), aoi, scale_m, maxPixels=1e13).get("c")
        stack = _add_indices(img).select(CANDIDATE_INDICES).updateMask(water).addBands(water.unmask(0).rename("wfrac"))
        red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale_m)
        return red.map(lambda f: f.setGeometry(None).set({"image_id": img.id(), "t": img.get("system:time_start"), "cloud": cloud}))

    col = ee.ImageCollection(S2).filterBounds(fc.geometry()).filterDate(d0, d1).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 95))
    return [f["properties"] for f in ee.FeatureCollection(col.map(per_image)).flatten().getInfo()["features"]]


def mock_satellite_features(insitu: pd.DataFrame) -> List[dict]:
    """Demo mode: one "image" every 5 days; indices linked NON-linearly to each site's in situ value."""
    rng = random.Random(42)
    rows = []
    t_all = pd.date_range(insitu["datetime"].min().normalize(), insitu["datetime"].max().normalize(), freq="5D")
    for t in t_all:
        t = t + pd.Timedelta(hours=10, minutes=55)
        cloud = rng.uniform(0, 40)
        for site, g in insitu.groupby("site"):
            near = g[(g["datetime"] >= t - pd.Timedelta(hours=3)) & (g["datetime"] <= t + pd.Timedelta(hours=3))]
            if near.empty:
                continue
            x = np.log1p(float(near["value"].mean())) / 5
            p = {"site": site, "image_id": f"DEMO_{t:%Y%m%d}", "t": int(t.timestamp() * 1000), "cloud": cloud, "wfrac": rng.uniform(0.6, 1)}
            for k, idx in enumerate(CANDIDATE_INDICES):
                p[idx] = 0.9 + (0.8 - 0.05 * k) * x + rng.gauss(0, 0.04 + 0.02 * k)
            rows.append(p)
    return rows


# ── Satellite ↔ in situ matching ─────────────────────────────────────────────
def _build_pairs_site(insitu: pd.DataFrame, rows: List[dict], max_cloud: float, min_water: float,
                max_hours: float, window_days: int) -> tuple[pd.DataFrame, Dict[str, int]]:
    """One row per valid IMAGE (no duplicates). y = mean of in situ measurements within ±max_hours
    of the satellite overpass; if none and window_days > 0, the closest measurement within ±window_days.
    Each in situ day is used at most once (with the closest image)."""
    imgs = []
    for p in rows:
        if p.get("t") is None:
            continue
        cloud = p.get("cloud"); wf = p.get("wfrac")
        imgs.append({**p, "dt": pd.Timestamp(p["t"], unit="ms", tz="UTC"),
                     "cloud": float(cloud) if cloud is not None else None,
                     "wfrac": float(wf) if wf is not None else 0.0})
    stats = {"images": len(imgs)}
    ok = [i for i in imgs if i["cloud"] is not None and i["cloud"] <= max_cloud and i["wfrac"] >= min_water
          and all(i.get(k) is not None for k in ("R705_R665", "NDCI_705_665"))]
    stats["images_valid"] = len(ok)
    best_by_day: Dict[str, dict] = {}
    for i in ok:  # one image per day (the least cloudy)
        d = i["dt"].strftime("%Y-%m-%d")
        if d not in best_by_day or i["cloud"] < best_by_day[d]["cloud"]:
            best_by_day[d] = i

    s = insitu.sort_values("datetime")
    times = s["datetime"].values.astype("datetime64[ns]")
    vals = s["value"].values
    out = []
    for d, i in best_by_day.items():
        t = np.datetime64(i["dt"].tz_convert(None))
        dh = np.abs((times - t) / np.timedelta64(1, "h"))
        m = dh <= max_hours
        if m.any():
            y, n, lag_h = float(np.mean(vals[m])), int(m.sum()), float(np.mean(dh[m]))
        elif window_days > 0 and (dh <= window_days * 24 + max_hours).any():
            j = int(np.argmin(dh))
            same = np.abs((times - times[j]) / np.timedelta64(1, "h")) <= max_hours
            y, n, lag_h = float(np.mean(vals[same])), int(same.sum()), float(dh[j])
        else:
            continue
        row = {"date": d, "image_id": i["image_id"], "overpass_utc": i["dt"].strftime("%Y-%m-%d %H:%M"),
               "y": y, "n_obs": n, "lag_h": round(lag_h, 2), "cloud": i["cloud"], "wfrac": i["wfrac"]}
        row.update({k: i.get(k) for k in CANDIDATE_INDICES})
        out.append(row)
    df = pd.DataFrame(out)
    if not df.empty and window_days > 0:  # each in situ measurement with a single image
        df["_key"] = df["y"].round(6).astype(str) + "_" + df["n_obs"].astype(str)
        df = df.sort_values("lag_h").drop_duplicates("_key").drop(columns="_key")
    df = df.sort_values("date").reset_index(drop=True) if not df.empty else df
    stats["pairs"] = int(len(df))
    return df, stats


def build_pairs(insitu: pd.DataFrame, rows: List[dict], max_cloud: float, min_water: float,
                max_hours: float, window_days: int) -> tuple[pd.DataFrame, Dict[str, int]]:
    parts, tot = [], {"images": 0, "images_valid": 0, "pairs": 0}
    for site, g in insitu.groupby("site"):
        r = [x for x in rows if x.get("site") == site]
        df, st = _build_pairs_site(g, r, max_cloud, min_water, max_hours, window_days)
        for k in tot:
            tot[k] += st.get(k, 0)
        if not df.empty:
            parts.append(df.assign(site=site))
    tot["images"] = len({x.get("image_id") for x in rows})
    pairs = pd.concat(parts, ignore_index=True).sort_values(["date", "site"]).reset_index(drop=True) if parts else pd.DataFrame()
    tot["pairs"] = int(len(pairs)); tot["sites"] = len(parts)
    return pairs, tot


# ── Modelos candidatos ──────────────────────────────────────────────────────
MODEL_LABELS = {"linear": "Lineal", "ridge": "Ridge", "lasso": "Lasso", "elastic_net": "ElasticNet",
                "logistic": "Logística", "poly2": "Polinómico 2", "svr_rbf": "SVR", "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting"}
RASTER_MODELS = {"linear", "ridge", "lasso", "elastic_net", "logistic"}
ALL_MODELS = list(MODEL_LABELS)
AUTO_POOLS = {
    "phycocyanin": ["R705_R665", "NDCI_705_665", "R740_R665", "MCI_705", "B5_B4_diff", "NDRE_740_705"],
    "chlorophyll": ["NDCI_705_665", "R705_R665", "MCI_705", "NDRE_740_705", "NDRE_783_705", "TB_740"],
    "other": list(CANDIDATE_INDICES),
}


class Cand:
    """(predictors, model, transformation). transform='log' fits log(1+y) and back-transforms with exp(·)−1."""

    def __init__(self, preds, model, transform="none"):
        self.preds, self.model, self.transform = tuple(preds), model, transform
        self.est = None; self.params = None; self.smear = 1.0

    @property
    def key(self):
        return f"{self.model}|{self.transform}|{'+'.join(self.preds)}"

    def _make(self):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        imp = ("imputer", SimpleImputer(strategy="median"))
        sc = ("scaler", StandardScaler())
        m = self.model
        if m == "linear":
            return Pipeline([imp, ("model", LinearRegression())])
        if m == "ridge":
            return Pipeline([imp, sc, ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        if m == "lasso":
            return Pipeline([imp, sc, ("model", LassoCV(cv=3, max_iter=20000, random_state=0))])
        if m == "elastic_net":
            return Pipeline([imp, sc, ("model", ElasticNetCV(cv=3, l1_ratio=[.2, .5, .8], max_iter=20000, random_state=0))])
        if m == "poly2":
            return Pipeline([imp, ("poly", PolynomialFeatures(2, include_bias=False)), sc, ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        if m == "svr_rbf":
            return Pipeline([imp, sc, ("model", SVR(C=10, epsilon=0.05))])
        if m == "random_forest":
            return Pipeline([imp, ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0))])
        if m == "gradient_boosting":
            return Pipeline([imp, ("model", GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0))])
        raise ValueError(m)

    def fit(self, df):
        X = df[list(self.preds)].astype(float); y = df["y"].values.astype(float)
        yt = np.log1p(np.clip(y, 0, None)) if self.transform == "log" else y
        if self.model == "logistic":
            from scipy.optimize import curve_fit
            x = X.iloc[:, 0].fillna(X.iloc[:, 0].median()).values
            ymax = max(float(np.max(y)), 1e-6); sx = float(np.std(x)) or 1.0
            f = lambda xx, L, k, x0: L / (1 + np.exp(-np.clip(k * (xx - x0), -50, 50)))
            try:
                p, _ = curve_fit(f, x, y, p0=[ymax * 1.2, 4 / sx, float(np.median(x))],
                                 bounds=([ymax * 0.3, -200 / sx, x.min() - 3 * sx], [ymax * 20, 200 / sx, x.max() + 3 * sx]), maxfev=20000)
                self.params = {"L": float(p[0]), "k": float(p[1]), "x0": float(p[2]), "fill": float(np.median(x))}
            except Exception:
                self.params = {"L": ymax, "k": 0.0, "x0": float(np.median(x)), "fill": float(np.median(x))}
            return self
        self.est = self._make().fit(X, yt)
        self.smear = 1.0
        if self.transform == "log":  # Duan (1983) smearing correction: E[y] = exp(ŷ_log)·mean(exp(residuals)) − 1
            res = yt - self.est.predict(X)
            self.smear = float(np.clip(np.mean(np.exp(res)), 1.0, 5.0))
        self.resid_q = None
        return self

    def predict(self, df):
        X = df[list(self.preds)].astype(float)
        if self.model == "logistic":
            p = self.params; x = X.iloc[:, 0].fillna(p["fill"]).values
            return p["L"] / (1 + np.exp(-np.clip(p["k"] * (x - p["x0"]), -50, 50)))
        yp = self.est.predict(X)
        if self.transform == "log":
            yp = np.exp(np.clip(yp, -10, 12)) * getattr(self, "smear", 1.0) - 1
        return np.clip(yp, 0, None)

    def raster(self):
        """Config to apply the model pixel by pixel in Earth Engine (or None if not possible)."""
        if self.model not in RASTER_MODELS:
            return None
        if self.model == "logistic":
            return {"type": "logistic", "predictor": self.preds[0], **self.params}
        steps = self.est.named_steps
        reg = steps["model"]
        coef = np.asarray(reg.coef_, dtype=float).ravel(); b = float(np.asarray(reg.intercept_).ravel()[0])
        if "scaler" in steps:
            sc = steps["scaler"]; s = np.where(sc.scale_ == 0, 1, sc.scale_)
            b = b - float(np.sum(coef * sc.mean_ / s)); coef = coef / s
        fill = [float(v) for v in steps["imputer"].statistics_]
        return {"type": "linear", "transform": self.transform, "smear": float(getattr(self, "smear", 1.0)), "predictors": list(self.preds),
                "intercept": b, "coefficients": [float(c) for c in coef], "fill_values": fill}

    def describe(self):
        t = "log · " if self.transform == "log" else ""
        return f"{t}{MODEL_LABELS.get(self.model, self.model)}"


# ── Temporal block validation (nested) ──────────────────────────────────────
def _blocks(df: pd.DataFrame, k: int) -> List[np.ndarray]:
    """Contiguous temporal blocks. All samples from the same day (even from different
    sites) fall in the same block, so the model never sees that day when evaluating it."""
    days = np.array(sorted(df["date"].unique()))
    out = []
    for chunk in np.array_split(days, min(k, len(days))):
        idx = np.where(df["date"].isin(chunk).values)[0]
        if len(idx):
            out.append(idx)
    return out


def _metrics(y, yp, thr: float) -> Dict[str, Optional[float]]:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y = np.asarray(y, float); yp = np.asarray(yp, float)
    ok = np.isfinite(yp)
    y, yp = y[ok], yp[ok]
    if len(y) < 3:
        return {"r2": None, "rmse": None, "mae": None, "r2_log": None, "sens": None, "far": None, "n": int(len(y))}
    pos, ppos = y > thr, yp > thr
    tp, fn, fp, tn = int((pos & ppos).sum()), int((pos & ~ppos).sum()), int((~pos & ppos).sum()), int((~pos & ~ppos).sum())
    return {
        "r2": float(r2_score(y, yp)), "rmse": float(np.sqrt(mean_squared_error(y, yp))), "mae": float(mean_absolute_error(y, yp)),
        "r2_log": float(r2_score(np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(yp, 0, None)))),
        "sens": tp / (tp + fn) if tp + fn else None, "far": fp / (fp + tn) if fp + tn else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "n": int(len(y)),
    }


CRITERIA = {"balanced": "Equilibrado (R² y R² log)", "peaks": "Precisión en picos (R²)",
            "general": "Precisión general (R² log)", "alerts": "Detección de alertas (F1)"}


def _score(y, yp, criterion: str, thr: float) -> float:
    from sklearn.metrics import r2_score
    y = np.asarray(y, float); yp = np.asarray(yp, float)
    r2 = r2_score(y, yp)
    r2l = r2_score(np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(yp, 0, None)))
    if criterion == "peaks":
        return float(r2)
    if criterion == "general":
        return float(r2l)
    if criterion == "alerts":
        pos, pp = y > thr, yp > thr
        tp, fp, fn = (pos & pp).sum(), (~pos & pp).sum(), (pos & ~pp).sum()
        return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return float(0.5 * r2 + 0.5 * r2l)


def _cv_score(cand: Cand, df: pd.DataFrame, k: int, criterion: str = "balanced", thr: float = 0.0) -> tuple[float, np.ndarray]:
    """Temporal cross-validation score over contiguous blocks (no shuffling)."""
    oof = np.full(len(df), np.nan)
    for b in _blocks(df, k):
        tr = np.setdiff1d(np.arange(len(df)), b)
        if len(tr) < 5:
            continue
        try:
            c = Cand(cand.preds, cand.model, cand.transform).fit(df.iloc[tr])
            oof[b] = c.predict(df.iloc[b])
        except Exception:
            return -np.inf, oof
    ok = np.isfinite(oof)
    if ok.sum() < 5:
        return -np.inf, oof
    return _score(df["y"].values[ok], oof[ok], criterion, thr), oof


def _select(cands: List[Cand], df: pd.DataFrame, k: int, parsimony: float = 0.01, criterion: str = "balanced", thr: float = 0.0):
    scored = []
    for c in cands:
        r2, _ = _cv_score(c, df, k, criterion, thr)
        scored.append((r2 - parsimony * (len(c.preds) - 1), r2, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def candidates_auto(kind: str) -> List[Cand]:
    from itertools import combinations
    pool = AUTO_POOLS.get(kind, AUTO_POOLS["other"])
    max_k = 2 if kind == "other" else 3
    out = []
    for kk in range(1, max_k + 1):
        for combo in combinations(pool, kk):
            out += [Cand(combo, "linear", "none"), Cand(combo, "linear", "log")]
    out += [Cand((p,), "logistic", "none") for p in pool]
    return out


def candidates_expert(preds: List[str], models: List[str], transform: str) -> List[Cand]:
    trs = ["none", "log"] if transform == "auto" else [transform]
    out = []
    for m in models:
        if m == "logistic":
            out += [Cand((p,), "logistic", "none") for p in preds]
        else:
            out += [Cand(preds, m, t) for t in trs]
    return out


# ── Full run ────────────────────────────────────────────────────────────────
def run(aoi, csv_text: str, cfg: Dict[str, Any], mock: bool, prog=None) -> Dict[str, Any]:
    p = prog or (lambda *_a, **_k: None)
    raw = to_insitu_frame(csv_text, cfg["date_col"], cfg.get("time_col"), cfg["value_col"], tz=cfg.get("tz", "Europe/Madrid"),
                          site_col=cfg.get("site_col"), lat_col=cfg.get("lat_col"), lon_col=cfg.get("lon_col"),
                          start_hour=cfg.get("start_hour", 0), end_hour=cfg.get("end_hour", 23))
    p(10, "Leyendo los datos in situ")
    insitu, sites = resolve_sites(raw, cfg.get("pois") or [], cfg.get("point"))
    dates = sorted(insitu["date"].unique())
    p(25, "Extrayendo índices de Sentinel-2")
    rows = mock_satellite_features(insitu) if mock else satellite_features(aoi, sites, dates, cfg["window_days"])
    pairs, stats = build_pairs(insitu, rows, cfg["max_cloud"], cfg["min_water"] / 100, cfg["max_hours"], cfg["window_days"])
    n = len(pairs)
    if n < 10:
        raise ValueError(f"Solo hay {n} pares satélite–in situ válidos (se necesitan ≥ 10). "
                         f"Imágenes: {stats['images']}, válidas (suma por punto): {stats['images_valid']}. "
                         "Prueba a subir la nubosidad máxima, la ventana horaria o a revisar el punto de medida.")

    cands = candidates_auto(cfg.get("kind", "other")) if cfg.get("auto") else \
        candidates_expert(cfg["predictors"], cfg["models"], cfg.get("transform", "auto"))
    if not cands:
        raise ValueError("No hay modelos candidatos")
    n_days = pairs["date"].nunique()
    K = 5 if n_days >= 25 else 4 if n_days >= 16 else 3
    thr = float(cfg["threshold"]) if cfg.get("threshold") else float(np.nanpercentile(pairs["y"], 90))
    blocks = _blocks(pairs, K)
    crit = cfg.get("criterion", "balanced") if cfg.get("criterion") in CRITERIA else "balanced"

    # 1) Honest estimate: nested validation (model selection is repeated inside each training block)
    oof = np.full(n, np.nan); chosen = []
    p(45, f"Validando {len(cands)} modelos candidatos")
    for i_b, b in enumerate(blocks):
        p(45 + 40 * i_b / max(1, len(blocks)), f"Bloque temporal {i_b + 1} de {len(blocks)}")
        tr = np.setdiff1d(np.arange(n), b)
        dtr = pairs.iloc[tr].reset_index(drop=True)
        inner = _select(cands, dtr, max(2, K - 1), criterion=crit, thr=thr)
        best = inner[0][2] if inner and np.isfinite(inner[0][1]) else cands[0]
        chosen.append(best.key)
        oof[b] = Cand(best.preds, best.model, best.transform).fit(dtr).predict(pairs.iloc[b])
    honest = _metrics(pairs["y"], oof, thr)
    ok = np.isfinite(oof)
    y_ok, p_ok = pairs["y"].values[ok], oof[ok]
    # 80 % prediction interval (conformal, with out-of-sample residuals): multiplicative in log scale
    lr = np.log1p(np.clip(y_ok, 0, None)) - np.log1p(np.clip(p_ok, 0, None))
    lo_q, hi_q = (float(np.quantile(lr, 0.10)), float(np.quantile(lr, 0.90))) if len(lr) >= 5 else (None, None)
    coverage = float(np.mean((lr >= lo_q) & (lr <= hi_q))) if lo_q is not None else None
    uncertainty = {"log_lo": lo_q, "log_hi": hi_q, "coverage": coverage,
                   "factor_lo": float(np.exp(lo_q)) if lo_q is not None else None,
                   "factor_hi": float(np.exp(hi_q)) if hi_q is not None else None}
    per_site = []
    if pairs["site"].nunique() > 1:
        for site in sorted(pairs["site"].unique()):
            m = (pairs["site"] == site).values
            per_site.append({"site": site, **_metrics(pairs["y"].values[m], oof[m], thr)})

    # 2) Final model: selection with all the data + full fit
    p(90, "Ajustando el modelo final")
    ranking = _select(cands, pairs, K, criterion=crit, thr=thr)
    final = Cand(ranking[0][2].preds, ranking[0][2].model, ranking[0][2].transform).fit(pairs)
    fit_all = _metrics(pairs["y"], final.predict(pairs), thr)

    top = [{"predictors": list(c.preds), "model": c.model, "transform": c.transform, "label": c.describe(),
            "cv_r2": r2} for _, r2, c in ranking[:8] if np.isfinite(r2)]
    block_of = np.zeros(n, int)
    for bi, blk in enumerate(blocks):
        block_of[blk] = bi
    def _pi(v, q):
        return float(max(0.0, (1 + v) * np.exp(q) - 1)) if (np.isfinite(v) and q is not None) else None
    pred_rows = [{"date": d, "site": st, "y_true": float(y), "y_oof": (float(v) if np.isfinite(v) else None), "block": int(bk),
                  "lo80": _pi(v, lo_q), "hi80": _pi(v, hi_q)}
                 for d, st, y, v, bk in zip(pairs["date"], pairs["site"], pairs["y"], oof, block_of)]
    raster = final.raster()
    rng = {p: [float(np.nanpercentile(pairs[p], 1)), float(np.nanpercentile(pairs[p], 99))] for p in final.preds}
    if raster:
        raster["train_range"] = rng
        raster["interval"] = uncertainty
    return {
        "final": final, "raster": raster, "uncertainty": uncertainty, "train_range": rng,
        "summary": {
            "model": final.model, "model_label": final.describe(), "transform": final.transform,
            "predictors": list(final.preds), "n_pairs": n, "n_days": int(n_days), "k_blocks": K, "threshold": thr,
            "point": cfg.get("point") if len(sites) == 1 else None, "sites": sites,
            "max_hours": cfg["max_hours"], "window_days": cfg["window_days"],
            "selection_stability": float(chosen.count(final.key) / len(chosen)),
            "n_candidates": len(cands), "auto": bool(cfg.get("auto")),
            "criterion": crit, "criterion_label": CRITERIA[crit], "smear": float(getattr(final, "smear", 1.0)),
            "water_mask": "polígono del embalse −20 m, píxeles despejados (sin nubes, sombras, cirros)",
        },
        "honest": honest, "per_site": per_site, "fit_all": fit_all, "ranking": top, "predictions": pred_rows,
        "pairs": pairs.drop(columns=[c for c in CANDIDATE_INDICES if c not in final.preds]).to_dict(orient="records"),
        "stats": {**stats, "n_insitu_rows": int(len(insitu)), "n_insitu_dates": len(dates)},
    }


def clean(obj):
    """Convert NaN/inf/numpy values to valid JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
