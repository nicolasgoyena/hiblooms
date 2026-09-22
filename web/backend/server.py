"""
HIBLOOMS · backend ligero para el frontend React.

Reutiliza hiblooms_core.py (misma lógica que la app Streamlit) y expone
endpoints síncronos sencillos para búsquedas satelitales:

  GET  /api/health
  GET  /api/reservoirs            → GeoJSON de embalses (EPSG:4326)
  GET  /api/indices               → catálogo de índices con rangos de color
  POST /api/search                → fechas Sentinel-2 válidas (nubes/cobertura)
  POST /api/image                 → teselas XYZ del índice + RGB + estadísticos
  POST /api/timeseries            → media del índice en el embalse por fecha

Credenciales GEE (por orden):
  1. GEE_SERVICE_ACCOUNT_JSON  (contenido JSON o ruta a fichero)
  2. credenciales locales de `earthengine authenticate`
Si no hay credenciales o HIBLOOMS_MOCK=1 → modo demo con datos simulados.

Arranque (desde la raíz del repo):
  uvicorn web.backend.server:app --reload --port 8000
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import base64
import io
import tempfile
import uuid
import zipfile
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SHAPEFILE = str(ROOT / "shapefiles" / "embalses_hiblooms.shp")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Catálogo de índices ──────────────────────────────────────────────────────
INDICES = [
    {"id": "PC_Val_cal", "label": "Ficocianina · El Val (experimental)", "unit": "µg/L", "min": 0, "max": 100, "group": "Experimentales", "reservoir": "VAL"},
    {"id": "Chla_Val_cal", "label": "Clorofila-a · El Val", "unit": "µg/L", "min": 0, "max": 60, "group": "Calibrados", "reservoir": "VAL"},
    {"id": "PC_Bellus_cal", "label": "Ficocianina · Bellús (experimental)", "unit": "µg/L", "min": 0, "max": 800, "group": "Experimentales", "reservoir": "BELLUS"},
    {"id": "Chla_Bellus_cal", "label": "Clorofila-a calibrada · Bellús", "unit": "µg/L", "min": 0, "max": 80, "group": "Calibrados", "reservoir": "BELLUS"},
    {"id": "UV_PC_Gral_cal", "label": "Ficocianina general (UV, experimental)", "unit": "µg/L", "min": 0, "max": 100, "group": "Experimentales"},
    {"id": "PCI_B5/B4", "label": "PCI (B5/B4, experimental)", "unit": "", "min": 0.5, "max": 3, "group": "Experimentales"},
    {"id": "NDCI_ind", "label": "NDCI", "unit": "", "min": -0.2, "max": 0.5, "group": "Espectrales"},
    {"id": "MCI", "label": "MCI", "unit": "", "min": -0.05, "max": 0.2, "group": "Espectrales"},
]
INDEX_BY_ID = {i["id"]: i for i in INDICES}
PALETTE = ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"]

# Carpeta de datos de la instancia (shapefiles subidos, calibraciones, cachés).
# Configurable con HIBLOOMS_DATA_DIR para poder desplegar en contenedores o en
# infraestructuras externas (LifeWatch) con volumen propio. Si se pierde, la web
# sigue funcionando: solo desaparecen los datos subidos por los usuarios.
UPLOADS = Path(os.getenv("HIBLOOMS_DATA_DIR") or (Path(__file__).resolve().parent / "_uploads"))
UPLOADS.mkdir(parents=True, exist_ok=True)


# ── Inicialización GEE / modo demo ───────────────────────────────────────────
MOCK = os.getenv("HIBLOOMS_MOCK") == "1"


def _secrets() -> dict:
    """Lee el mismo .streamlit/secrets.toml que usa la app Streamlit (si existe)."""
    try:
        import tomllib  # Python ≥ 3.11
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # pip install tomli (Python ≤ 3.10)
        except ModuleNotFoundError:
            return {}
    for cand in (ROOT / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"):
        if cand.exists():
            try:
                return tomllib.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}

GEE_ERROR: Optional[str] = None
core = None
if not MOCK:
    try:
        import hiblooms_core as core  # noqa: E402

        sa = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        if not sa:  # mismo secrets.toml que usa la app Streamlit
            sec = _secrets()
            if "GEE_SERVICE_ACCOUNT_JSON" in sec:
                sa = json.loads(sec["GEE_SERVICE_ACCOUNT_JSON"], strict=False)
        core.init_ee(sa if sa else None)
    except Exception as e:  # sin credenciales → demo
        GEE_ERROR = str(e)
        MOCK = True

app = FastAPI(title="HIBLOOMS web API", version="0.1.0")
# Respuestas comprimidas: las series y perfiles viajan 5–10 veces más ligeros
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Acceso (login) ───────────────────────────────────────────────────────────
# Usuarios: variable HIBLOOMS_USERS="user1:pass1,user2:pass2" o la sección [auth]
# de secrets.toml (username1/password1, username2/password2… igual que Streamlit).
# Si no hay usuarios configurados, la web queda abierta (sin login).
import hashlib, hmac, secrets as _pysecrets, time  # noqa: E402


def _load_users() -> dict:
    env = os.getenv("HIBLOOMS_USERS")
    if env:
        return dict(x.split(":", 1) for x in env.split(",") if ":" in x)
    a = _secrets().get("auth", {})
    return {str(v): str(a[f"password{k[8:]}"]) for k, v in a.items()
            if k.startswith("username") and f"password{k[8:]}" in a}


USERS = _load_users()
TOKEN_DAYS = 7


def _secret_key() -> bytes:
    if os.getenv("HIBLOOMS_SECRET"):
        return os.environ["HIBLOOMS_SECRET"].encode()
    f = UPLOADS / ".secret"
    f.parent.mkdir(exist_ok=True)
    if not f.exists():
        f.write_text(_pysecrets.token_hex(32))
    return f.read_text().strip().encode()


SECRET = _secret_key()


def _make_token(user: str) -> str:
    body = base64.urlsafe_b64encode(f"{user}|{int(time.time()) + TOKEN_DAYS * 86400}".encode()).decode()
    return body + "." + hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()


def _check_token(tok: str) -> Optional[str]:
    try:
        body, sig = tok.split(".")
        if not hmac.compare_digest(sig, hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()):
            return None
        user, exp = base64.urlsafe_b64decode(body).decode().rsplit("|", 1)
        return user if int(exp) > time.time() and user in USERS else None
    except Exception:
        return None


PUBLIC = {"/api/health", "/api/login"}


@app.middleware("http")
async def auth_mw(request: Request, call_next):
    request.state.user = None
    if USERS and request.url.path.startswith("/api") and request.url.path not in PUBLIC and request.method != "OPTIONS":
        h = request.headers.get("authorization", "")
        tok = h[7:] if h.lower().startswith("bearer ") else request.query_params.get("token", "")
        user = _check_token(tok)
        if not user:
            return JSONResponse({"detail": "Sesión caducada o no iniciada"}, status_code=401)
        request.state.user = user
    return await call_next(request)


class LoginReq(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(req: LoginReq):
    ok = req.username in USERS and hmac.compare_digest(req.password, USERS[req.username])
    if not ok:
        time.sleep(0.5)
        raise HTTPException(401, "Usuario o contraseña incorrectos")
    return {"token": _make_token(req.username), "user": req.username}


@app.get("/api/me")
def me(request: Request):
    return {"user": request.state.user, "auth": bool(USERS)}


# ── Utilidades ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _reservoirs_gdf() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs is None or (gdf.crs.to_epsg() or 0) != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


# ── Embalses subidos por el usuario (shapefile ZIP) ──────────────────────────
# Se guardan como GeoJSON en web/backend/_uploads/ para sobrevivir a reinicios.
# Identificador de embalse propio: "u:<upload_id>:<n>" (n = nº de polígono).
CUSTOM_PREFIX = "u:"


@lru_cache(maxsize=32)
def _upload_gdf(upload_id: str) -> gpd.GeoDataFrame:
    f = UPLOADS / f"{upload_id}.geojson"
    if not re.fullmatch(r"[a-f0-9]{12}", upload_id) or not f.exists():
        raise HTTPException(404, "Shapefile subido no encontrado (vuelve a subirlo)")
    return gpd.read_file(f)


def _custom_geom(res_id: str):
    try:
        _, upload_id, n = res_id.split(":")
        return _upload_gdf(upload_id).geometry.iloc[int(n)]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(404, f"Embalse propio desconocido: {res_id}")


@lru_cache(maxsize=64)
def _aoi(name: str):
    import ee
    if name.startswith(CUSTOM_PREFIX):
        return ee.Geometry(_custom_geom(name).__geo_interface__, geodesic=False)
    gdf = core.load_reservoir_shapefile(name, SHAPEFILE)
    return core.gdf_to_ee_geometry(gdf)


def _check(name: str, index: Optional[str] = None):
    if name.startswith(CUSTOM_PREFIX):
        _custom_geom(name)
    elif name not in set(_reservoirs_gdf()["NOMBRE"]):
        raise HTTPException(404, f"Embalse desconocido: {name}")
    if index is not None and index not in INDEX_BY_ID and index not in CALS:
        raise HTTPException(400, f"Índice desconocido: {index}")


def _mock_rng(*parts) -> random.Random:
    return random.Random("|".join(map(str, parts)))


# ── Imagen procesada (cacheada) ─────────────────────────────────────────────
ALL_INDEX_IDS = [i["id"] for i in INDICES]
POIS_CSV = ROOT / "data" / "puntos_interes.csv"


S2 = "COPERNICUS/S2_SR_HARMONIZED"


def _image_stats(img, aoi):
    """Nubosidad y cobertura de UNA imagen sobre el embalse, calculadas en el servidor de GEE
    (misma fórmula que hiblooms_core.calculate_cloud_percentage / calculate_coverage_percentage)."""
    import ee
    scl = img.select("SCL")
    cloud_scl = scl.eq(7).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    valid = scl.mask().And(scl.eq(4).Or(scl.eq(5)).Not())
    fr = ee.Image.cat(
        cloud_scl.updateMask(valid).rename("c_scl"),
        img.select("MSK_CLDPRB").gte(10).updateMask(valid).rename("c_prob"),
    ).reduceRegion(ee.Reducer.mean(), aoi, 20, maxPixels=1e13)
    cnt = ee.Image.cat(
        ee.Image(1).rename("tot"),
        ee.Image(1).updateMask(img.select("B4").mask()).rename("val"),
    ).reduceRegion(ee.Reducer.count(), aoi, 20, maxPixels=1e13)
    cs, cp = fr.get("c_scl"), fr.get("c_prob")
    isnull = lambda x: ee.Algorithms.IsEqual(x, None)
    cloud = ee.Algorithms.If(
        isnull(cs),
        ee.Algorithms.If(isnull(cp), None, ee.Number(cp).multiply(100)),
        ee.Algorithms.If(isnull(cp), ee.Number(cs).multiply(100),
                         ee.Number(cs).multiply(0.95).add(ee.Number(cp).multiply(0.05)).multiply(100)),
    )
    tot = ee.Number(cnt.get("tot"))
    cov = ee.Algorithms.If(tot.gt(0), ee.Number(cnt.get("val")).divide(tot).multiply(100), 0)
    return cloud, cov


# (reservoir, max_cloud, min_cov) -> {día: {"id", "time", "cloud", "cov"}}  mejor imagen de cada día
_BEST: dict = {}
# rangos ya consultados por (reservoir, max_cloud, min_cov): lista de (start, end)
_SCANNED: dict = {}


def _scan(reservoir: str, start: str, end: str, max_cloud: int, min_cov: float) -> dict:
    """UNA sola llamada a GEE: estadísticas de todas las imágenes del rango → mejor imagen por día."""
    import ee
    key = (reservoir, max_cloud, float(min_cov))
    best = _BEST.setdefault(key, {})
    for a, b in _SCANNED.get(key, []):
        if a <= start and end <= b:
            return best
    aoi = _aoi(reservoir)
    col = ee.ImageCollection(S2).filterBounds(aoi).filterDate(start, end)

    def feat(img):
        cloud, cov = _image_stats(img, aoi)
        return ee.Feature(None, {"id": img.get("system:index"), "t": img.get("system:time_start"),
                                 "cloud": cloud, "cov": cov})

    rows = col.map(feat).getInfo()["features"]
    for r in rows:
        p = r["properties"]
        if p.get("cloud") is None or p.get("t") is None:
            continue
        cloud, cov = float(p["cloud"]), float(p.get("cov") or 0)
        if not ((max_cloud == 100 or cloud <= max_cloud) and cov >= min_cov):
            continue
        day = datetime.utcfromtimestamp(p["t"] / 1000).strftime("%Y-%m-%d")
        if day not in best or cloud < best[day]["cloud"]:
            best[day] = {"id": p["id"], "time": p["t"], "cloud": cloud, "cov": cov}
    _SCANNED.setdefault(key, []).append((start, end))
    return best


def _best_for(reservoir: str, day: str, max_cloud: int, min_cov: float = 50.0) -> dict:
    best = _BEST.get((reservoir, max_cloud, float(min_cov)), {})
    if day not in best:
        d = date.fromisoformat(day)
        best = _scan(reservoir, day, (d + timedelta(days=1)).isoformat(), max_cloud, min_cov)
    if day not in best:
        raise HTTPException(404, f"No hay imagen válida el {day} con esos umbrales")
    return best[day]


def _indices_image(image_id: str, aoi):
    """Imagen escalada (RGB) + imagen de índices a partir del id de la imagen.
    Nota: _build_indices_image ya divide entre 10000, así que se le pasa la imagen SIN escalar.
    (process_sentinel2 le pasa la ya escalada → reflectancias /10000 dos veces; afecta a MCI y
    PC_Bellus_cal y a los modelos calibrados con diferencias de bandas.)"""
    import ee
    img = ee.Image(f"{S2}/{image_id}")
    req = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A"]
    clipped = img.clip(aoi)
    scaled = clipped.addBands(clipped.select(req).divide(10000), overwrite=True)
    ind = core._build_indices_image(clipped, aoi, ALL_INDEX_IDS)
    return scaled, ind



# ── Máscara de lámina de agua (evita falsos positivos de orilla) ─────────────
# El polígono del embalse es el vaso lleno: con el embalse bajo, buena parte queda
# en seco (suelo, limo, vegetación de ribera) y esos píxeles dan índices altísimos
# que se confunden con una floración. Se resuelve en dos pasos:
#   1. agua detectada en la propia imagen: MNDWI > 0 (SWIR, robusto en agua turbia)
#      o NDWI > 0 (mantiene las natas de bloom, que suben el NIR y bajan el NDWI)
#   2. erosión de 20 m (1 píxel) para tirar los píxeles mezclados del borde
SHORE_ERODE_M = 20


def _water_mask(ind, erode_m: float = SHORE_ERODE_M):
    import ee
    b3, b8a = ind.select("B3"), ind.select("B8A")           # ya escaladas (/10000)
    b11 = ind.select("B11").divide(10000)
    ndwi = b3.subtract(b8a).divide(b3.add(b8a))
    mndwi = b3.subtract(b11).divide(b3.add(b11))
    w = mndwi.gt(0).Or(ndwi.gt(0))
    if erode_m:
        w = w.focal_min(radius=erode_m, units="meters")
    return w.rename("WATER")


def _apply_water(ind, on: bool):
    """Deja solo la lámina de agua (sin orillas) en todas las bandas de índices."""
    if not on:
        return ind
    try:
        return ind.updateMask(_water_mask(ind))
    except Exception:
        return ind


# ── Calibraciones del usuario (índices extra) ────────────────────────────────
CALS: dict = {}  # id -> {"meta": {...}, "raster": {...}, "summary": {...}}
CAL_PREFIX = "cal:"


def _load_cals():
    if UPLOADS.exists():
        for f in UPLOADS.glob("cal_*.json"):
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
                CALS[c["meta"]["id"]] = c
            except Exception:
                pass


def _meta(index: str) -> dict:
    if index in INDEX_BY_ID:
        return INDEX_BY_ID[index]
    if index in CALS:
        return CALS[index]["meta"]
    raise HTTPException(400, f"Índice desconocido: {index}")


def _band(index: str) -> str:
    return index if not index.startswith(CAL_PREFIX) else "CAL_" + index[len(CAL_PREFIX):]


def _raster_expr(ind, rc: dict):
    """Aplica una calibración (lineal, lineal en log o logística) píxel a píxel."""
    import ee
    if rc.get("type") == "logistic":
        x = ind.select(rc["predictor"]).unmask(float(rc.get("fill", 0)))
        return ee.Image(float(rc["L"])).divide(ee.Image(1).add(x.subtract(float(rc["x0"])).multiply(-float(rc["k"])).exp()))
    preds = rc.get("predictors") or rc.get("predictor_set") or []
    fills = rc.get("fill_values") or [0.0] * len(preds)
    expr = ee.Image.constant(float(rc.get("intercept", 0.0)))
    for p, c, f in zip(preds, rc["coefficients"], fills):
        expr = expr.add(ind.select(p).unmask(float(f)).multiply(float(c)))
    if rc.get("transform") == "log":
        expr = expr.min(12).exp().multiply(float(rc.get("smear", 1.0))).subtract(1)
    return expr.max(0)


def _with_index(ind, index: str, aoi=None):
    """Añade la banda del modelo calibrado si el índice es una calibración del usuario, con la
    misma máscara que en el entrenamiento (embalse −20 m, píxeles despejados). Añade también
    la banda EXTRAP_* = 1 donde algún índice está fuera del rango con el que se calibró."""
    if not index.startswith(CAL_PREFIX):
        return ind
    import ee
    rc = CALS[index]["raster"]
    scl = ind.select("SCL")
    mask = cal.clear_water_mask(scl, aoi.buffer(-cal.SHORE_BUFFER_M)) if aoi is not None else scl.eq(6)
    band = _raster_expr(ind, rc).updateMask(mask).rename(_band(index))
    out = ind.addBands(band, overwrite=True)
    rng = rc.get("train_range") or {}
    if rng:
        ext = ee.Image(0)
        for p, (lo, hi) in rng.items():
            x = ind.select(p)
            ext = ext.Or(x.lt(lo)).Or(x.gt(hi))
        out = out.addBands(ext.updateMask(mask).rename("EXTRAP_" + _band(index)), overwrite=True)
    return out


@lru_cache(maxsize=64)
def _processed(reservoir: str, day: str, max_cloud: int):
    b = _best_for(reservoir, day, max_cloud)
    scaled, ind = _indices_image(b["id"], _aoi(reservoir))
    iso = datetime.utcfromtimestamp(b["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
    return scaled, ind, iso, b["cloud"], b["cov"]


def _values_dict(ind, index: str, aoi, points: list, time_ms: int, scale: int = 20, tile_scale: int = 1):
    """ee.Dictionary con la media del embalse (agua SCL=6; en 2018 también SCL=2) y el valor en cada punto."""
    import ee
    scl = ind.select("SCL")
    water = scl.eq(6).Or(scl.eq(2)) if datetime.utcfromtimestamp(time_ms / 1000).year == 2018 else scl.eq(6)
    if index.startswith("CAL_"):  # las calibraciones ya llevan su propia máscara de agua despejada
        water = ee.Image(1)
    band = ind.select(index)
    d = {"mean": band.updateMask(water).reduceRegion(ee.Reducer.mean(), aoi, scale, maxPixels=1e13, tileScale=tile_scale).get(index)}
    for i, p in enumerate(points):
        d[f"p{i}"] = band.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([p.lon, p.lat]).buffer(30), 20).get(index)
    return d


def _unpack(vals: dict, points: list):
    num = lambda v: float(v) if v is not None else None
    return num(vals.get("mean")), {p.name: num(vals.get(f"p{i}")) for i, p in enumerate(points)}




def _mock_value(meta, *seed):
    rng = _mock_rng(*seed)
    return round(meta["min"] + (meta["max"] - meta["min"]) * rng.betavariate(1.5, 5), 3)


# ── Modelos ──────────────────────────────────────────────────────────────────
class Point(BaseModel):
    name: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class SearchReq(BaseModel):
    reservoir: str
    start: date
    end: date
    max_cloud: int = Field(30, ge=0, le=100)
    min_coverage: float = Field(50, ge=0, le=100)


class ImageReq(BaseModel):
    reservoir: str
    date: date
    index: str
    max_cloud: int = Field(30, ge=0, le=100)
    points: List[Point] = Field(default_factory=list, max_length=50)
    water_only: bool = True


class SeriesReq(BaseModel):
    reservoir: str
    index: str
    dates: List[date] = Field(..., max_length=120)
    max_cloud: int = Field(30, ge=0, le=100)
    points: List[Point] = Field(default_factory=list, max_length=50)
    water_only: bool = True


class ClassesReq(BaseModel):
    reservoir: str
    date: date
    index: str
    max_cloud: int = Field(30, ge=0, le=100)
    n_classes: int = Field(5, ge=2, le=10)
    water_only: bool = True


class DownloadReq(BaseModel):
    reservoir: str
    date: date
    indices: List[str] = Field(..., min_length=1)
    max_cloud: int = Field(30, ge=0, le=100)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True, "mode": "demo" if MOCK else "gee", "gee_error": GEE_ERROR, "auth": bool(USERS)}


MODELOS_DIR = ROOT / "data" / "modelos"


@app.get("/api/models")
def models():
    """Fichas de los modelos de la plataforma (data/modelos/*.json)."""
    out = []
    for f in sorted(MODELOS_DIR.glob("*.json")) if MODELOS_DIR.exists() else []:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return {"models": out}


@app.get("/api/indices")
def indices():
    return {"indices": INDICES + [c["meta"] for c in CALS.values()], "palette": PALETTE}


@app.get("/api/reservoirs")
def reservoirs(request: Request):
    gdf = _reservoirs_gdf()[["NOMBRE", "DEMARC", "PROVINCIA", "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.simplify(0.0002, preserve_topology=True)
    fc = json.loads(gdf.to_json())
    for u in _list_uploads(request.state.user):  # shapefiles subidos: persisten entre recargas
        try:
            fc["features"] += _upload_features(u["upload_id"])["features"]
        except Exception:
            pass
    return fc


def _meta_file(upload_id: str) -> Path:
    return UPLOADS / f"{upload_id}.meta.json"


def _list_uploads(user: Optional[str]) -> list:
    out = []
    if UPLOADS.exists():
        for f in sorted(UPLOADS.glob("*.geojson"), key=lambda p: p.stat().st_mtime):
            uid = f.stem
            if not re.fullmatch(r"[a-f0-9]{12}", uid):
                continue
            m = json.loads(_meta_file(uid).read_text(encoding="utf-8")) if _meta_file(uid).exists() else {}
            if USERS and m.get("owner") not in (None, user):
                continue
            out.append({"upload_id": uid, "filename": m.get("filename", uid), "owner": m.get("owner"),
                        "created": m.get("created"), "count": m.get("count")})
    return out


def _upload_features(upload_id: str) -> dict:
    simple = _upload_gdf(upload_id).copy()
    simple["geometry"] = simple.geometry.simplify(0.0002, preserve_topology=True)
    simple["NOMBRE"] = [f"{CUSTOM_PREFIX}{upload_id}:{i}" for i in range(len(simple))]
    simple["CUSTOM"] = True
    return json.loads(simple[["NOMBRE", "LABEL", "CUSTOM", "geometry"]].to_json())


@app.get("/api/uploads")
def uploads(request: Request):
    return {"uploads": _list_uploads(request.state.user)}


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str, request: Request):
    if not any(u["upload_id"] == upload_id for u in _list_uploads(request.state.user)):
        raise HTTPException(404, "Shapefile no encontrado")
    for f in (UPLOADS / f"{upload_id}.geojson", _meta_file(upload_id)):
        f.unlink(missing_ok=True)
    _upload_gdf.cache_clear()
    return {"ok": True}


@app.get("/api/pois")
def pois(reservoir: str):
    """Puntos de interés predefinidos (data/puntos_interes.csv) del embalse."""
    import pandas as pd
    if not POIS_CSV.exists() or reservoir.startswith(CUSTOM_PREFIX):
        return {"points": []}
    df = pd.read_csv(POIS_CSV)
    df = df[df["embalse"].str.upper() == reservoir.upper()]
    return {"points": [{"name": r.nombre, "lat": float(r.latitud), "lon": float(r.longitud)} for r in df.itertuples()]}


@app.post("/api/search")
def search(req: SearchReq):
    _check(req.reservoir)
    if req.end <= req.start:
        raise HTTPException(400, "La fecha final debe ser posterior a la inicial")
    if (req.end - req.start).days > 366 * 2:
        raise HTTPException(400, "Rango máximo: 2 años por búsqueda")

    if MOCK:
        rng = _mock_rng(req.reservoir, req.start, req.end, req.max_cloud)
        d, out = req.start, []
        while d <= req.end:
            if rng.random() < 0.35 + req.max_cloud / 200:
                out.append({"date": d.isoformat(), "cloud": round(rng.uniform(0, req.max_cloud), 1)})
            d += timedelta(days=5)
        return {"reservoir": req.reservoir, "dates": out, "mode": "demo"}

    try:
        s0, s1 = req.start.isoformat(), (req.end + timedelta(days=1)).isoformat()
        best = _scan(req.reservoir, s0, s1, req.max_cloud, req.min_coverage)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    days = sorted(d for d in best if s0 <= d < s1)
    return {"reservoir": req.reservoir, "dates": [{"date": d, "cloud": round(best[d]["cloud"], 1)} for d in days], "mode": "gee"}


@app.post("/api/image")
def image(req: ImageReq):
    _check(req.reservoir, req.index)
    meta = _meta(req.index)

    if MOCK:
        rng = _mock_rng(req.reservoir, req.date, req.index)
        return {
            "mode": "demo", "tile_url": None, "rgb_tile_url": None,
            "datetime": f"{req.date.isoformat()} 10:5{rng.randint(0, 9)}:00",
            "cloud": round(rng.uniform(0, req.max_cloud), 1), "coverage": round(rng.uniform(80, 100), 1),
            "mean": _mock_value(meta, req.reservoir, req.date, req.index),
            "points": {p.name: _mock_value(meta, req.reservoir, req.date, req.index, p.name) for p in req.points},
            **({"interval80": [0, 0], "extrapolation_pct": round(rng.uniform(0, 15), 1)} if req.index in CALS else {}),
        }

    try:
        scaled, ind, iso, cloud, cov = _processed(req.reservoir, req.date.isoformat(), req.max_cloud)
        ind = _apply_water(_with_index(ind, req.index, _aoi(req.reservoir)), req.water_only)
        vis = {"min": meta["min"], "max": meta["max"], "palette": [p.lstrip("#") for p in PALETTE]}
        tile = ind.select(_band(req.index)).getMapId(vis)["tile_fetcher"].url_format
        rgb = scaled.select(["B4", "B3", "B2"]).getMapId({"min": 0, "max": 0.25, "gamma": 1.2})["tile_fetcher"].url_format
        t = _best_for(req.reservoir, req.date.isoformat(), req.max_cloud)["time"]
        import ee
        dvals = _values_dict(ind, _band(req.index), _aoi(req.reservoir), req.points, t)
        eb = "EXTRAP_" + _band(req.index)
        if req.index in CALS and CALS[req.index]["raster"].get("train_range"):
            dvals["extrap"] = ind.select(eb).reduceRegion(ee.Reducer.mean(), _aoi(req.reservoir), 20, maxPixels=1e13).get(eb)
        vals = ee.Dictionary(dvals).getInfo()
        mean, pts = _unpack(vals, req.points)
        extra = {}
        if req.index in CALS:
            iv = CALS[req.index]["raster"].get("interval") or {}
            if mean is not None and iv.get("factor_lo") is not None:
                extra["interval80"] = [max(0.0, (1 + mean) * iv["factor_lo"] - 1), (1 + mean) * iv["factor_hi"] - 1]
            if vals.get("extrap") is not None:
                extra["extrapolation_pct"] = round(100 * float(vals["extrap"]), 1)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    return {"mode": "gee", "tile_url": tile, "rgb_tile_url": rgb, "datetime": iso,
            "cloud": cloud, "coverage": cov, "mean": mean, "points": pts, **extra}


@app.post("/api/timeseries")
def timeseries(req: SeriesReq):
    _check(req.reservoir, req.index)
    meta = _meta(req.index)
    out = []
    if MOCK:
        for d in sorted(req.dates):
            rng = _mock_rng(req.reservoir, d, req.index)
            season = 0.5 + 0.5 * math.sin((d.timetuple().tm_yday - 150) / 365 * 2 * math.pi)
            v = meta["min"] + (meta["max"] - meta["min"]) * min(1, rng.betavariate(1.5, 6) + 0.35 * season * rng.random())
            row = {"date": d.isoformat(), "mean": round(v, 3)}
            for p in req.points:
                row[p.name] = round(v * _mock_rng(d, p.name).uniform(0.6, 1.5), 3)
            out.append(row)
        return {"index": req.index, "series": out, "mode": "demo"}

    import ee
    try:
        aoi = _aoi(req.reservoir)
        days = sorted(d.isoformat() for d in req.dates)
        _scan(req.reservoir, days[0], (date.fromisoformat(days[-1]) + timedelta(days=1)).isoformat(), req.max_cloud, 50.0)
        best = _BEST.get((req.reservoir, req.max_cloud, 50.0), {})
        feats, order = [], []
        for d in days:
            if d not in best:
                continue
            _, ind = _indices_image(best[d]["id"], aoi)
            ind = _apply_water(_with_index(ind, req.index, aoi), req.water_only)
            feats.append(ee.Feature(None, _values_dict(ind, _band(req.index), aoi, req.points, best[d]["time"])).set("day", d))
            order.append(d)
        got = {f["properties"]["day"]: f["properties"] for f in ee.FeatureCollection(feats).getInfo()["features"]} if feats else {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    for d in days:
        mean, pts = _unpack(got.get(d, {}), req.points)
        out.append({"date": d, "mean": mean, **pts})
    return {"index": req.index, "series": out, "mode": "gee"}


@app.post("/api/classes")
def classes(req: ClassesReq):
    """Superficie del embalse por clases del índice (rangos iguales entre min y max)."""
    _check(req.reservoir, req.index)
    meta = _meta(req.index)
    step = (meta["max"] - meta["min"]) / req.n_classes
    bins = [round(meta["min"] + i * step, 4) for i in range(req.n_classes)] + [1e9]

    if MOCK:
        rng = _mock_rng(req.reservoir, req.date, req.index, "cls")
        w = sorted([rng.random() ** 2 for _ in range(req.n_classes)], reverse=True)
        tot, area = sum(w), rng.uniform(100, 600)
        return {"classes": [
            {"low": bins[i], "high": bins[i + 1] if i < req.n_classes - 1 else None,
             "area_ha": round(area * w[i] / tot, 2), "pct": round(100 * w[i] / tot, 1)}
            for i in range(req.n_classes)], "mode": "demo"}

    try:
        _, ind, _, _, _ = _processed(req.reservoir, req.date.isoformat(), req.max_cloud)
        res = core.calcular_distribucion_area_por_clases(_apply_water(_with_index(ind, req.index, _aoi(req.reservoir)), req.water_only), _band(req.index), _aoi(req.reservoir), bins)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    return {"classes": [
        {"low": bins[i], "high": bins[i + 1] if i < req.n_classes - 1 else None,
         "area_ha": round(r["area_ha"], 2), "pct": round(r["porcentaje"], 1)}
        for i, r in enumerate(res)], "mode": "gee"}


@app.post("/api/download")
def download(req: DownloadReq):
    """URL de descarga GeoTIFF multibanda (índices elegidos) para esa fecha."""
    _check(req.reservoir)
    bad = [i for i in req.indices if i not in INDEX_BY_ID and i not in CALS]
    if bad:
        raise HTTPException(400, f"Índices desconocidos: {bad}")
    if MOCK:
        raise HTTPException(400, "La descarga GeoTIFF necesita conexión a Earth Engine (modo demo)")
    try:
        _, ind, _, _, _ = _processed(req.reservoir, req.date.isoformat(), req.max_cloud)
        for i in req.indices:
            ind = _with_index(ind, i, _aoi(req.reservoir))
        url = core.generar_url_geotiff_multibanda(ind, [_band(i) for i in req.indices], _aoi(req.reservoir), scale=20)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    if not url:
        raise HTTPException(502, "Earth Engine no pudo generar la descarga (¿embalse demasiado grande?)")
    return {"url": url, "filename": f"{req.reservoir}_{req.date.isoformat()}.tif"}


class UploadReq(BaseModel):
    filename: str
    content_b64: str = Field(..., max_length=40_000_000)  # ~30 MB de ZIP


@app.post("/api/upload-shapefile")
def upload_shapefile(req: UploadReq, request: Request):
    """Recibe un ZIP con un shapefile (.shp/.shx/.dbf/.prj) y devuelve sus polígonos como embalses propios."""
    try:
        raw = base64.b64decode(req.content_b64)
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "El archivo no es un ZIP válido")
    shp = [n for n in zf.namelist() if n.lower().endswith(".shp") and not n.startswith("__MACOSX")]
    if not shp:
        raise HTTPException(400, "El ZIP no contiene ningún .shp")
    with tempfile.TemporaryDirectory() as tmp:
        for n in zf.namelist():
            if ".." in n or n.startswith(("/", "\\")):
                continue
            zf.extract(n, tmp)
        try:
            gdf = gpd.read_file(Path(tmp) / shp[0])
        except Exception as e:
            raise HTTPException(400, f"No se pudo leer el shapefile: {e}")
    if gdf.empty:
        raise HTTPException(400, "El shapefile está vacío")
    if gdf.crs is None:
        raise HTTPException(400, "El shapefile no tiene sistema de referencia (.prj). Inclúyelo en el ZIP.")
    gdf = gdf.to_crs(epsg=4326)
    import shapely
    gdf["geometry"] = shapely.force_2d(gdf.geometry.values)
    gdf = gdf[gdf.geometry.notna() & gdf.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    if gdf.empty:
        raise HTTPException(400, "El shapefile no contiene polígonos")
    if len(gdf) > 200:
        raise HTTPException(400, "Máximo 200 polígonos por shapefile")

    # Columna de nombre: NOMBRE, o la primera de texto razonable, o numeración
    stem = Path(req.filename).stem
    cand = [c for c in gdf.columns if c.upper() in ("NOMBRE", "NAME", "NOM", "EMBALSE")]
    cand += [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == object and c not in cand]
    col = cand[0] if cand else None
    labels = [str(v) if col and v is not None and str(v).strip() else f"{stem} {i + 1}" for i, v in enumerate(gdf[col] if col else [None] * len(gdf))]

    upload_id = uuid.uuid4().hex[:12]
    out = gpd.GeoDataFrame({"LABEL": labels}, geometry=gdf.geometry, crs=4326)
    out.to_file(UPLOADS / f"{upload_id}.geojson", driver="GeoJSON")
    _meta_file(upload_id).write_text(json.dumps({"filename": req.filename, "owner": request.state.user,
                                                 "created": datetime.now().isoformat(timespec="seconds"),
                                                 "count": len(out)}), encoding="utf-8")

    simple = out.copy()
    simple["geometry"] = simple.geometry.simplify(0.0002, preserve_topology=True)
    simple["NOMBRE"] = [f"{CUSTOM_PREFIX}{upload_id}:{i}" for i in range(len(simple))]
    simple["CUSTOM"] = True
    return {"upload_id": upload_id, "name_column": col, "count": len(simple),
            "geojson": json.loads(simple[["NOMBRE", "LABEL", "CUSTOM", "geometry"]].to_json())}


# ── Calibración ──────────────────────────────────────────────────────────────
import calibration as cal  # noqa: E402  (web/backend/calibration.py)

_load_cals()


class CsvReq(BaseModel):
    csv_text: str = Field(..., max_length=30_000_000)


class CalPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: Optional[str] = None


class CalibrateReq(BaseModel):
    reservoir: str
    csv_text: str = Field(..., max_length=30_000_000)
    date_col: str
    time_col: Optional[str] = None
    value_col: str
    unit: str = ""
    tz: str = "Europe/Madrid"
    point: Optional[CalPoint] = None          # None = media de todo el embalse (si el CSV no trae ubicación)
    site_col: Optional[str] = None            # columna con el nombre del punto de cada fila
    lat_col: Optional[str] = None             # columnas de coordenadas de cada fila
    lon_col: Optional[str] = None
    pois: List[Point] = Field(default_factory=list, max_length=200)  # puntos del embalse para resolver nombres
    start_hour: int = Field(0, ge=0, le=23)
    end_hour: int = Field(23, ge=0, le=23)
    max_cloud: int = Field(30, ge=0, le=100)  # nubosidad sobre el embalse (SCL)
    min_water: int = Field(50, ge=0, le=100)  # % de píxeles de agua despejada en la zona de extracción
    max_hours: float = Field(3, gt=0, le=24)  # ± horas entre medida in situ y paso del satélite
    window_days: int = Field(0, ge=0, le=5)   # tolerancia en días si no hay medida en ±max_hours
    auto: bool = True
    kind: str = Field("other", pattern="^(phycocyanin|chlorophyll|other)$")
    predictors: List[str] = Field(default_factory=lambda: ["R705_R665"])
    models: List[str] = Field(default_factory=lambda: ["linear", "ridge"])
    transform: str = Field("auto", pattern="^(auto|none|log)$")
    criterion: str = Field("balanced", pattern="^(balanced|peaks|general|alerts)$")
    threshold: Optional[float] = None          # umbral de alerta (por defecto P90 de los datos)


@app.get("/api/calibration/options")
def calibration_options():
    return {"predictors": cal.CANDIDATE_INDICES, "models": cal.MODEL_NAMES, "rasterizable": cal.RASTERIZABLE}


@app.post("/api/calibration/preview")
def calibration_preview(req: CsvReq):
    try:
        return cal.csv_preview(req.csv_text)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/calibrate")
def calibrate(req: CalibrateReq, wait: bool = False):
    """Por defecto devuelve {job_id}; con ?wait=1 calcula y devuelve el resultado."""
    if wait:
        return _do_calibrate(req, lambda *_a, **_k: None)
    return _start_job("calibrate", lambda prog: _do_calibrate(req, prog))


def _do_calibrate(req: CalibrateReq, prog):
    _check(req.reservoir)
    bad = [p for p in req.predictors if p not in cal.CANDIDATE_INDICES] + [m for m in req.models if m not in cal.MODEL_NAMES]
    if bad:
        raise HTTPException(400, f"Opciones desconocidas: {bad}")
    if not req.auto and (not req.predictors or not req.models):
        raise HTTPException(400, "Elige al menos un índice y un modelo")
    cfg = req.model_dump()
    cfg["pois"] = [p.model_dump() for p in req.pois]
    if not cfg["pois"] and req.site_col and not (req.lat_col and req.lon_col):
        cfg["pois"] = pois(req.reservoir)["points"]
    try:
        out = cal.run(None if MOCK else _aoi(req.reservoir), req.csv_text, cfg, MOCK, prog)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Error en la calibración: {e}")

    sm, raster = out["summary"], out["raster"]
    cal_id = None
    if raster:
        cid = uuid.uuid4().hex[:8]
        cal_id = CAL_PREFIX + cid
        y = np.array([p["y_true"] for p in out["predictions"]])
        ns = len(sm.get("sites") or [])
        where = f"{ns} puntos" if ns > 1 else ((req.point.name or "punto") if req.point else "embalse")
        label = f"{req.value_col} · {sm['model_label']} ({_label_of(req.reservoir)}, {where})"
        meta = {"id": cal_id, "label": label, "unit": req.unit, "group": "Tus calibraciones", "reservoir": req.reservoir,
                "min": round(float(max(0.0, np.nanpercentile(y, 2))), 2), "max": round(float(np.nanpercentile(y, 98)), 2) or 1.0}
        entry = {"meta": meta, "raster": cal.clean(raster), "summary": cal.clean({**sm, "honest": out["honest"]})}
        CALS[cal_id] = entry
        (UPLOADS / f"cal_{cid}.json").write_text(json.dumps(entry), encoding="utf-8")
        import joblib
        joblib.dump({"candidate": out["final"].key, "estimator": out["final"].est, "params": out["final"].params,
                     "transform": out["final"].transform, "predictors": list(out["final"].preds), "raster": raster},
                    UPLOADS / f"cal_{cid}.joblib")

    return cal.clean({
        "calibration_id": cal_id, "rasterizable": raster is not None, "raster": raster,
        "summary": sm, "honest": out["honest"], "uncertainty": out["uncertainty"], "per_site": out["per_site"], "fit_all": out["fit_all"], "ranking": out["ranking"],
        "predictions": out["predictions"], "pairs": out["pairs"], "stats": out["stats"],
        "mode": "demo" if MOCK else "gee",
    })


def _label_of(reservoir: str) -> str:
    if reservoir.startswith(CUSTOM_PREFIX):
        _, uid, n = reservoir.split(":")
        return str(_upload_gdf(uid)["LABEL"].iloc[int(n)])
    return reservoir.title()


@app.get("/api/calibration/{cid}/model")
def calibration_model(cid: str):
    from fastapi.responses import FileResponse
    if not re.fullmatch(r"[a-f0-9]{8}", cid):
        raise HTTPException(404, "Calibración no encontrada")
    f = UPLOADS / f"cal_{cid}.joblib"
    if not f.exists():
        raise HTTPException(404, "Solo se guarda el modelo de las calibraciones exportables al mapa")
    return FileResponse(f, filename=f"hiblooms_modelo_{cid}.joblib", media_type="application/octet-stream")


@app.delete("/api/calibration/{cid}")
def calibration_delete(cid: str):
    key = CAL_PREFIX + cid
    CALS.pop(key, None)
    for ext in ("json", "joblib"):
        f = UPLOADS / f"cal_{cid}.{ext}"
        if re.fullmatch(r"[a-f0-9]{8}", cid) and f.exists():
            f.unlink()
    return {"ok": True}



# ── Tareas en segundo plano ─────────────────────────────────────────────────
# Las operaciones largas (calibración, monitor, climatología) pueden tardar
# minutos y los servidores donde se despliega esto cortan las peticiones al
# minuto o dos. Se lanzan en un hilo y el cliente pregunta por el resultado.
# Sin base de datos: viven en memoria; si el servidor se reinicia, se repiten.
import threading  # noqa: E402

JOBS: dict = {}
JOB_TTL = 3600          # segundos que se guarda un resultado ya terminado
JOB_MAX = 40


class Progress:
    """Se pasa a la función de cálculo para que informe de por dónde va."""

    def __init__(self, job: dict):
        self.job = job

    def __call__(self, pct: float, step: str = ""):
        self.job["progress"] = max(0, min(100, round(pct)))
        if step:
            self.job["step"] = step


def _gc_jobs():
    ahora = time.time()
    viejos = [k for k, j in JOBS.items() if j["status"] != "running" and ahora - j["ended"] > JOB_TTL]
    for k in viejos:
        JOBS.pop(k, None)
    while len(JOBS) > JOB_MAX:
        k = min(JOBS, key=lambda k: JOBS[k]["started"])
        JOBS.pop(k, None)


def _start_job(kind: str, fn) -> dict:
    """fn(progress) -> resultado. Devuelve {job_id} al instante."""
    _gc_jobs()
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "kind": kind, "status": "running", "progress": 0, "step": "",
           "started": time.time(), "ended": 0.0, "result": None, "detail": None}
    JOBS[jid] = job

    def run():
        try:
            job["result"] = fn(Progress(job))
            job["status"] = "done"
            job["progress"] = 100
        except HTTPException as e:
            job["status"] = "error"; job["detail"] = str(e.detail)
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"; job["detail"] = str(e)
        finally:
            job["ended"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid, "status": "running"}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Tarea desconocida o caducada: vuelve a lanzarla")
    out = {k: j[k] for k in ("id", "kind", "status", "progress", "step")}
    out["seconds"] = round((j["ended"] or time.time()) - j["started"], 1)
    if j["status"] == "done":
        out["result"] = j["result"]
    if j["status"] == "error":
        out["detail"] = j["detail"]
    return out


# ── Monitorización general (todos los embalses de un vistazo) ────────────────
# Estado orientativo: no usa calibraciones locales (solo existen en algunos embalses),
# sino dos índices espectrales generalistas sobre el píxel despejado más reciente
# de cada embalse en los últimos N días:
#   NDCI  (B5−B4)/(B5+B4) → proxy de clorofila-a: biomasa algal en general
#   PCI   B5/B4           → proxy de ficocianina: sospecha de cianobacterias
# Umbrales orientativos, pensados para comparar embalses y priorizar, no para dar cifras.
NDCI_LEVELS = [(0.00, "bajo"), (0.10, "moderado"), (0.20, "alto"), (0.35, "muy alto")]
PCI_SUSPECT = 1.35


def _level(ndci_p90: Optional[float]) -> str:
    if ndci_p90 is None:
        return "sin datos"
    lvl = "muy bajo"
    for thr, name in NDCI_LEVELS:
        if ndci_p90 >= thr:
            lvl = name
    return lvl


# ── Volumen embalsado oficial (Boletín Hidrológico Semanal, MITECO) ─────────
# data/boletin_embalses.csv lo genera scripts/actualizar_boletin.py. Solo cubre
# embalses de más de 5 hm³, así que el resto se queda con la lámina por satélite.
BOLETIN_CSV = ROOT / "data" / "boletin_embalses.csv"


def _boletin() -> dict:
    if not BOLETIN_CSV.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(BOLETIN_CSV)
        out = {}
        for r in df.itertuples():
            pct = getattr(r, "pct", None)
            if pct is None or (isinstance(pct, float) and math.isnan(pct)):
                continue
            out[str(r.hiblooms_id)] = {
                "pct": round(float(pct), 1), "hm3": float(r.volumen_hm3),
                "cap_hm3": float(r.capacidad_hm3), "fecha": str(r.fecha), "fuente": "Boletín Hidrológico",
            }
        return out
    except Exception:
        return {}


BOLETIN_HIST = ROOT / "data" / "boletin_historico_hiblooms.csv"


@lru_cache(maxsize=4)
def _boletin_hist():
    """{hiblooms_id: DataFrame(fecha, pct)} para calcular la tendencia del nivel."""
    if not BOLETIN_HIST.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(BOLETIN_HIST, parse_dates=["fecha"])
        return {k: v.sort_values("fecha")[["fecha", "pct"]] for k, v in df.groupby("hiblooms_id")}
    except Exception:
        return {}


def _boletin_trend(hid: str, days: int) -> Optional[float]:
    """Cambio en puntos de capacidad entre el último dato y el de hace `days` días."""
    h = _boletin_hist().get(hid)
    if h is None or len(h) < 2:
        return None
    try:
        import pandas as pd
        last = h.iloc[-1]
        ref_date = last["fecha"] - pd.Timedelta(days=days)
        prev = h[h["fecha"] <= ref_date]
        if prev.empty:
            return None
        return round(float(last["pct"]) - float(prev.iloc[-1]["pct"]), 1)
    except Exception:
        return None


MONITOR_BATCH = 6      # embalses por llamada a GEE (memoria)
MONITOR_SCALE = 30     # m por píxel en el resumen general (20 m es el nativo)
LOW_FILL = 60.0        # lámina por debajo del 60 % de la habitual = embalse tocado


def _combined(level: str, fill_pct: Optional[float]) -> bool:
    """Señal alta Y embalse bajo: el cruce que interesa mirar primero."""
    return level in ("alto", "muy alto") and fill_pct is not None and fill_pct < LOW_FILL


def _with_boletin(row: dict, bol: dict, days: int = 30) -> dict:
    """Añade el volumen oficial si ese embalse está en el boletín; manda sobre la lámina."""
    b = bol.get(row["id"])
    row["oficial"] = b
    nivel = b["pct"] if b else row.get("fill_pct")
    row["nivel_pct"] = nivel
    row["nivel_fuente"] = "boletin" if b else ("satelite" if row.get("fill_pct") is not None else None)
    # tendencia del nivel: puntos de capacidad (boletín) o % de lámina (satélite)
    row["nivel_trend"] = _boletin_trend(row["id"], days) if b else row.get("trend_pct")
    row["combined"] = _combined(row["level"], nivel)
    return row


class MonitorReq(BaseModel):
    days: int = Field(30, ge=5, le=120)
    max_cloud: int = Field(60, ge=0, le=100)
    reservoirs: List[str] = Field(default_factory=list, max_length=300)


def _monitor_geoms(names: List[str], user: Optional[str]) -> list:
    """[(id, label, geometría shapely)] de los embalses a vigilar."""
    gdf = _reservoirs_gdf()
    out = [(str(r.NOMBRE), niceish(str(r.NOMBRE)), r.geometry) for r in gdf.itertuples()]
    for u in _list_uploads(user):
        try:
            g = _upload_gdf(u["upload_id"])
            out += [(f"{CUSTOM_PREFIX}{u['upload_id']}:{i}", str(g['LABEL'].iloc[i]), g.geometry.iloc[i])
                    for i in range(len(g))]
        except Exception:
            pass
    if names:
        keep = set(names)
        out = [o for o in out if o[0] in keep]
    return out


def niceish(n: str) -> str:
    m = re.match(r"^(.*), (LA|EL|LAS|LOS)$", n)
    s = f"{m.group(2)} {m.group(1)}" if m else n
    return s.title()


@app.post("/api/monitor")
def monitor(req: MonitorReq, request: Request, wait: bool = False):
    """Por defecto devuelve {job_id}; con ?wait=1 calcula y devuelve el resultado."""
    user = request.state.user
    if wait:
        return _do_monitor(req, user, lambda *_a, **_k: None)
    return _start_job("monitor", lambda prog: _do_monitor(req, user, prog))


def _do_monitor(req: MonitorReq, user: Optional[str], prog):
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=req.days)
    prev_start = start - timedelta(days=req.days)      # ventana anterior, misma duración
    items = _monitor_geoms(req.reservoirs, user)
    if MOCK:
        rows = []
        for rid, label, _g in items:
            rng = _mock_rng("monitor", rid, start.isoformat())
            ndci = round(rng.uniform(-0.05, 0.45), 3)
            fill = round(rng.uniform(25, 105), 1)
            trend = round(rng.uniform(-18, 8), 1)
            rows.append({"id": rid, "label": label, "ndci_mean": round(ndci - rng.uniform(0.02, 0.12), 3),
                         "ndci_p90": ndci, "pci_p90": round(rng.uniform(0.9, 2.1), 2),
                         "date": (end - timedelta(days=rng.randint(1, max(2, req.days // 2)))).isoformat(),
                         "coverage": round(rng.uniform(35, 100), 1), "level": _level(ndci),
                         "cyano": rng.random() < .3,
                         "water_ha": round(rng.uniform(40, 1200), 1), "fill_pct": fill, "trend_pct": trend,
                         "ndci_trend": round(rng.uniform(-0.12, 0.12), 3),
                         "pci_trend": round(rng.uniform(-0.4, 0.4), 2),
                         "combined": _combined(_level(ndci), fill)})
        bol = _boletin()
        rows = [_with_boletin(r, bol, req.days) for r in rows]
        rows.sort(key=lambda r: (not r["combined"], -(r["ndci_p90"] or -9)))
        return {"mode": "demo", "start": start.isoformat(), "end": end.isoformat(),
                "days": req.days, "rows": rows}

    import ee
    feats = [ee.Feature(ee.Geometry(g.simplify(0.0004).__geo_interface__, geodesic=False), {"rid": rid})
             for rid, _l, g in items]
    fc = ee.FeatureCollection(feats)
    labels = {rid: label for rid, label, _g in items}

    def bands(img):
        """Índices + lámina de agua del día (erosionada 20 m) + marca de píxel despejado."""
        scl = img.select("SCL")
        clear = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)) \
                   .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        b = img.select(["B3", "B4", "B5", "B8A", "B11"]).divide(10000)
        b3, b4, b5, b8a, b11 = b.select("B3"), b.select("B4"), b.select("B5"), b.select("B8A"), b.select("B11")
        ndwi = b3.subtract(b8a).divide(b3.add(b8a))
        mndwi = b3.subtract(b11).divide(b3.add(b11))
        water = mndwi.gt(0).Or(ndwi.gt(0)).focal_min(radius=SHORE_ERODE_M, units="meters")
        return clear, water, b4, b5, img

    def prep(img):
        clear, water, b4, b5, im = bands(img)
        m = clear.And(water)
        ndci = b5.subtract(b4).divide(b5.add(b4)).rename("ndci")
        pci = b5.divide(b4).rename("pci")
        ts = ee.Image.constant(ee.Number(im.get("system:time_start"))).divide(1e6).toFloat().rename("ts")
        return ee.Image.cat(ndci, pci, ts).updateMask(m)

    def wmask(img):
        """1 donde se vio agua despejada ese día (para el área de lámina de la ventana)."""
        clear, water, _b4, _b5, _im = bands(img)
        return clear.And(water).selfMask().rename("w")

    def window(a, b):
        return (ee.ImageCollection(S2).filterBounds(fc).filterDate(str(a), str(b))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", req.max_cloud)))

    col, col_prev = window(start, end), window(prev_start, start)
    latest = col.map(prep).qualityMosaic("ts")       # píxel despejado más reciente de la ventana
    latest_prev = (col_prev.map(prep).qualityMosaic("ts").select(["ndci", "pci"])
                   .rename(["ndci_ant", "pci_ant"]))
    w_now = col.map(wmask).max().rename("wnow").selfMask()      # agua vista en cualquier día: menos sesgo por nubes
    w_prev = col_prev.map(wmask).max().rename("wprev").selfMask()
    # lámina habitual de referencia: JRC Global Surface Water, agua presente ≥ 50 % del tiempo (1984–2021)
    ref = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).rename("ref").selfMask()

    comp = ee.Image.cat(latest, latest_prev, w_now, w_prev, ref, ee.Image(1).rename("tot"))
    red = (ee.Reducer.mean().combine(ee.Reducer.percentile([90]), sharedInputs=True)
           .combine(ee.Reducer.count(), sharedInputs=True))
    # En lotes: una sola llamada con 30 embalses agota la memoria de Earth Engine.
    stats = []
    lotes = max(1, math.ceil(len(feats) / MONITOR_BATCH))
    for k, i in enumerate(range(0, len(feats), MONITOR_BATCH)):
        prog(90 * k / lotes, f"Embalses {i + 1}–{min(i + MONITOR_BATCH, len(feats))} de {len(feats)}")
        lote = ee.FeatureCollection(feats[i:i + MONITOR_BATCH])
        stats += comp.reduceRegions(collection=lote, reducer=red,
                                    scale=MONITOR_SCALE, tileScale=4).getInfo()["features"]

    PX_HA = MONITOR_SCALE * MONITOR_SCALE / 10000.0
    rows = []
    for f in stats:
        p = f["properties"]
        rid = p.get("rid")
        n_ok, n_tot = p.get("ndci_count") or 0, p.get("tot_count") or 0
        n_now, n_prev, n_ref = p.get("wnow_count") or 0, p.get("wprev_count") or 0, p.get("ref_count") or 0
        ndci90 = p.get("ndci_p90")
        ts = p.get("ts_p90") or p.get("ts_mean")
        day = datetime.utcfromtimestamp(float(ts) * 1e6 / 1000).strftime("%Y-%m-%d") if ts else None
        cov = round(100 * n_ok / n_tot, 1) if n_tot else 0.0
        enough = n_ok >= 8 and cov >= 10
        fill = round(min(150.0, 100 * n_now / n_ref), 1) if n_ref >= 8 and n_now else None
        ant90, n_ant = p.get("ndci_ant_p90"), p.get("ndci_ant_count") or 0
        ndci_trend = (round(float(ndci90) - float(ant90), 3)
                      if enough and ndci90 is not None and ant90 is not None and n_ant >= 8 else None)
        pci90, pci_ant = p.get("pci_p90"), p.get("pci_ant_p90")
        pci_trend = (round(float(pci90) - float(pci_ant), 2)
                     if enough and pci90 is not None and pci_ant is not None and n_ant >= 8 else None)
        trend = round(100 * (n_now - n_prev) / n_prev, 1) if n_prev >= 8 and n_now else None
        lvl = _level(ndci90 if enough else None)
        rows.append({
            "id": rid, "label": labels.get(rid, rid),
            "ndci_mean": round(p["ndci_mean"], 3) if enough and p.get("ndci_mean") is not None else None,
            "ndci_p90": round(ndci90, 3) if enough and ndci90 is not None else None,
            "pci_p90": round(p["pci_p90"], 2) if enough and p.get("pci_p90") is not None else None,
            "date": day if enough else None, "coverage": cov, "level": lvl,
            "cyano": bool(enough and (p.get("pci_p90") or 0) >= PCI_SUSPECT),
            "water_ha": round(n_now * PX_HA, 1) if n_now else None,
            "fill_pct": fill, "trend_pct": trend, "ndci_trend": ndci_trend, "pci_trend": pci_trend,
            "combined": _combined(lvl, fill),
        })
    bol = _boletin()
    rows = [_with_boletin(r, bol, req.days) for r in rows]
    # primero los que juntan señal alta y embalse bajo; luego por señal
    rows.sort(key=lambda r: (not r["combined"], -(r["ndci_p90"] if r["ndci_p90"] is not None else -9)))
    return {"mode": "gee", "start": start.isoformat(), "end": end.isoformat(), "days": req.days, "rows": rows}


# ── Climatología: ¿lo de este año es normal para esta época? ────────────────
# Solo NDCI y PCI: son comparables entre años y embalses. Los índices calibrados
# son locales (El Val, Bellús) y su serie histórica en otro embalse no significa nada.
# Todas las imágenes válidas de cada año, percentil 90 sobre la lámina de agua
# (igual que el monitor, para no mezclar criterios), agrupadas por quincenas.
CLIM_INDICES = {"NDCI_ind": "ndci", "PCI_B5/B4": "pci"}
CLIM_SCALE = 30        # m por píxel


class ClimReq(BaseModel):
    reservoir: str
    index: str
    years: int = Field(8, ge=2, le=15)
    max_cloud: int = Field(40, ge=0, le=100)
    water_only: bool = True


def _clim_cache(key: str) -> Path:
    return UPLOADS / f"clim_{key}.json"


def _quincena(d: date) -> int:
    """1–24: dos tramos por mes (días 1–15 y 16–fin)."""
    return (d.month - 1) * 2 + (1 if d.day <= 15 else 2)


@app.post("/api/climatology")
def climatology(req: ClimReq, wait: bool = False):
    """Por defecto devuelve {job_id}; con ?wait=1 calcula y devuelve el resultado."""
    if wait:
        return _do_climatology(req, lambda *_a, **_k: None)
    return _start_job("climatology", lambda prog: _do_climatology(req, prog))


def _do_climatology(req: ClimReq, prog):
    _check(req.reservoir)
    if req.index not in CLIM_INDICES:
        raise HTTPException(400, "La climatología solo está disponible para NDCI y PCI: "
                                 "los índices calibrados son locales y su histórico no es comparable.")
    meta = _meta(req.index)
    end = date.today()
    start = date(end.year - req.years + 1, 1, 1)

    if MOCK:
        pts = []
        d = start
        while d <= end:
            rng = _mock_rng("clim", req.reservoir, req.index, d.isoformat())
            if rng.random() < .45:
                season = 0.5 + 0.5 * math.sin((d.timetuple().tm_yday - 200) / 365 * 2 * math.pi)
                v = meta["min"] + (meta["max"] - meta["min"]) * (0.1 + 0.7 * season * rng.uniform(.4, 1.2))
                pts.append({"date": d.isoformat(), "q": _quincena(d), "year": d.year, "value": round(v, 3)})
            d += timedelta(days=5)
        return {"mode": "demo", "index": req.index, "unit": meta["unit"], "label": meta["label"],
                "years": [start.year, end.year], "points": pts}

    import hashlib
    key = hashlib.md5(f"v2|{req.reservoir}|{req.index}|{req.years}|{req.max_cloud}|{req.water_only}".encode()).hexdigest()[:10]
    f = _clim_cache(key)
    if f.exists() and (datetime.now().timestamp() - f.stat().st_mtime) < 86400:
        return json.loads(f.read_text(encoding="utf-8"))

    import ee
    band = CLIM_INDICES[req.index]
    aoi = _aoi(req.reservoir)

    def stat(img):
        """Percentil 90 del índice sobre la lámina de agua despejada de esa imagen."""
        scl = img.select("SCL")
        clear = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8))
                 .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
        b = img.select(["B3", "B4", "B5", "B8A", "B11"]).divide(10000)
        b3, b4, b5, b8a, b11 = b.select("B3"), b.select("B4"), b.select("B5"), b.select("B8A"), b.select("B11")
        ndwi = b3.subtract(b8a).divide(b3.add(b8a))
        mndwi = b3.subtract(b11).divide(b3.add(b11))
        water = mndwi.gt(0).Or(ndwi.gt(0))
        if req.water_only:
            water = water.focal_min(radius=SHORE_ERODE_M, units="meters")
        ix = (b5.subtract(b4).divide(b5.add(b4)) if band == "ndci" else b5.divide(b4)).rename("v")
        r = ix.updateMask(clear.And(water)).reduceRegion(
            ee.Reducer.percentile([90]).combine(ee.Reducer.count(), sharedInputs=True),
            aoi, CLIM_SCALE, maxPixels=1e13, tileScale=4)
        return ee.Feature(None, {"t": img.get("system:time_start"), "v": r.get("v_p90"), "n": r.get("v_count")})

    pts = []
    try:
        n_años = end.year - start.year + 1
        for k, y in enumerate(range(start.year, end.year + 1)):
            prog(95 * k / n_años, f"Año {y}")
            a, b_ = date(y, 1, 1), min(end + timedelta(days=1), date(y + 1, 1, 1))
            col = (ee.ImageCollection(S2).filterBounds(aoi).filterDate(str(a), str(b_))
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", req.max_cloud)))
            # Una sola llamada por año: Earth Engine recorre la colección en su servidor.
            for ft in col.map(stat).getInfo()["features"]:
                p = ft["properties"]
                if p.get("v") is None or (p.get("n") or 0) < 20:
                    continue
                d = datetime.utcfromtimestamp(p["t"] / 1000).date()
                pts.append({"date": d.isoformat(), "q": _quincena(d), "year": d.year, "value": round(float(p["v"]), 4)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")

    out = {"mode": "gee", "index": req.index, "unit": meta["unit"], "label": meta["label"],
           "years": [start.year, end.year], "points": sorted(pts, key=lambda r: r["date"])}
    f.write_text(json.dumps(out), encoding="utf-8")
    return out



# ── Base de datos del proyecto (pestaña «Datos», solo lectura) ───────────────
import projectdb as pdb  # noqa: E402  (web/backend/projectdb.py)


@app.on_event("startup")
def _warm_db():
    pdb.warm()  # lee la base al arrancar: el primer usuario ya no espera
from fastapi.responses import PlainTextResponse  # noqa: E402


def _pdb(fn, *a, **k):
    try:
        return fn(*a, **k)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Error en la base de datos del proyecto: {e}")


@app.get("/api/db/status")
def db_status():
    return pdb.status()


@app.get("/api/db/sites")
def db_sites(water_body: Optional[str] = None):
    return _pdb(pdb.sites, water_body)


@app.get("/api/db/parameters")
def db_parameters(water_body: Optional[str] = None, depth: str = "surface"):
    return _pdb(pdb.parameters, water_body, depth)


@app.get("/api/db/series")
def db_series(parameter: str, water_body: Optional[str] = None, sites: Optional[str] = None,
              depth: str = "surface"):
    return _pdb(pdb.series, parameter, water_body, sites.split(",") if sites else None, depth)


@app.get("/api/db/export")
def db_export(parameter: Optional[str] = None, water_body: Optional[str] = None):
    csv = _pdb(pdb.export_csv, parameter, water_body)
    name = "hiblooms_" + "_".join(x for x in (water_body, parameter) if x).replace(" ", "-") + ".csv"
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name or "hiblooms_datos.csv"}"'})


@app.get("/api/db/sources")
def db_sources():
    return _pdb(pdb.sources)


@app.get("/api/db/campaigns")
def db_campaigns(water_body: Optional[str] = None):
    """Campañas y visitas de muestreo, con lo que se tomó en cada visita."""
    return _pdb(pdb.campaigns, water_body)


@app.get("/api/db/phyto")
def db_phyto(water_body: Optional[str] = None, sites: Optional[str] = None, metric: str = "biovolume"):
    return _pdb(pdb.phyto, water_body, [x for x in (sites or "").split(",") if x] or None, metric)


@app.get("/api/db/cores")
def db_cores(water_body: Optional[str] = None):
    return _pdb(pdb.cores, water_body)


@app.get("/api/db/core/{core_id}")
def db_core(core_id: int):
    return _pdb(pdb.core, core_id)


@app.get("/api/db/sensors")
def db_sensors():
    return _pdb(pdb.sensor_reservoirs)


@app.get("/api/db/sensors/{reservoir_id}")
def db_sensor_series(reservoir_id: int, variable: str = "phycocyanin", layer: str = "surface"):
    return _pdb(pdb.sensor_series, reservoir_id, variable, layer)


@app.post("/api/db/reload")
def db_reload():
    """Vuelve a leer la base de datos (tras cargar datos nuevos)."""
    _pdb(pdb.data, True)
    return pdb.status()


# ── Frontend estático (mismo servicio que la API, un solo enlace) ────────────
# En el despliegue (Docker) el frontend compilado vive en web/frontend/dist.
FRONT = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONT.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=FRONT / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        """Cualquier ruta que no sea /api devuelve la web (React se encarga del resto)."""
        f = FRONT / full_path
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(FRONT / "index.html")
