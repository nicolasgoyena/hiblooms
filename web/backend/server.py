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
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SHAPEFILE = str(ROOT / "shapefiles" / "embalses_hiblooms.shp")

import geopandas as gpd  # noqa: E402

# ── Catálogo de índices ──────────────────────────────────────────────────────
INDICES = [
    {"id": "PC_Val_cal", "label": "Ficocianina calibrada · El Val", "unit": "µg/L", "min": 0, "max": 100, "group": "Calibrados"},
    {"id": "Chla_Val_cal", "label": "Clorofila-a calibrada · El Val", "unit": "µg/L", "min": 0, "max": 150, "group": "Calibrados"},
    {"id": "PC_Bellus_cal", "label": "Ficocianina calibrada · Bellús", "unit": "µg/L", "min": 0, "max": 800, "group": "Calibrados"},
    {"id": "Chla_Bellus_cal", "label": "Clorofila-a calibrada · Bellús", "unit": "µg/L", "min": 0, "max": 80, "group": "Calibrados"},
    {"id": "UV_PC_Gral_cal", "label": "Ficocianina general (UV)", "unit": "µg/L", "min": 0, "max": 100, "group": "Calibrados"},
    {"id": "PCI_B5/B4", "label": "PCI (B5/B4)", "unit": "", "min": 0.5, "max": 3, "group": "Espectrales"},
    {"id": "NDCI_ind", "label": "NDCI", "unit": "", "min": -0.2, "max": 0.5, "group": "Espectrales"},
    {"id": "MCI", "label": "MCI", "unit": "", "min": -0.05, "max": 0.2, "group": "Espectrales"},
]
INDEX_BY_ID = {i["id"]: i for i in INDICES}
PALETTE = ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"]

# ── Inicialización GEE / modo demo ───────────────────────────────────────────
MOCK = os.getenv("HIBLOOMS_MOCK") == "1"
GEE_ERROR: Optional[str] = None
core = None
if not MOCK:
    try:
        import hiblooms_core as core  # noqa: E402

        sa = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        if not sa:  # mismo secrets.toml que usa la app Streamlit
            try:
                import tomllib  # Python ≥ 3.11
            except ModuleNotFoundError:
                import tomli as tomllib  # pip install tomli (Python ≤ 3.10)
            for cand in (ROOT / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"):
                if cand.exists():
                    sec = tomllib.loads(cand.read_text(encoding="utf-8"))
                    if "GEE_SERVICE_ACCOUNT_JSON" in sec:
                        sa = json.loads(sec["GEE_SERVICE_ACCOUNT_JSON"], strict=False)
                        break
        core.init_ee(sa if sa else None)
    except Exception as e:  # sin credenciales → demo
        GEE_ERROR = str(e)
        MOCK = True

app = FastAPI(title="HIBLOOMS web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utilidades ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _reservoirs_gdf() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs is None or (gdf.crs.to_epsg() or 0) != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


@lru_cache(maxsize=64)
def _aoi(name: str):
    gdf = core.load_reservoir_shapefile(name, SHAPEFILE)
    return core.gdf_to_ee_geometry(gdf)


def _check(name: str, index: Optional[str] = None):
    if name not in set(_reservoirs_gdf()["NOMBRE"]):
        raise HTTPException(404, f"Embalse desconocido: {name}")
    if index is not None and index not in INDEX_BY_ID:
        raise HTTPException(400, f"Índice desconocido: {index}")


def _mock_rng(*parts) -> random.Random:
    return random.Random("|".join(map(str, parts)))


# ── Imagen procesada (cacheada) ─────────────────────────────────────────────
ALL_INDEX_IDS = [i["id"] for i in INDICES]
POIS_CSV = ROOT / "data" / "puntos_interes.csv"


@lru_cache(maxsize=48)
def _processed(reservoir: str, day: str, max_cloud: int):
    """Mejor imagen del día con TODOS los índices (los objetos ee son perezosos)."""
    aoi = _aoi(reservoir)
    scaled, ind, iso, cloud, cov = core.process_sentinel2(aoi, day, max_cloud, ALL_INDEX_IDS)
    if ind is None:
        raise HTTPException(404, f"No hay imagen válida el {day} con esos umbrales")
    return scaled, ind, iso, cloud, cov


def _points_values(ind, index: str, points: list) -> dict:
    if not points:
        return {}
    import ee
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p.lon, p.lat]).buffer(30), {"name": p.name}) for p in points
    ])
    red = ind.select(index).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=20)
    out = {}
    for f in red.getInfo()["features"]:
        v = f["properties"].get("mean")
        out[f["properties"]["name"]] = float(v) if v is not None else None
    return out


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


class SeriesReq(BaseModel):
    reservoir: str
    index: str
    dates: List[date] = Field(..., max_length=120)
    max_cloud: int = Field(30, ge=0, le=100)
    points: List[Point] = Field(default_factory=list, max_length=50)


class ClassesReq(BaseModel):
    reservoir: str
    date: date
    index: str
    max_cloud: int = Field(30, ge=0, le=100)
    n_classes: int = Field(5, ge=2, le=10)


class DownloadReq(BaseModel):
    reservoir: str
    date: date
    indices: List[str] = Field(..., min_length=1)
    max_cloud: int = Field(30, ge=0, le=100)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True, "mode": "demo" if MOCK else "gee", "gee_error": GEE_ERROR}


@app.get("/api/indices")
def indices():
    return {"indices": INDICES, "palette": PALETTE}


@app.get("/api/reservoirs")
def reservoirs():
    gdf = _reservoirs_gdf()[["NOMBRE", "DEMARC", "PROVINCIA", "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.simplify(0.0002, preserve_topology=True)
    return json.loads(gdf.to_json())


@app.get("/api/pois")
def pois(reservoir: str):
    """Puntos de interés predefinidos (data/puntos_interes.csv) del embalse."""
    import pandas as pd
    if not POIS_CSV.exists():
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
        dates = core.get_available_dates(
            _aoi(req.reservoir), req.start.isoformat(), (req.end + timedelta(days=1)).isoformat(),
            req.max_cloud, req.min_coverage,
        )
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    return {"reservoir": req.reservoir, "dates": [{"date": d, "cloud": None} for d in dates], "mode": "gee"}


@app.post("/api/image")
def image(req: ImageReq):
    _check(req.reservoir, req.index)
    meta = INDEX_BY_ID[req.index]

    if MOCK:
        rng = _mock_rng(req.reservoir, req.date, req.index)
        return {
            "mode": "demo", "tile_url": None, "rgb_tile_url": None,
            "datetime": f"{req.date.isoformat()} 10:5{rng.randint(0, 9)}:00",
            "cloud": round(rng.uniform(0, req.max_cloud), 1), "coverage": round(rng.uniform(80, 100), 1),
            "mean": _mock_value(meta, req.reservoir, req.date, req.index),
            "points": {p.name: _mock_value(meta, req.reservoir, req.date, req.index, p.name) for p in req.points},
        }

    try:
        scaled, ind, iso, cloud, cov = _processed(req.reservoir, req.date.isoformat(), req.max_cloud)
        vis = {"min": meta["min"], "max": meta["max"], "palette": [p.lstrip("#") for p in PALETTE]}
        tile = ind.select(req.index).getMapId(vis)["tile_fetcher"].url_format
        rgb = scaled.select(["B4", "B3", "B2"]).getMapId({"min": 0, "max": 0.25, "gamma": 1.2})["tile_fetcher"].url_format
        mean = core.calcular_media_diaria_embalse(ind, req.index, _aoi(req.reservoir))
        pts = _points_values(ind, req.index, req.points)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    return {"mode": "gee", "tile_url": tile, "rgb_tile_url": rgb, "datetime": iso,
            "cloud": cloud, "coverage": cov, "mean": mean, "points": pts}


@app.post("/api/timeseries")
def timeseries(req: SeriesReq):
    _check(req.reservoir, req.index)
    meta = INDEX_BY_ID[req.index]
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

    aoi = _aoi(req.reservoir)
    for d in sorted(req.dates):
        row = {"date": d.isoformat(), "mean": None}
        try:
            _, ind, _, _, _ = _processed(req.reservoir, d.isoformat(), req.max_cloud)
            row["mean"] = core.calcular_media_diaria_embalse(ind, req.index, aoi)
            row.update(_points_values(ind, req.index, req.points))
        except Exception:
            pass
        out.append(row)
    return {"index": req.index, "series": out, "mode": "gee"}


@app.post("/api/classes")
def classes(req: ClassesReq):
    """Superficie del embalse por clases del índice (rangos iguales entre min y max)."""
    _check(req.reservoir, req.index)
    meta = INDEX_BY_ID[req.index]
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
        res = core.calcular_distribucion_area_por_clases(ind, req.index, _aoi(req.reservoir), bins)
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
    bad = [i for i in req.indices if i not in INDEX_BY_ID]
    if bad:
        raise HTTPException(400, f"Índices desconocidos: {bad}")
    if MOCK:
        raise HTTPException(400, "La descarga GeoTIFF necesita conexión a Earth Engine (modo demo)")
    try:
        _, ind, _, _, _ = _processed(req.reservoir, req.date.isoformat(), req.max_cloud)
        url = core.generar_url_geotiff_multibanda(ind, req.indices, _aoi(req.reservoir), scale=20)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error en Earth Engine: {e}")
    if not url:
        raise HTTPException(502, "Earth Engine no pudo generar la descarga (¿embalse demasiado grande?)")
    return {"url": url, "filename": f"{req.reservoir}_{req.date.isoformat()}.tif"}
