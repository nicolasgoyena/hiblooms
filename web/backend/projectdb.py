"""
Acceso de SOLO LECTURA a la base de datos del proyecto (PostgreSQL/PostGIS,
unav_water_sampling) para la pestaña «Datos» del visor.

Conexión: variable de entorno DATABASE_URL, por ejemplo
    postgresql+psycopg2://hiblooms_ro:CLAVE@host:5432/unav_water_sampling
Conviene usar un rol que solo tenga permisos SELECT. Además, cada transacción
se abre en modo READ ONLY, así que aunque el rol tuviera más permisos la web
no podría modificar nada.

La físico-química está repartida en tres tablas con formatos distintos:
  · insitu_sampling        (ancha: una columna por parámetro, una fila por muestreo)
  · insitu_determinations  (ancha, con profundidad: sonda multiparamétrica)
  · lab_measurements       (larga: parameter_code + value, vía lab_samples)
Aquí se unifican en una tabla larga de observaciones:
  punto · fecha · profundidad · parámetro · valor · unidad · qc_flag · origen
Son pocas miles de filas, así que se cargan una vez y se cachean unos minutos.

Sin DATABASE_URL (o con HIBLOOMS_DB_MOCK=1) se generan datos simulados que
respetan el esquema, para desarrollar y probar sin la base real.
"""
from __future__ import annotations

import math
import os
import random
import threading
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd

CACHE_TTL = 6 * 3600  # s: pasado este tiempo se refresca en segundo plano

# Columnas numéricas de las tablas anchas que NO son parámetros
NOT_PARAMS = {
    "extraction_point_id", "qc_flag", "depth_m", "water_depth_meters",
}

_lock = threading.Lock()
_cache: dict = {"t": 0.0, "data": None, "refreshing": False}
_engine = None


def db_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or None


def is_mock() -> bool:
    return os.getenv("HIBLOOMS_DB_MOCK") == "1" or not db_url()


def engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        url = db_url()
        if url.startswith("postgres://"):  # Neon/Heroku dan a veces este prefijo
            url = "postgresql+psycopg2://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        _engine = create_engine(
            url, pool_pre_ping=True, pool_size=2, max_overflow=2,
            # toda sesión en solo lectura, pase lo que pase con los permisos del rol
            # (sin "options": el pooler de Neon no admite parámetros de arranque)
        )
        from sqlalchemy import event

        @event.listens_for(_engine, "connect")
        def _ro(dbapi_conn, _rec):  # toda sesión en solo lectura, pase lo que pase con el rol
            cur = dbapi_conn.cursor()
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("SET statement_timeout = 120000")
            cur.close()
            dbapi_conn.commit()
    return _engine


def q(sql: str, **params) -> pd.DataFrame:
    from sqlalchemy import text
    with engine().connect() as c:
        return pd.read_sql(text(sql), c, params=params)


# ── Carga desde la base real ────────────────────────────────────────────────

def _numeric_cols(table: str) -> list[str]:
    df = q("""SELECT column_name FROM information_schema.columns
              WHERE table_schema='public' AND table_name=:t
                AND data_type IN ('numeric','double precision','real','integer','smallint','bigint')
              ORDER BY ordinal_position""", t=table)
    return [c for c in df["column_name"] if c not in NOT_PARAMS and not c.endswith("_id")]


def _unpivot(table: str, depth_expr: str, date_expr: Optional[str] = None,
             exclude: tuple = (), rename: Optional[dict] = None) -> str:
    """SQL que pasa una tabla ancha a formato largo con LATERAL (VALUES ...).
    depth_expr: expresión SQL de la profundidad de la medida (o '0' si es de superficie).
    date_expr: fecha propia de la tabla; si no, se toma de samples vía extraction_id.
    rename: columnas que se publican con otro código (p. ej. dos conductividades distintas)."""
    rename = rename or {}
    cols = [c for c in _numeric_cols(table) if c not in exclude]
    if not cols:
        return ""
    vals = ", ".join(f"('{rename.get(c, c)}', t.\"{c}\"::double precision)" for c in cols)
    if date_expr:
        date_sql, join = date_expr, ""
    else:
        date_sql = "s.sample_date"
        join = """LEFT JOIN (SELECT DISTINCT ON (extraction_id) extraction_id, sample_date
                 FROM samples ORDER BY extraction_id, sample_date) s
        ON s.extraction_id = t.extraction_id"""
    return f"""
      SELECT t.extraction_point_id, {date_sql} AS date, ({depth_expr})::double precision AS depth_m,
             '{table}' AS source_table, v.col AS column_name, v.val AS value,
             t.qc_flag, t.source_code, NULL::text AS unit_raw, 'water'::text AS matrix_code
      FROM {table} t
      {join}
      CROSS JOIN LATERAL (VALUES {vals}) AS v(col, val)
      WHERE v.val IS NOT NULL"""


def _load_real() -> dict:
    # Puntos: agrupamos las extracciones por punto de muestreo fijo cuando lo tienen
    points = q("""
      SELECT ep.extraction_point_id, ep.sampling_point_id, ep.campaign_id,
             ep.water_body_name, ep.location_code, ep.reservoir_id, ep.river_id,
             COALESCE(ep.latitude,  ST_Y(ST_Transform(ST_PointOnSurface(ep.geometry), 4326))) AS lat,
             COALESCE(ep.longitude, ST_X(ST_Transform(ST_PointOnSurface(ep.geometry), 4326))) AS lon,
             sp.point_code, sp.point_number,
             ST_Y(ST_Transform(ST_PointOnSurface(sp.geometry), 4326)) AS sp_lat,
             ST_X(ST_Transform(ST_PointOnSurface(sp.geometry), 4326)) AS sp_lon,
             c.campaign_code, c.start_date AS campaign_start
      FROM extraction_points ep
      LEFT JOIN sampling_points sp ON sp.sampling_point_id = ep.sampling_point_id
      LEFT JOIN campaigns c ON c.campaign_id = ep.campaign_id""")

    parts = [p for p in (
        # insitu_sampling: medidas de SUPERFICIE. Su water_depth_meters es la profundidad
        # TOTAL de la columna en el punto, no la de la medida.
        _unpivot("insitu_sampling", "0", exclude=("chlorophyll_volume_estimation",)),
        # Sonda multiparamétrica (Hanna): perfiles con profundidad real. Sus dos
        # conductividades no son la misma magnitud: se publican por separado.
        _unpivot("insitu_determinations", "t.depth_m", rename={"conduc_uscm_2": "conduc_uscm_2"}),
        # Sonda de fluorescencia (FluoroProbe): perfiles finos de clorofila por grupo algal.
        # Su temperatura se omite para no mezclar dos instrumentos en el mismo perfil.
        _unpivot("profiles_data", "t.depth_m", date_expr="t.datetime::date",
                 exclude=("sample_temperature_celsius",)),
    ) if p]
    # Laboratorio: agua, material en suspensión y sedimento superficial. Los testigos
    # de sedimento (secciones en cm) quedan fuera: su profundidad no es la del agua.
    parts.append("""
      SELECT ls.extraction_point_id, ls.sample_date AS date,
             CASE WHEN ls.matrix_code = 'water' THEN COALESCE(ls.depth_m, 0) END::double precision AS depth_m,
             'lab_measurements' AS source_table, m.parameter_code AS column_name,
             m.value::double precision AS value, m.qc_flag, m.source_code, m.unit_code AS unit_raw,
             ls.matrix_code
      FROM lab_measurements m JOIN lab_samples ls ON ls.lab_sample_id = m.lab_sample_id
      WHERE m.value IS NOT NULL AND COALESCE(ls.matrix_code, 'water') <> 'sediment_core'""")
    obs = q(" UNION ALL ".join(parts))

    # Vocabulario: columna de tabla ancha → parámetro normalizado y unidad
    vocab = q("SELECT table_name, column_name, parameter_code, unit_code FROM variable_vocabulary")
    params = q("SELECT parameter_code, name, parameter_group, default_unit FROM parameters")
    units = q("SELECT unit_code, name AS unit_name FROM units")
    qc = q("SELECT qc_flag, label, description FROM qc_flags ORDER BY qc_flag")
    reservoirs = q("""SELECT DISTINCT r.reservoir_id, r.reservoir_name
                      FROM reservoirs_spain r JOIN extraction_points ep ON ep.reservoir_id = r.reservoir_id""")
    sources = q("SELECT source_code, name, provider, licence, citation FROM data_sources")
    matrices = q("SELECT matrix_code, name FROM matrices")
    return _assemble(points, obs, vocab, params, units, qc, reservoirs, sources, matrices)


# Caja aproximada de España (península, Baleares y Canarias) en grados
_ES = (-19.0, 26.5, 5.0, 44.5)  # lon_min, lat_min, lon_max, lat_max
_UTM = None


def _in_es(lon, lat) -> bool:
    return _ES[0] <= lon <= _ES[2] and _ES[1] <= lat <= _ES[3]


def _fix_coord(lat, lon):
    """Devuelve (lat, lon) en grados WGS84 o (None, None).
    Corrige los dos errores habituales en datos de campo: latitud y longitud
    cruzadas, y coordenadas en UTM 30N ETRS89 (EPSG:25830) guardadas como si
    fueran grados."""
    global _UTM
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None
    if math.isnan(lat) or math.isnan(lon):
        return None, None
    if _in_es(lon, lat):
        return lat, lon
    if _in_es(lat, lon):                      # cruzadas
        return lon, lat
    # ¿UTM? x ~ 100 000–1 000 000 m, y ~ 3 000 000–4 900 000 m (en cualquier orden)
    for x, y in ((lon, lat), (lat, lon)):
        if 100_000 <= x <= 1_000_000 and 3_000_000 <= y <= 4_900_000:
            if _UTM is None:
                from pyproj import Transformer
                _UTM = Transformer.from_crs(25830, 4326, always_xy=True)
            lo, la = _UTM.transform(x, y)
            if _in_es(lo, la):
                return la, lo
    return None, None


def _fix_cols(df: pd.DataFrame, lat: str, lon: str) -> None:
    fixed = [_fix_coord(a, b) for a, b in zip(df[lat], df[lon])]
    df[lat] = [f[0] for f in fixed]
    df[lon] = [f[1] for f in fixed]


# Nombres legibles de los grupos de parameters.parameter_group
GROUP_NAMES = {
    "field": "Campo", "pigments": "Pigmentos", "nutrients": "Nutrientes y carbono",
    "toxins": "Toxinas", "icp_ms": "Metales y elementos (ICP-MS)", "optics": "Óptica",
    "stable_isotopes": "Isótopos estables", "elemental": "Análisis elemental",
    "radioisotopes": "Radioisótopos", "sediment_physics": "Física del sedimento",
    "sediment_chemistry": "Química del sedimento", "organic_matter": "Materia orgánica",
    "spm": "Material en suspensión", "meteorology": "Meteorología",
    "remote_sensing": "Teledetección",
}
# Parámetros que no están en variable_vocabulary con su propio código
EXTRA_PARAMS = {
    "conduc_uscm_2": ("Conductividad eléctrica (2)", "field", "uS/cm"),
}


def _assemble(points, obs, vocab, params, units, qc, reservoirs, sources, matrices=None) -> dict:
    # Sitio = punto de muestreo fijo si existe; si no, la propia extracción
    points = points.copy()
    _fix_cols(points, "lat", "lon")
    _fix_cols(points, "sp_lat", "sp_lon")
    points["site"] = points.apply(
        lambda r: f"sp{int(r.sampling_point_id)}" if pd.notna(r.sampling_point_id) else f"ep{int(r.extraction_point_id)}", axis=1)
    points["site_code"] = points["point_code"].fillna(points["location_code"]).fillna(points["site"])
    points["site_lat"] = points["sp_lat"].fillna(points["lat"])
    points["site_lon"] = points["sp_lon"].fillna(points["lon"])
    ep2site = dict(zip(points.extraction_point_id, points.site))

    # Parámetro normalizado
    vmap = {(r.table_name, r.column_name): (r.parameter_code, r.unit_code) for r in vocab.itertuples()}
    def norm(r):
        if r.column_name in EXTRA_PARAMS:
            return r.column_name, EXTRA_PARAMS[r.column_name][2]
        if r.source_table == "lab_measurements":
            return r.column_name, getattr(r, "unit_raw", None)
        return vmap.get((r.source_table, r.column_name), (r.column_name, None))
    obs = obs.copy()
    pc = obs.apply(norm, axis=1, result_type="expand")
    obs["parameter_code"], obs["unit_code"] = pc[0], pc[1]
    extra = pd.DataFrame([(k, n, g, u) for k, (n, g, u) in EXTRA_PARAMS.items()],
                         columns=["parameter_code", "name", "parameter_group", "default_unit"])
    params = pd.concat([params, extra[~extra.parameter_code.isin(params.parameter_code)]], ignore_index=True)
    pinfo = params.set_index("parameter_code")
    obs["unit_code"] = obs["unit_code"].fillna(obs["parameter_code"].map(pinfo["default_unit"]))
    obs["site"] = obs["extraction_point_id"].map(ep2site)
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date", "site", "value"])

    unames = dict(zip(units.unit_code, units.unit_name))
    if "matrix_code" not in obs:
        obs["matrix_code"] = "water"
    obs["matrix_code"] = obs["matrix_code"].fillna("water")
    # Un mismo código en agua y en sedimento son parámetros distintos para el visor
    obs.loc[obs.matrix_code != "water", "parameter_code"] = (
        obs.loc[obs.matrix_code != "water", "matrix_code"] + ":" + obs.loc[obs.matrix_code != "water", "parameter_code"])
    catalog = (obs.groupby("parameter_code")
               .agg(n=("value", "size"), sites=("site", "nunique"),
                    first=("date", "min"), last=("date", "max"), unit=("unit_code", "first"))
               .reset_index())
    base = catalog["parameter_code"].str.split(":").str[-1]
    mtx = catalog["parameter_code"].where(catalog["parameter_code"].str.contains(":"), "water:").str.split(":").str[0]
    mnames = dict(zip(matrices.matrix_code, matrices.name)) if matrices is not None else {}
    catalog["name"] = base.map(pinfo["name"]).fillna(base)
    grp = base.map(pinfo["parameter_group"]).map(lambda g: GROUP_NAMES.get(g, g) if isinstance(g, str) else "Otros")
    catalog["group"] = [g if m == "water" else f"{mnames.get(m, m)} · {g}" for g, m in zip(grp, mtx)]
    # Símbolo corto para los ejes y el desplegable (el nombre largo es para el catálogo)
    SHORT = {"degC": "°C", "percent": "%", "permil": "‰", "uS/cm": "µS/cm", "ug/L": "µg/L",
             "ug/g": "µg/g", "mg/m3": "mg/m³", "um3/mL": "µm³/mL", "g/cm2": "g/cm²", "W/m2": "W/m²"}
    catalog["unit_name"] = catalog["unit"].map(lambda u: SHORT.get(u, u) if isinstance(u, str) else None)

    return {"points": points, "obs": obs, "catalog": catalog, "qc": qc,
            "reservoirs": reservoirs, "sources": sources, "loaded": time.time()}


# ── Datos simulados (mismo esquema) ─────────────────────────────────────────

def _load_mock() -> dict:
    rng = random.Random(7)
    bodies = [("Embalse de El Val", 4, 41.87, -1.80), ("Embalse de Bellús", 3, 38.94, -0.49),
              ("Embalse de Alloz", 3, 42.71, -1.95), ("Río Queiles", 2, 41.95, -1.85)]
    rows, sid, eid = [], 1, 1
    for name, n, la, lo in bodies:
        for k in range(1, n + 1):
            for camp in range(6):
                rows.append(dict(extraction_point_id=eid, sampling_point_id=sid, campaign_id=camp + 1,
                                 water_body_name=name, location_code=f"{name[-4:].upper()}-{k}",
                                 reservoir_id=1, river_id=None, lat=la + rng.uniform(-.01, .01),
                                 lon=lo + rng.uniform(-.01, .01), point_code=f"{name.split()[-1][:3].upper()}-{k}",
                                 point_number=k, sp_lat=la + k * .004, sp_lon=lo + k * .004,
                                 campaign_code=f"C{camp + 1:02d}", campaign_start=date(2025, 1 + camp * 2, 10)))
                eid += 1
            sid += 1
    points = pd.DataFrame(rows)
    params = pd.DataFrame([
        ("WTEMP", "Temperatura del agua", "Físico-química", "degC"),
        ("PH", "pH", "Físico-química", "pH"),
        ("DOXY", "Oxígeno disuelto", "Físico-química", "mg_L"),
        ("COND", "Conductividad eléctrica", "Físico-química", "uS_cm"),
        ("SECCHI", "Profundidad de disco de Secchi", "Óptica", "m"),
        ("CHLA", "Clorofila a", "Pigmentos", "mg_m3"),
        ("NO3", "Nitrato", "Nutrientes", "mg_L"),
        ("NH4", "Amonio", "Nutrientes", "mg_L"),
        ("TN", "Nitrógeno total", "Nutrientes", "mg_L"),
        ("P", "Fósforo", "Nutrientes", "ppm"),
        ("MC", "Microcistinas (ELISA)", "Toxinas", "ng_mL"),
    ], columns=["parameter_code", "name", "parameter_group", "default_unit"])
    base = {"WTEMP": (8, 26), "PH": (7.6, 9.2), "DOXY": (6, 12), "COND": (350, 900), "SECCHI": (0.6, 3.5),
            "CHLA": (2, 60), "NO3": (0.2, 6), "NH4": (0.01, .4), "TN": (0.5, 4), "P": (.01, .2), "MC": (0, 2.5)}
    obs = []
    for r in points.itertuples():
        d0 = date(2024, 4, 1) + timedelta(days=45 * (r.campaign_id - 1) + rng.randint(0, 6))
        season = .5 + .5 * math.sin((d0.timetuple().tm_yday - 110) / 365 * 2 * math.pi)
        # perfiles simulados de la sonda: temperatura y oxígeno cada metro, con termoclina
        for z in range(0, 16):
            for code, surf, deep in (("WTEMP", 12 + 12 * season, 9.0), ("DOXY", 9 + rng.uniform(-.5, .5), 9 - 8 * season)):
                w = 1 / (1 + math.exp((z - 6) * 1.2))
                obs.append(dict(extraction_point_id=r.extraction_point_id, date=d0, depth_m=float(z) + .5,
                                source_table="insitu_determinations", column_name=code,
                                value=round(deep + (surf - deep) * w + rng.uniform(-.2, .2), 2), qc_flag=1,
                                source_code="HIBLOOMS"))
        for code, (lo, hi) in base.items():
            if code in ("WTEMP", "DOXY") or rng.random() < .1:
                continue
            v = lo + (hi - lo) * max(0, min(1, season * rng.uniform(.6, 1.2) + rng.uniform(-.1, .1)))
            obs.append(dict(extraction_point_id=r.extraction_point_id, date=d0, depth_m=.5,
                            source_table="lab_measurements" if code in ("NO3", "NH4", "TN", "P", "MC") else "insitu_sampling",
                            column_name=code, value=round(v, 3), qc_flag=rng.choices([1, 2, 3], [90, 7, 3])[0],
                            source_code="HIBLOOMS"))
    obs = pd.DataFrame(obs)
    vocab = pd.DataFrame(columns=["table_name", "column_name", "parameter_code", "unit_code"])
    units = pd.DataFrame([("degC", "°C"), ("pH", "pH"), ("mg_L", "mg/L"), ("uS_cm", "µS/cm"), ("m", "m"),
                          ("mg_m3", "mg/m³"), ("ppm", "ppm"), ("ng_mL", "ng/mL")], columns=["unit_code", "unit_name"])
    qc = pd.DataFrame([(1, "Bueno", "Valor validado"), (2, "Dudoso", "Revisar"), (3, "Malo", "No usar")],
                      columns=["qc_flag", "label", "description"])
    reservoirs = pd.DataFrame([(1, "EL VAL")], columns=["reservoir_id", "reservoir_name"])
    sources = pd.DataFrame([("HIBLOOMS", "Muestreos HIBLOOMS", "BIOMA-UNAV", "CC-BY 4.0", "HIBLOOMS (2026)")],
                           columns=["source_code", "name", "provider", "licence", "citation"])
    return _assemble(points, obs, vocab, params, units, qc, reservoirs, sources)


# ── API interna ─────────────────────────────────────────────────────────────

def _load() -> dict:
    return _load_mock() if is_mock() else _load_real()


def _refresh_bg() -> None:
    """Relee la base sin bloquear: mientras tanto se sirven los datos anteriores."""
    try:
        d = _load()
        with _lock:
            _cache["data"], _cache["t"] = d, time.time()
    except Exception as e:  # noqa: BLE001
        print("[projectdb] refresco fallido:", e)
    finally:
        _cache["refreshing"] = False


def data(force: bool = False) -> dict:
    with _lock:
        if force or _cache["data"] is None:
            _cache["data"] = _load()
            _cache["t"] = time.time()
        elif time.time() - _cache["t"] > CACHE_TTL and not _cache["refreshing"]:
            _cache["refreshing"] = True
            threading.Thread(target=_refresh_bg, daemon=True).start()
        return _cache["data"]


def warm() -> None:
    """Precarga en segundo plano al arrancar el servidor."""
    def _go():
        try:
            data()
            print("[projectdb] datos precargados")
        except Exception as e:  # noqa: BLE001
            print("[projectdb] precarga fallida:", e)
    threading.Thread(target=_go, daemon=True).start()


def _jsonable(df: pd.DataFrame) -> list:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict(orient="records")


def status() -> dict:
    try:
        d = data()
        return {"ok": True, "mode": "demo" if is_mock() else "db",
                "n_obs": int(len(d["obs"])), "n_sites": int(d["points"]["site"].nunique()),
                "n_params": int(len(d["catalog"])),
                "first": d["obs"]["date"].min().strftime("%Y-%m-%d") if len(d["obs"]) else None,
                "last": d["obs"]["date"].max().strftime("%Y-%m-%d") if len(d["obs"]) else None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "mode": "error", "detail": str(e)}


def sites(water_body: Optional[str] = None) -> dict:
    d = data()
    p, o = d["points"], d["obs"]
    agg = o.groupby("site").agg(n_obs=("value", "size"), first=("date", "min"), last=("date", "max"),
                                n_dates=("date", "nunique")).reset_index()
    s = (p.groupby("site").agg(code=("site_code", "first"), water_body=("water_body_name", "first"),
                               lat=("site_lat", "first"), lon=("site_lon", "first"),
                               n_campaigns=("campaign_id", "nunique"))
         .reset_index().merge(agg, on="site", how="left"))
    s["n_obs"] = s["n_obs"].fillna(0).astype(int)
    if water_body:
        s = s[s["water_body"] == water_body]
    s = s.dropna(subset=["lat", "lon"])
    bodies = (p.groupby("water_body_name")["site"].nunique().reset_index()
              .rename(columns={"water_body_name": "name", "site": "n_sites"}).sort_values("name"))
    return {"sites": _jsonable(s.sort_values(["water_body", "code"])), "water_bodies": _jsonable(bodies)}


def _depth_filter(o: pd.DataFrame, depth: str) -> pd.DataFrame:
    """surface: ≤ 1 m (o sin profundidad) · bottom: la más profunda de cada muestreo · all: todas."""
    if depth == "surface":
        return o[o["depth_m"].isna() | (o["depth_m"] <= 1.0)]
    if depth == "bottom":
        k = ["site", "date", "parameter_code"]
        mx = o.groupby(k)["depth_m"].transform("max")
        return o[o["depth_m"].isna() | (o["depth_m"] == mx)]
    return o


def parameters(water_body: Optional[str] = None, depth: str = "surface") -> dict:
    d = data()
    o = _depth_filter(d["obs"], depth)
    if water_body:
        keep = set(d["points"].loc[d["points"]["water_body_name"] == water_body, "site"])
        o = o[o["site"].isin(keep)]
    counts = o.groupby("parameter_code").size().rename("n_here")
    cat = d["catalog"].merge(counts, left_on="parameter_code", right_index=True, how="inner")
    cat = cat.sort_values(["group", "name"])
    return {"parameters": _jsonable(cat), "qc": _jsonable(d["qc"])}


def series(parameter: str, water_body: Optional[str] = None, sites_: Optional[list] = None,
           depth: str = "surface") -> dict:
    d = data()
    o = _depth_filter(d["obs"][d["obs"]["parameter_code"] == parameter], depth)
    p = d["points"].drop_duplicates("site").set_index("site")
    if water_body:
        o = o[o["site"].map(p["water_body_name"]) == water_body]
    if sites_:
        o = o[o["site"].isin(sites_)]
    o = o.assign(site_code=o["site"].map(p["site_code"]), water_body=o["site"].map(p["water_body_name"]))
    o = o.sort_values(["date", "site_code"])
    cat = d["catalog"].set_index("parameter_code")
    meta = cat.loc[parameter].to_dict() if parameter in cat.index else {}
    summ = (o.groupby("site_code")
            .agg(n=("value", "size"), n_dates=("date", "nunique"), min=("value", "min"),
                 median=("value", "median"), max=("value", "max")).reset_index())
    last = o.sort_values("date").groupby("site_code").tail(1)[["site_code", "date", "value"]]
    summ = summ.merge(last.rename(columns={"date": "last_date", "value": "last_value"}), on="site_code")
    cols = ["date", "site", "site_code", "water_body", "depth_m", "value", "qc_flag", "source_table", "source_code"]
    return {"parameter": parameter, "name": meta.get("name", parameter), "unit": meta.get("unit_name"),
            "n_dates": int(o[["site", "date"]].drop_duplicates().shape[0]), "depth": depth,
            "group": meta.get("group"), "rows": _jsonable(o[cols]), "summary": _jsonable(summ),
            "qc": _jsonable(d["qc"])}


def export_csv(parameter: Optional[str] = None, water_body: Optional[str] = None) -> str:
    d = data()
    o = d["obs"]
    p = d["points"].drop_duplicates("site").set_index("site")
    if parameter:
        o = o[o["parameter_code"] == parameter]
    if water_body:
        o = o[o["site"].map(p["water_body_name"]) == water_body]
    o = o.assign(site_code=o["site"].map(p["site_code"]), water_body=o["site"].map(p["water_body_name"]),
                 lat=o["site"].map(p["site_lat"]), lon=o["site"].map(p["site_lon"]))
    cols = ["water_body", "site_code", "lat", "lon", "date", "depth_m", "parameter_code", "value",
            "unit_code", "qc_flag", "source_table", "source_code"]
    out = o[cols].sort_values(["water_body", "site_code", "date", "parameter_code"]).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False)


def sources() -> dict:
    return {"sources": _jsonable(data()["sources"])}
