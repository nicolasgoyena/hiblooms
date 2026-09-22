"""
Actualiza sensor_data con la estación SAICA 945 del embalse de El Val (CHE).

Descarga desde el último dato que ya hay en la base hasta ayer, así que se puede
ejecutar cuando sea (a diario, una vez por semana o tras meses parado) y se pone al día.
Es idempotente: antes de insertar un tramo borra lo que hubiera de esta fuente en esas
mismas fechas, de modo que repetirlo no duplica datos.

Uso:
    set DATABASE_URL_WRITE=postgresql://usuario:clave@host/base
    python scripts/actualizar_sonda_val.py                  # desde el último dato hasta ayer
    python scripts/actualizar_sonda_val.py --desde 2026-06-01
    python scripts/actualizar_sonda_val.py --prueba         # descarga y muestra, sin escribir

Varias bases a la vez (p. ej. Neon y la local): separa las URL con ';' en DATABASE_URL_WRITE.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata

# La consola de Windows no siempre admite UTF-8: que los símbolos no rompan el script
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text

ESTACION = 945                    # estación SAICA del embalse de El Val
RESERVOIR_ID = 2351               # El Val en reservoirs_spain
# source_code de estos datos en la base. OJO: "che_aquadam_el_val" es la sonda de perfiles
# antigua (2018–2024), otra fuente distinta; la estación SAICA 945 continúa desde 2024.
SOURCE = "che_saica_945"
URL = "https://saica.chebro.es/fichaDataTabla.php?estacion={e}&fini={a}&ffin={b}"
TRAMO_DIAS = 7                    # días por petición (la web devuelve tablas grandes)
HEADERS = {"User-Agent": "Mozilla/5.0 (HIBLOOMS BIOMA-UNAV; actualizacion de datos de sonda)"}

# Cabecera de la tabla SAICA (sin tildes, en minúsculas) → columna de sensor_data
COLUMNAS = [
    ("fecha", "date_time"),
    ("ficocianina", "phycocyanin"),
    ("clorofila", "chlorophyll"),
    ("temperatura", "water_temp"),
    ("turbidez", "turbidity"),
    ("ph", "ph"),
    ("profundidad maxima embalse", "reservoir_max_depth_m"),
    ("profundidad", "depth"),
]


def sin_tildes(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip().lower()


def num(x):
    if x is None:
        return None
    s = str(x).strip().replace("\xa0", "")
    if "," in s:                      # 1.234,5 → 1234.5 ; 23,8 → 23.8
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def descargar(desde: date, hasta: date) -> pd.DataFrame:
    url = URL.format(e=ESTACION, a=desde.strftime("%d-%m-%Y"), b=hasta.strftime("%d-%m-%Y"))
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tabla in soup.find_all("table"):
        cab = [sin_tildes(th.get_text(" ", strip=True)) for th in tabla.find_all("th")]
        if not any("ficocianina" in c or "clorofila" in c for c in cab):
            continue
        # Texto crudo de cada celda: pandas.read_html confunde el punto decimal con el de
        # miles (23.8 → 238), así que los números se convierten a mano con num()
        filas = [[td.get_text(strip=True) for td in tr.find_all("td")]
                 for tr in (tabla.find("tbody") or tabla).find_all("tr")]
        filas = [f for f in filas if len(f) == len(cab)]
        df = pd.DataFrame(filas, columns=cab)
        out = pd.DataFrame(index=df.index)
        usadas = set()
        for clave, destino in COLUMNAS:
            col = next((c for c in df.columns if c.startswith(clave) and c not in usadas), None)
            if col is None or destino in out:
                continue
            usadas.add(col)
            out[destino] = df[col]
        # La profundidad máxima del embalse solo es válida en las celdas marcadas "f1"
        if "reservoir_max_depth_m" in out:
            idx = next((i for i, c in enumerate(cab) if c.startswith("profundidad maxima embalse")), None)
            if idx is not None and tabla.find("tbody"):
                vals = []
                for tr in tabla.find("tbody").find_all("tr"):
                    td = tr.find_all("td")
                    ok = len(td) > idx and (td[idx].get("class") or [None])[0] == "f1"
                    vals.append(num(td[idx].get_text(strip=True)) if ok else None)
                if len(vals) == len(out):
                    out["reservoir_max_depth_m"] = vals
        if "date_time" not in out:
            raise RuntimeError(f"La tabla de {url} no tiene columna de fecha: {list(df.columns)}")
        out["date_time"] = pd.to_datetime(out["date_time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        for c in out.columns:
            if c != "date_time":
                out[c] = out[c].map(num)
        out = out.dropna(subset=["date_time"])
        # La web da hora local peninsular
        out["date_time"] = out["date_time"].dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="shift_forward")
        return out.dropna(subset=["date_time"])
    return pd.DataFrame()


def control_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo criterio que los datos ya cargados: 0 exacto en temperatura o pH = fallo de sensor."""
    notas = pd.Series("", index=df.index)
    for c, nombre in (("water_temp", "temperatura"), ("ph", "pH")):
        if c in df:
            notas = notas.where(~(df[c] == 0), notas + f"{nombre} = 0 (fallo de sensor); ")
    if "phycocyanin" in df:
        notas = notas.where(~(df["phycocyanin"] < 0), notas + "ficocianina negativa; ")
    df["qc_flag"] = (notas != "").map({True: 2, False: 0})
    df["qc_note"] = notas.str.strip().str.rstrip(";").replace("", None)
    return df


def columnas_tabla(engine) -> list[str]:
    with engine.connect() as c:
        r = c.execute(text("""SELECT column_name FROM information_schema.columns
                              WHERE table_schema='public' AND table_name='sensor_data'"""))
        return [x[0] for x in r]


def ultimo_dato(engine) -> date | None:
    with engine.connect() as c:
        v = c.execute(text("SELECT MAX(date_time::timestamptz) FROM sensor_data WHERE source_code = :s"),
                      {"s": SOURCE}).scalar()
    return None if v is None else pd.Timestamp(v).tz_convert("Europe/Madrid").date()


def escribir(engine, df: pd.DataFrame, desde: date, hasta: date) -> int:
    cols = columnas_tabla(engine)
    df = df.assign(reservoir_id=RESERVOIR_ID, source_code=SOURCE)
    df = df[[c for c in df.columns if c in cols]]
    with engine.begin() as c:
        c.execute(text("""DELETE FROM sensor_data WHERE source_code = :s
                          AND date_time::timestamptz >= :a AND date_time::timestamptz < :b"""),
                  {"s": SOURCE, "a": f"{desde} 00:00:00 Europe/Madrid",
                   "b": f"{hasta + timedelta(days=1)} 00:00:00 Europe/Madrid"})
        df.to_sql("sensor_data", c, if_exists="append", index=False, method="multi", chunksize=1000)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", help="AAAA-MM-DD (por defecto, el día del último dato que hay en la base)")
    ap.add_argument("--hasta", help="AAAA-MM-DD (por defecto, ayer)")
    ap.add_argument("--prueba", action="store_true", help="descarga y muestra, sin escribir en la base")
    a = ap.parse_args()

    urls = [u.strip() for u in os.getenv("DATABASE_URL_WRITE", "").split(";") if u.strip()]
    if not urls and not a.prueba:
        sys.exit("Falta DATABASE_URL_WRITE (usuario con permiso de escritura, no hiblooms_ro).")
    engines = [create_engine(u.replace("postgres://", "postgresql://", 1), pool_pre_ping=True) for u in urls]

    hasta = date.fromisoformat(a.hasta) if a.hasta else date.today() - timedelta(days=1)
    for eng in engines or [None]:
        nombre = eng.url.host if eng else "(prueba)"
        if a.desde:
            desde = date.fromisoformat(a.desde)
        elif eng:
            u = ultimo_dato(eng)
            desde = u if u else date(2018, 1, 1)
        else:
            desde = hasta - timedelta(days=2)
        print(f"▶ {nombre}: {desde} → {hasta}")
        total, d = 0, desde
        while d <= hasta:
            f = min(d + timedelta(days=TRAMO_DIAS - 1), hasta)
            try:
                df = descargar(d, f)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {d} → {f}: {e}")
                d = f + timedelta(days=1)
                continue
            if df.empty:
                print(f"  · {d} → {f}: sin datos publicados")
            else:
                df = control_calidad(df)
                if eng is None:
                    print(df.head(10).to_string()); print(f"  ({len(df)} filas; columnas: {list(df.columns)})")
                else:
                    try:
                        n = escribir(eng, df, d, f)
                    except Exception as e:  # noqa: BLE001
                        msg = str(getattr(e, "orig", e)).splitlines()[0]
                        sys.exit(f"  ✗ {d} → {f}: no se pudo escribir: {msg}")
                    total += n
                    print(f"  ✓ {d} → {f}: {n} medidas ({int((df['qc_flag'] > 0).sum())} marcadas)")
            d = f + timedelta(days=1)
            time.sleep(1)  # sin prisas con el servidor de la CHE
        print(f"■ {nombre}: {total} medidas escritas")


if __name__ == "__main__":
    main()
