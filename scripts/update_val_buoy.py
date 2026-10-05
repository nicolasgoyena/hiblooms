"""
Update sensor_data with the SAICA 945 station of El Val reservoir (CHE).

Downloads from the latest record already in the database up to yesterday, so it can be
run at any time (daily, weekly or after months idle) and it catches up.
It is idempotent: before inserting a time span it deletes any data from this source on
those same dates, so re-running it does not duplicate data.

Usage:
    set DATABASE_URL_WRITE=postgresql://user:password@host/db
    python scripts/update_val_buoy.py                  # from the latest record up to yesterday
    python scripts/update_val_buoy.py --desde 2026-06-01
    python scripts/update_val_buoy.py --prueba         # download and show, without writing

Several databases at once (e.g. Neon and the local one): separate the URLs with ';' in DATABASE_URL_WRITE.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata

# The Windows console does not always support UTF-8: keep symbols from breaking the script
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

ESTACION = 945                    # SAICA station of El Val reservoir
RESERVOIR_ID = 2351               # El Val in reservoirs_spain
# source_code of this data in the database. NOTE: "che_aquadam_el_val" is the old profiling
# buoy (2018–2024), a different source; SAICA station 945 continues from 2024.
SOURCE = "che_saica_945"
URL = "https://saica.chebro.es/fichaDataTabla.php?estacion={e}&fini={a}&ffin={b}"
TRAMO_DIAS = 7                    # days per request (the website returns large tables)
HEADERS = {"User-Agent": "Mozilla/5.0 (HIBLOOMS BIOMA-UNAV; actualizacion de datos de sonda)"}

# SAICA table header (no accents, lowercase) → sensor_data column
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
        # Raw text of each cell: pandas.read_html confuses the decimal point with the thousands
        # separator (23.8 → 238), so numbers are converted by hand with num()
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
        # The reservoir maximum depth is only valid in cells flagged "f1"
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
        # The website gives mainland Spain local time
        out["date_time"] = out["date_time"].dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="shift_forward")
        return out.dropna(subset=["date_time"])
    return pd.DataFrame()


def control_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Same criterion as the data already loaded: exactly 0 in temperature or pH = sensor failure."""
    notas = pd.Series("", index=df.index)
    for c, nombre in (("water_temp", "temperatura"), ("ph", "pH")):
        if c in df:
            notas = notas.where(~(df[c] == 0), notas + f"{nombre} = 0 (fallo de sensor); ")
    if "phycocyanin" in df:
        notas = notas.where(~(df["phycocyanin"] < 0), notas + "ficocianina negativa; ")
    df["qc_flag"] = (notas != "").map({True: 2, False: 0})
    df["qc_note"] = notas.str.strip().str.rstrip(";").replace("", None)
    return df


# ── Neon over HTTPS (port 443) ─────────────────────────────────────────────
# The university network blocks PostgreSQL port 5432, and the CHE website does not
# accept connections from outside Spain (GitHub Actions). Neon supports SQL over HTTPS,
# so from UNAV we write to Neon over 443 when 5432 does not respond.

class NeonHTTP:
    def __init__(self, url: str):
        from urllib.parse import urlparse
        self.url = url.replace("postgres://", "postgresql://", 1)
        self.host = urlparse(self.url).hostname
        region = self.host.split(".", 1)[1]
        self.endpoints = [f"https://api.{region}/sql", f"https://{self.host}/sql"]

    def _post(self, body: dict) -> dict:
        err = None
        for ep in self.endpoints:
            try:
                r = requests.post(ep, json=body, timeout=120, headers={
                    "Neon-Connection-String": self.url, "Content-Type": "application/json"})
            except requests.RequestException as e:
                err = e
                continue
            if r.status_code == 404:
                err = RuntimeError(f"{ep}: 404")
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Neon HTTP {r.status_code}: {r.text[:300]}")
            self.endpoints = [ep]          # the one that works, for the next ones
            return r.json()
        raise RuntimeError(f"No se pudo usar Neon por HTTPS: {err}")

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        return self._post({"query": sql, "params": params or []}).get("rows", [])

    def transaction(self, queries: list[tuple[str, list]]) -> None:
        self._post({"queries": [{"query": q, "params": p} for q, p in queries]})


def puerto_abierto(host: str, port: int = 5432, espera: float = 6) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=espera):
            return True
    except OSError:
        return False


def _jsonval(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, pd.Timestamp):
        return str(v)          # same format as the rest: 2026-09-20 23:30:00+02:00
    if hasattr(v, "item"):               # numpy → Python
        return v.item()
    return v


def columnas_tabla(engine) -> list[str]:
    if isinstance(engine, NeonHTTP):
        return [r["column_name"] for r in engine.query(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='sensor_data'")]
    with engine.connect() as c:
        r = c.execute(text("""SELECT column_name FROM information_schema.columns
                              WHERE table_schema='public' AND table_name='sensor_data'"""))
        return [x[0] for x in r]


def ultimo_dato(engine) -> date | None:
    if isinstance(engine, NeonHTTP):
        rows = engine.query("SELECT MAX(date_time::timestamptz)::text AS m FROM sensor_data WHERE source_code = $1", [SOURCE])
        v = rows[0]["m"] if rows else None
        return None if v is None else pd.Timestamp(v).tz_convert("Europe/Madrid").date()
    with engine.connect() as c:
        v = c.execute(text("SELECT MAX(date_time::timestamptz) FROM sensor_data WHERE source_code = :s"),
                      {"s": SOURCE}).scalar()
    return None if v is None else pd.Timestamp(v).tz_convert("Europe/Madrid").date()


def escribir(engine, df: pd.DataFrame, desde: date, hasta: date) -> int:
    cols = columnas_tabla(engine)
    df = df.assign(reservoir_id=RESERVOIR_ID, source_code=SOURCE)
    df = df[[c for c in df.columns if c in cols]]
    a_ = f"{desde} 00:00:00 Europe/Madrid"
    b_ = f"{hasta + timedelta(days=1)} 00:00:00 Europe/Madrid"
    if isinstance(engine, NeonHTTP):
        qs = [("""DELETE FROM sensor_data WHERE source_code = $1
                  AND date_time::timestamptz >= $2::timestamptz AND date_time::timestamptz < $3::timestamptz""",
               [SOURCE, a_, b_])]
        names = ", ".join(f'"{c}"' for c in df.columns)
        k = len(df.columns)
        for i in range(0, len(df), 400):
            part = df.iloc[i:i + 400]
            vals = ", ".join("(" + ", ".join(f"${j * k + m + 1}" for m in range(k)) + ")" for j in range(len(part)))
            params = [_jsonval(v) for row in part.itertuples(index=False) for v in row]
            qs.append((f"INSERT INTO sensor_data ({names}) VALUES {vals}", params))
        engine.transaction(qs)     # all or nothing, as with the normal connection
        return len(df)
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
    engines = []
    for u in urls:
        eng = create_engine(u.replace("postgres://", "postgresql://", 1), pool_pre_ping=True)
        host = eng.url.host or ""
        if "neon.tech" in host and not puerto_abierto(host):
            print(f"· {host}: el puerto 5432 está bloqueado en esta red; se usa Neon por HTTPS")
            eng = NeonHTTP(u)
        engines.append(eng)

    hasta = date.fromisoformat(a.hasta) if a.hasta else date.today() - timedelta(days=1)
    for eng in engines or [None]:
        nombre = (eng.host + " (HTTPS)" if isinstance(eng, NeonHTTP) else eng.url.host) if eng else "(prueba)"
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
                if eng is None or a.prueba:
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
            time.sleep(1)  # go easy on the CHE server
        print(f"■ {nombre}: {total} medidas escritas" + (" (modo prueba: no se escribe nada)" if a.prueba else ""))


if __name__ == "__main__":
    main()
