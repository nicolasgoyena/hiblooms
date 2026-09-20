"""Actualiza data/boletin_embalses.csv con el último dato del Boletín Hidrológico Semanal (MITECO).

Descarga la base histórica oficial (Access, ~220 MB), se queda con la última semana de los
embalses HIBLOOMS y reescribe el CSV que lee el backend. También refresca el histórico.

Uso:   python scripts/actualizar_boletin.py
Necesita: pip install pandas requests  y  mdbtools (Linux) o pyodbc/Access (Windows).
En Windows sin mdbtools, abre el .mdb con Access y exporta la tabla a CSV, luego:
       python scripts/actualizar_boletin.py --csv ruta\\tabla_exportada.csv
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

URL = ("https://www.miteco.gob.es/content/dam/miteco/es/agua/temas/evaluacion-de-los-recursos-hidricos/"
       "boletin-hidrologico/Historico-de-embalses/BD-Embalses.zip")
TABLE = "T_Datos Embalses 1988-2026"
ROOT = Path(__file__).resolve().parents[1]
MAP_CSV = ROOT / "data" / "boletin_embalses.csv"
HIST_CSV = ROOT / "data" / "boletin_historico_hiblooms.csv"


def descargar_mdb(dest: Path) -> Path:
    import requests
    print("Descargando el boletín histórico (~220 MB)…")
    r = requests.get(URL, timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = [n for n in z.namelist() if n.lower().endswith((".mdb", ".accdb"))][0]
        z.extract(name, dest)
    return dest / name


def leer_tabla(mdb: Path) -> pd.DataFrame:
    try:
        out = subprocess.run(["mdb-export", str(mdb), TABLE], capture_output=True, check=True)
        return pd.read_csv(io.BytesIO(out.stdout), dtype=str)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("No encuentro mdbtools. Exporta la tabla a CSV desde Access y pásala con --csv.")


def limpiar(d: pd.DataFrame) -> pd.DataFrame:
    d["FECHA"] = pd.to_datetime(d["FECHA"], format="%m/%d/%y %H:%M:%S", errors="coerce")
    for c in ("AGUA_TOTAL", "AGUA_ACTUAL"):
        d[c] = (d[c].astype(str).str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False).astype(float))
    return d.dropna(subset=["FECHA"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Tabla ya exportada a CSV (salta la descarga)")
    a = ap.parse_args()

    if a.csv:
        d = limpiar(pd.read_csv(a.csv, dtype=str))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            d = limpiar(leer_tabla(descargar_mdb(Path(tmp))))

    m = pd.read_csv(MAP_CSV, dtype={"boletin_nombre": str}).fillna({"boletin_nombre": ""})
    last = d.sort_values("FECHA").groupby("EMBALSE_NOMBRE").tail(1).set_index("EMBALSE_NOMBRE")

    filas = []
    for r in m.itertuples():
        bn = (r.boletin_nombre or "").strip()
        if bn and bn in last.index:
            c = last.loc[bn]
            pct = round(100 * c.AGUA_ACTUAL / c.AGUA_TOTAL, 1) if c.AGUA_TOTAL else None
            filas.append(dict(hiblooms_id=r.hiblooms_id, boletin_nombre=bn, ambito=c.AMBITO_NOMBRE,
                              capacidad_hm3=c.AGUA_TOTAL, volumen_hm3=c.AGUA_ACTUAL,
                              fecha=c.FECHA.date().isoformat(), pct=pct))
        else:
            filas.append(dict(hiblooms_id=r.hiblooms_id, boletin_nombre=bn, ambito="",
                              capacidad_hm3=None, volumen_hm3=None, fecha="", pct=None))
    pd.DataFrame(filas).to_csv(MAP_CSV, index=False)

    nombres = {r.boletin_nombre: r.hiblooms_id for r in m.itertuples() if (r.boletin_nombre or "").strip()}
    sel = d[d.EMBALSE_NOMBRE.isin(nombres)].copy()
    sel["hiblooms_id"] = sel.EMBALSE_NOMBRE.map(nombres)
    sel["pct"] = 100 * sel.AGUA_ACTUAL / sel.AGUA_TOTAL
    (sel[["hiblooms_id", "EMBALSE_NOMBRE", "FECHA", "AGUA_TOTAL", "AGUA_ACTUAL", "pct"]]
     .rename(columns={"EMBALSE_NOMBRE": "boletin_nombre", "FECHA": "fecha",
                      "AGUA_TOTAL": "capacidad_hm3", "AGUA_ACTUAL": "volumen_hm3"})
     .to_csv(HIST_CSV, index=False))
    print(f"Listo · {MAP_CSV.name} y {HIST_CSV.name} actualizados "
          f"(última fecha del boletín: {d.FECHA.max().date()})")


if __name__ == "__main__":
    main()
