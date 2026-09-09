# api/jobs_store.py
#
# Persistencia de los jobs de HIBLOOMS en PostgreSQL.
#
# Sustituye al antiguo dict en memoria de api/main.py, que perdía todos los
# jobs en cada reinicio del servicio (habitual en Render: cold starts,
# redespliegues) y que no funcionaba con más de una instancia de la API.
#
# Requiere la variable de entorno:
#   DATABASE_URL  →  postgresql+psycopg2://usuario:password@host:puerto/dbname
#
# La tabla se crea sola la primera vez que arranca la API.

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

TABLE = "hiblooms_jobs"

# Un job terminado (done/error) se conserva este tiempo y luego se borra.
# Evita que la tabla crezca sin límite, que era el otro problema del dict.
JOB_TTL = timedelta(days=2)

# Un job que lleva demasiado tiempo "running" sin actualizarse se considera
# muerto: casi siempre significa que el worker que lo procesaba se cayó.
STALE_AFTER = timedelta(hours=1)

_engine: Optional[Engine] = None


# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------
def get_engine() -> Engine:
    """
    Engine de SQLAlchemy hacia PostgreSQL, creado una sola vez por proceso.

    A diferencia de db_utils.get_engine(), lee de una variable de entorno y no
    de st.secrets: la API no es un proceso de Streamlit y no tiene acceso a él.
    """
    global _engine
    if _engine is None:
        uri = os.environ.get("DATABASE_URL")
        if not uri:
            raise RuntimeError(
                "Variable de entorno DATABASE_URL no definida. Debe contener la "
                "cadena de conexión a PostgreSQL, por ejemplo: "
                "postgresql+psycopg2://usuario:password@host:5432/dbname"
            )
        # Render entrega la URL como 'postgres://', que SQLAlchemy ya no acepta.
        if uri.startswith("postgres://"):
            uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)
        elif uri.startswith("postgresql://"):
            uri = uri.replace("postgresql://", "postgresql+psycopg2://", 1)

        _engine = create_engine(uri, pool_pre_ping=True, pool_recycle=300)
    return _engine


def init_db() -> None:
    """Crea la tabla de jobs si no existe. Idempotente."""
    ddl = text(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            job_id      UUID PRIMARY KEY,
            workflow    TEXT        NOT NULL,
            state       TEXT        NOT NULL DEFAULT 'pending',
            progress    INTEGER     NOT NULL DEFAULT 0,
            step        TEXT        NOT NULL DEFAULT '',
            error       TEXT        NOT NULL DEFAULT '',
            config      JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
            results     JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    idx = text(
        f"CREATE INDEX IF NOT EXISTS {TABLE}_state_updated_idx "
        f"ON {TABLE} (state, updated_at)"
    )
    with get_engine().begin() as con:
        con.execute(ddl)
        con.execute(idx)
    log.info("Tabla de jobs lista (%s)", TABLE)


# ---------------------------------------------------------------------------
# Operaciones
# ---------------------------------------------------------------------------
def create(workflow: str, config: Dict[str, Any]) -> str:
    """Registra un job nuevo en estado 'pending' y devuelve su job_id."""
    import json

    job_id = str(uuid.uuid4())
    q = text(f"""
        INSERT INTO {TABLE} (job_id, workflow, state, config)
        VALUES (:job_id, :workflow, 'pending', CAST(:config AS jsonb))
    """)
    with get_engine().begin() as con:
        con.execute(q, {"job_id": job_id, "workflow": workflow, "config": json.dumps(config)})
    return job_id


def update_progress(job_id: str, step: str, progress: int) -> None:
    """Marca el job como 'running' y actualiza paso y porcentaje."""
    progress = max(0, min(100, int(progress)))
    q = text(f"""
        UPDATE {TABLE}
        SET state = 'running', step = :step, progress = :progress, updated_at = now()
        WHERE job_id = :job_id
    """)
    with get_engine().begin() as con:
        con.execute(q, {"job_id": job_id, "step": step, "progress": progress})


def complete(job_id: str, results: Dict[str, Any]) -> None:
    """Marca el job como terminado y guarda los resultados."""
    import json

    q = text(f"""
        UPDATE {TABLE}
        SET state = 'done', progress = 100, step = 'Completed',
            results = CAST(:results AS jsonb), updated_at = now()
        WHERE job_id = :job_id
    """)
    with get_engine().begin() as con:
        con.execute(q, {"job_id": job_id, "results": json.dumps(results, default=str)})


def fail(job_id: str, error: str) -> None:
    """Marca el job como fallido con su mensaje de error."""
    q = text(f"""
        UPDATE {TABLE}
        SET state = 'error', error = :error, updated_at = now()
        WHERE job_id = :job_id
    """)
    with get_engine().begin() as con:
        con.execute(q, {"job_id": job_id, "error": str(error)[:4000]})


def get(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Devuelve el job completo, o None si no existe.

    Un job 'running' que lleva más de STALE_AFTER sin actualizarse se devuelve
    como 'error': el worker que lo procesaba ya no existe, y sin esto la app se
    quedaría esperando indefinidamente.
    """
    q = text(f"""
        SELECT job_id, workflow, state, progress, step, error, results, updated_at
        FROM {TABLE}
        WHERE job_id = :job_id
    """)
    with get_engine().connect() as con:
        row = con.execute(q, {"job_id": job_id}).mappings().first()

    if row is None:
        return None

    job = dict(row)
    job["job_id"] = str(job["job_id"])

    if job["state"] in ("pending", "running"):
        updated = job["updated_at"]
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - updated > STALE_AFTER:
            job["state"] = "error"
            job["error"] = (
                "El proceso que ejecutaba este cálculo se interrumpió "
                "(probablemente un reinicio del servidor). Vuelve a lanzarlo."
            )

    job.pop("updated_at", None)
    return job


def cleanup() -> int:
    """Borra los jobs terminados más antiguos que JOB_TTL. Devuelve cuántos."""
    q = text(f"""
        DELETE FROM {TABLE}
        WHERE state IN ('done', 'error') AND updated_at < :limite
    """)
    with get_engine().begin() as con:
        res = con.execute(q, {"limite": datetime.now(timezone.utc) - JOB_TTL})
    return res.rowcount or 0
