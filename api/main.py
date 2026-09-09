# api/main.py
#
# API de jobs asíncronos para HIBLOOMS.
# Arrancar con:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000
#
# Variables de entorno:
#   GEE_SERVICE_ACCOUNT_JSON  JSON de la cuenta de servicio de Google Earth Engine
#                             (el mismo que en secrets.toml)
#   DATABASE_URL              Cadena de conexión a PostgreSQL donde se persisten
#                             los jobs (ver api/jobs_store.py)
#   HIBLOOMS_API_TOKEN        Token compartido con la app Streamlit. Si no está
#                             definido, la API arranca abierta y lo avisa en el log.
#   APP_ORIGIN                Origen permitido por CORS, p.ej.
#                             https://hiblooms-app.onrender.com  (por defecto: "*")

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import jobs_store

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
_APP_ORIGIN = os.environ.get("APP_ORIGIN", "*")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Crea la tabla de jobs y limpia los caducados al arrancar."""
    jobs_store.init_db()
    try:
        borrados = jobs_store.cleanup()
        if borrados:
            log.info("Limpieza inicial: %d jobs caducados eliminados", borrados)
    except Exception as e:  # la limpieza nunca debe impedir el arranque
        log.warning("No se pudo ejecutar la limpieza de jobs: %s", e)

    if not os.environ.get("HIBLOOMS_API_TOKEN"):
        log.warning(
            "HIBLOOMS_API_TOKEN no está definido: la API acepta peticiones "
            "sin autenticar. Defínelo en producción."
        )
    if _APP_ORIGIN == "*":
        log.warning(
            "APP_ORIGIN no está definido: CORS acepta cualquier origen. "
            "Defínelo con el dominio de la app en producción."
        )

    yield


app = FastAPI(title="HIBLOOMS Jobs API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _APP_ORIGIN.split(",") if o.strip()],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Autenticación
# ---------------------------------------------------------------------------
def require_token(x_api_token: str | None = Header(default=None)) -> None:
    """
    Valida la cabecera X-API-Token contra HIBLOOMS_API_TOKEN.

    Sin esto, cualquiera que conozca la URL pública de la API puede lanzar
    trabajos de Earth Engine contra la cuenta de servicio del proyecto.
    Si la variable no está definida, no se exige token (modo desarrollo).
    """
    import hmac

    esperado = os.environ.get("HIBLOOMS_API_TOKEN")
    if not esperado:
        return
    if not x_api_token or not hmac.compare_digest(x_api_token, esperado):
        raise HTTPException(status_code=401, detail="Token de API inválido o ausente")


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class JobSubmitRequest(BaseModel):
    workflow: str                  # "visualization" | "calibration"
    model_config = {"extra": "allow"}

    def full_config(self) -> Dict[str, Any]:
        return self.model_dump()


class JobProgressUpdate(BaseModel):
    step:     str
    progress: int   # 0-100


# ---------------------------------------------------------------------------
# Callbacks que reciben los workers
# ---------------------------------------------------------------------------
def _update_progress(job_id: str, step: str, progress: int) -> None:
    jobs_store.update_progress(job_id, step, progress)


def _complete(job_id: str, results: Dict[str, Any]) -> None:
    jobs_store.complete(job_id, results)


def _fail(job_id: str, error: str) -> None:
    jobs_store.fail(job_id, error)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Comprobación de vida, incluida la conexión a la base de datos."""
    try:
        jobs_store.get("00000000-0000-0000-0000-000000000000")
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Base de datos no disponible: {e}")


@app.post("/jobs/submit", status_code=202, dependencies=[Depends(require_token)])
async def submit_job(body: JobSubmitRequest, background_tasks: BackgroundTasks):
    """
    Recibe la configuración desde la app Streamlit, persiste el job
    y lo lanza en background. Devuelve el job_id inmediatamente.
    """
    workflow = body.workflow
    if workflow not in ("visualization", "calibration"):
        raise HTTPException(status_code=400, detail=f"Unknown workflow: {workflow}")

    config = body.full_config()
    try:
        job_id = jobs_store.create(workflow, config)
    except Exception as e:
        log.exception("No se pudo registrar el job")
        raise HTTPException(status_code=503, detail=f"No se pudo registrar el job: {e}")

    log.info("Job created: %s (%s)", job_id, workflow)

    if workflow == "visualization":
        from api.worker import run_visualization_job
        background_tasks.add_task(run_visualization_job, job_id, config, _update_progress, _complete, _fail)
    else:
        from api.worker import run_calibration_job
        background_tasks.add_task(run_calibration_job, job_id, config, _update_progress, _complete, _fail)

    return {"job_id": job_id, "state": "pending"}


@app.get("/jobs/{job_id}/status", dependencies=[Depends(require_token)])
async def get_job_status(job_id: str):
    """
    Devuelve el estado actual del job.
    Llamado por la app cada 5 s mediante st.fragment.
    """
    job = jobs_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response: Dict[str, Any] = {
        "job_id":   job["job_id"],
        "workflow": job["workflow"],
        "state":    job["state"],
        "progress": job["progress"],
        "step":     job["step"],
        "error":    job["error"],
    }

    if job["state"] == "done":
        response["results"] = job["results"]

    return response


@app.patch("/jobs/{job_id}", dependencies=[Depends(require_token)])
async def patch_job_progress(job_id: str, body: JobProgressUpdate):
    """
    Permite actualizar el progreso desde un proceso externo
    (útil si el worker corre en un contenedor separado en LifeWatch/NaaVRE).
    """
    if jobs_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    jobs_store.update_progress(job_id, body.step, body.progress)
    return {"job_id": job_id, "step": body.step, "progress": body.progress}
