"""
Tests de la API de jobs: autenticación, endpoints y persistencia.

Requieren TEST_DATABASE_URL (ver tests/test_jobs_store.py). El worker se
sustituye por un doble para no lanzar trabajos reales de Earth Engine.
"""

import os
import sys
import types

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
TOKEN = "token-de-prueba"

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="Define TEST_DATABASE_URL para ejecutar los tests de la API"
)


@pytest.fixture(scope="module")
def client():
    os.environ["DATABASE_URL"] = TEST_DB
    os.environ["HIBLOOMS_API_TOKEN"] = TOKEN
    os.environ["APP_ORIGIN"] = "https://hiblooms-app.onrender.com"

    # Doble del worker: importa ee/geopandas y ejecutaría jobs reales de GEE.
    doble = types.ModuleType("api.worker")
    doble.run_visualization_job = lambda *a, **k: None
    doble.run_calibration_job = lambda *a, **k: None
    sys.modules["api.worker"] = doble

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth():
    return {"X-API-Token": TOKEN}


def test_sin_token_rechaza(client):
    assert client.post("/jobs/submit", json={"workflow": "visualization"}).status_code == 401


def test_token_incorrecto_rechaza(client):
    r = client.post(
        "/jobs/submit",
        json={"workflow": "visualization"},
        headers={"X-API-Token": "incorrecto"},
    )
    assert r.status_code == 401


def test_envio_valido_devuelve_job_id(client, auth):
    r = client.post(
        "/jobs/submit",
        json={"workflow": "visualization", "reservoir": "RIBARROJA"},
        headers=auth,
    )
    assert r.status_code == 202
    assert r.json()["job_id"]


def test_workflow_desconocido_es_400(client, auth):
    r = client.post("/jobs/submit", json={"workflow": "inventado"}, headers=auth)
    assert r.status_code == 400


def test_job_inexistente_es_404(client, auth):
    r = client.get("/jobs/00000000-0000-0000-0000-000000000000/status", headers=auth)
    assert r.status_code == 404


def test_estado_requiere_token(client, auth):
    jid = client.post(
        "/jobs/submit", json={"workflow": "visualization"}, headers=auth
    ).json()["job_id"]
    assert client.get(f"/jobs/{jid}/status").status_code == 401
    assert client.get(f"/jobs/{jid}/status", headers=auth).status_code == 200


def test_patch_actualiza_progreso(client, auth):
    """Endpoint usado por procesos externos (LifeWatch/NaaVRE)."""
    jid = client.post(
        "/jobs/submit", json={"workflow": "visualization"}, headers=auth
    ).json()["job_id"]

    r = client.patch(f"/jobs/{jid}", json={"step": "paso externo", "progress": 55}, headers=auth)
    assert r.status_code == 200
    assert client.get(f"/jobs/{jid}/status", headers=auth).json()["progress"] == 55


def test_el_job_sobrevive_a_un_reinicio(client, auth):
    """Esta es la razón de ser del cambio: antes el job vivía en memoria."""
    import importlib

    jid = client.post(
        "/jobs/submit", json={"workflow": "visualization"}, headers=auth
    ).json()["job_id"]
    client.patch(f"/jobs/{jid}", json={"step": "a medias", "progress": 40}, headers=auth)

    from api import jobs_store

    jobs_store._engine = None      # simula un proceso completamente nuevo
    importlib.reload(jobs_store)

    r = client.get(f"/jobs/{jid}/status", headers=auth)
    assert r.status_code == 200
    assert r.json()["progress"] == 40


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
