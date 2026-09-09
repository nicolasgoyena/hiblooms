"""
Tests del store de jobs en PostgreSQL.

Requieren una base de datos de pruebas. Indícala con la variable de entorno
TEST_DATABASE_URL, por ejemplo:

    export TEST_DATABASE_URL="postgresql://postgres:test@localhost:5432/hiblooms_test"
    pytest tests/

Si no está definida, los tests se saltan (no fallan).
"""

import os

import pytest
from sqlalchemy import text

TEST_DB = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="Define TEST_DATABASE_URL para ejecutar los tests del store"
)


@pytest.fixture(scope="module")
def store():
    os.environ["DATABASE_URL"] = TEST_DB
    from api import jobs_store

    jobs_store.init_db()
    yield jobs_store


def test_init_db_es_idempotente(store):
    store.init_db()
    store.init_db()


def test_ciclo_de_vida_completo(store):
    jid = store.create("visualization", {"reservoir": "El Val", "indices": ["MCI"]})

    job = store.get(jid)
    assert job["state"] == "pending"
    assert job["progress"] == 0

    store.update_progress(jid, "Descargando imágenes", 42)
    job = store.get(jid)
    assert job["state"] == "running"
    assert job["progress"] == 42
    assert job["step"] == "Descargando imágenes"

    resultados = {"data_time": [{"Point": "Media_Embalse", "MCI": 0.31}]}
    store.complete(jid, resultados)
    job = store.get(jid)
    assert job["state"] == "done"
    assert job["progress"] == 100
    # El JSON anidado debe volver intacto, no como texto
    assert job["results"]["data_time"][0]["MCI"] == 0.31


def test_progreso_se_limita_al_rango(store):
    jid = store.create("visualization", {})
    store.update_progress(jid, "fuera de rango", 250)
    assert store.get(jid)["progress"] == 100
    store.update_progress(jid, "negativo", -10)
    assert store.get(jid)["progress"] == 0


def test_job_fallido_guarda_el_error(store):
    jid = store.create("calibration", {})
    store.fail(jid, "GEE devolvió un error de cuota")
    job = store.get(jid)
    assert job["state"] == "error"
    assert "cuota" in job["error"]


def test_job_inexistente_devuelve_none(store):
    assert store.get("00000000-0000-0000-0000-000000000000") is None


def test_job_colgado_se_reporta_como_error(store):
    """Si el worker muere, el job no puede quedarse en 'running' para siempre."""
    jid = store.create("visualization", {})
    store.update_progress(jid, "procesando", 10)
    with store.get_engine().begin() as con:
        con.execute(
            text(
                f"UPDATE {store.TABLE} SET updated_at = now() - interval '3 hours' "
                "WHERE job_id = :j"
            ),
            {"j": jid},
        )
    job = store.get(jid)
    assert job["state"] == "error"
    assert "interrumpió" in job["error"]


def test_cleanup_borra_solo_los_caducados(store):
    viejo = store.create("visualization", {})
    store.complete(viejo, {})
    reciente = store.create("visualization", {})
    store.complete(reciente, {})

    with store.get_engine().begin() as con:
        con.execute(
            text(
                f"UPDATE {store.TABLE} SET updated_at = now() - interval '5 days' "
                "WHERE job_id = :j"
            ),
            {"j": viejo},
        )

    assert store.cleanup() >= 1
    assert store.get(viejo) is None
    assert store.get(reciente) is not None
