# HIBLOOMS — Detección y seguimiento de floraciones de cianobacterias en embalses

Repositorio de la herramienta digital del proyecto **HIBLOOMS** (Instituto BIOMA, Universidad de Navarra).
Combina imágenes **Sentinel-2** (Google Earth Engine) con datos de campo, laboratorio y sondas para
seguir la calidad del agua en embalses.

Versión publicada: <https://hiblooms.onrender.com>

## Qué hace

- **Proyecto**: información del proyecto, equipo y financiación.
- **Seguimiento actual**: estado de todos los embalses (boletín hidrológico, sonda de El Val en tiempo casi real).
- **Visor satelital**: búsqueda de imágenes Sentinel-2 por embalse, mapas de índices, serie temporal,
  distribución por clases y contexto histórico. Admite shapefiles propios (ZIP) para cualquier embalse.
- **Datos del proyecto**: muestreos de campo, laboratorio, sondas y campañas (base PostgreSQL/PostGIS).
- **Modelos y calibraciones**: fichas de los modelos validados y calibración con datos in situ del usuario.

## Modelos

Los modelos con coeficientes ajustados viven en `data/modelos/*.json` (fórmula, coeficientes, validación y
límites), no en el código: recalibrar un modelo no exige tocar Python.

| Índice | Estado | Embalse |
|---|---|---|
| `Chla_Val_cal` — clorofila-a | Validado (sonda CHE, 2018–2024, validación dejando fuera un año) | El Val |
| `PC_Bellus_cal`, `Chla_Bellus_cal`, `SDD_Bellus_UV`, `TSM_Bellus_UV` | Modelos de la Universitat de València | Bellús |
| `PC_Val_cal`, `UV_PC_Gral_cal`, `PCI_B5/B4` | Experimentales (Sentinel-2 no estima bien la ficocianina) | — |
| `NDCI_ind`, `MCI` | Índices espectrales | Cualquiera |

Los análisis que justifican estas decisiones están en `analisis/` (ver más abajo).

## Estructura

```
hiblooms/
├── hiblooms_core.py          # Núcleo GEE: embalses, nubosidad, índices y modelos sobre Sentinel-2
├── shapefiles/               # Embalses del proyecto (embalses_hiblooms.*)
├── data/
│   ├── modelos/              # Fichas y coeficientes de los modelos (p. ej. chla_val.json)
│   ├── puntos_interes.csv    # Puntos de muestreo por embalse
│   └── boletin_*.csv         # Boletín hidrológico (lo genera scripts/actualizar_boletin.py)
├── web/
│   ├── backend/              # API FastAPI (server.py, calibration.py, projectdb.py, sonda_val.py)
│   ├── frontend/             # React + TypeScript + Vite + MapLibre
│   └── arrancar.bat          # Arranque local en Windows (backend + frontend)
├── scripts/                  # Actualización de datos
│   ├── actualizar_boletin.py
│   └── actualizar_sonda_val.py
├── analisis/                 # Validaciones y ajuste de modelos (no se despliegan)
│   ├── validacion_sonda.py, validacion_muestreos.py
│   ├── exprimir_clorofila.py, mejorar_clorofila.py, ajustar_clorofila_final.py
│   ├── modelo_riesgo.py
│   └── resultados/           # Tablas, figuras e informes de cada análisis
├── Dockerfile                # Un solo contenedor: compila el frontend y sirve web + API
├── render.yaml               # Despliegue en Render
├── DESPLIEGUE.md             # Cómo desplegar (Render, servidor propio, LifeWatch)
└── BASE_DE_DATOS.md          # Conexión de la pestaña Datos a la base del proyecto
```

## Arrancar en local

```bash
# Backend (desde la raíz del repo)
pip install -r web/backend/requirements.txt
export GEE_SERVICE_ACCOUNT_JSON=/ruta/a/service_account.json   # o el JSON como texto
export DATABASE_URL=postgresql://usuario:clave@host:5432/base   # opcional (pestaña Datos)
uvicorn web.backend.server:app --reload --port 8000

# Frontend
cd web/frontend
npm install
npm run dev      # http://localhost:5173
```

Sin credenciales de GEE el backend arranca en **modo demo** con datos simulados
(`HIBLOOMS_MOCK=1` lo fuerza). En Windows, `web\arrancar.bat` lanza las dos partes.

## Desplegar

```bash
docker build -t hiblooms .
docker run -p 8000:7860 -e GEE_SERVICE_ACCOUNT_JSON="$(cat gee_key.json)" hiblooms
```

Variables de entorno: `GEE_SERVICE_ACCOUNT_JSON`, `DATABASE_URL` (solo lectura), `HIBLOOMS_USERS`
(`usuario:contraseña,...`), `HIBLOOMS_SECRET`, `HIBLOOMS_DATA_DIR`. Detalle en `DESPLIEGUE.md`.

## Componente en LifeWatch ERIC

El workflow de HIBLOOMS en el VRE de LifeWatch (componentes *HiBloomsCalibration* y
*HiBloomsVisualization*) se mantiene en el GitLab de LifeWatch y toma como referencia
`hiblooms_core.py`, `web/backend/calibration.py` y `data/modelos/` de este repositorio.

## Licencia y cita

HIBLOOMS Project — Detection and Visualization of Cyanobacterial Blooms in Reservoirs.
Universidad de Navarra.
