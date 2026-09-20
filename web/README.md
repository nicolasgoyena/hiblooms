# HIBLOOMS · Visor web (React)

Frontend React + backend FastAPI ligero que reutiliza `hiblooms_core.py`.
Primera versión: búsqueda de imágenes Sentinel-2 por embalse, mapa del índice, serie temporal.

```
web/
├── backend/server.py   # FastAPI: /api/reservoirs, /api/search, /api/image, /api/timeseries
└── frontend/           # Vite + React + TypeScript + MapLibre + Recharts
```

## Arrancar en local

**1. Backend** (desde la raíz del repo `hiblooms/`):

```bash
pip install fastapi uvicorn geopandas earthengine-api
# Credenciales GEE: el mismo JSON de cuenta de servicio que usa Streamlit
export GEE_SERVICE_ACCOUNT_JSON=/ruta/a/service_account.json   # o el JSON como texto
uvicorn web.backend.server:app --reload --port 8000
```

Sin credenciales arranca en **modo demo** (datos simulados) para trabajar en la interfaz:
`HIBLOOMS_MOCK=1 uvicorn web.backend.server:app --reload --port 8000`

**2. Frontend**:

```bash
cd web/frontend
npm install
npm run dev      # http://localhost:5173 (redirige /api → :8000)
```

## Notas

- La búsqueda usa `get_available_dates` del core (evalúa nubosidad y cobertura imagen a imagen):
  con rangos largos tarda. Límite: 2 años por búsqueda.
- La serie temporal recalcula la media del embalse fecha a fecha (lento con GEE; máx. 120 fechas).
  Siguiente paso: pasarlo al worker asíncrono de `api/`.
- Mapas base: Esri World Imagery y CARTO (sin API key).
- Producción: `npm run build` → servir `frontend/dist` (Vercel/Netlify/Render static) y fijar
  `VITE_API_URL` con la URL del backend; `CORS_ORIGINS` en el backend.
