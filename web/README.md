# HIBLOOMS · Web viewer (developer notes)

React frontend + FastAPI backend that reuses `hiblooms_core.py`. See the root `README.md`
for an overview of the tool and how to deploy it.

```
web/
├── backend/
│   ├── server.py        # FastAPI app: reservoirs, search, images, series, monitor, historical context, data
│   ├── calibration.py   # Satellite ↔ in situ calibration (one GEE call per run)
│   ├── projectdb.py     # Read-only access to the project database (Data tab)
│   └── val_buoy.py      # El Val SAICA buoy: live data, seasonal risk and trend
├── frontend/            # Vite + React + TypeScript + MapLibre + Recharts
└── start.bat            # Windows: starts backend and frontend in two windows
```

## Run locally

**1. Backend** (from the repo root):

```bash
pip install -r web/backend/requirements.txt
export GEE_SERVICE_ACCOUNT_JSON=/path/to/service_account.json   # or the JSON as text
uvicorn web.backend.server:app --reload --port 8000
```

Without credentials it starts in **demo mode** (simulated data) so you can work on the interface:
`HIBLOOMS_MOCK=1 uvicorn web.backend.server:app --reload --port 8000`

**2. Frontend**:

```bash
cd web/frontend
npm install
npm run dev      # http://localhost:5173 (proxies /api → :8000)
```

## Notes

- The interface is bilingual (Spanish/English): UI strings live in `frontend/src/i18n.ts`.
- Searches collect the statistics of every image in the period with a single GEE call.
  Limit: 2 years per search.
- The time series recomputes the reservoir mean date by date (slow with GEE; max. 120 dates).
- Long operations (monitor, historical context, calibration) run as background jobs (`/api/jobs/{id}`).
- Base maps: Esri World Imagery and Esri World Light Gray Canvas (no API key).
- Production: the `Dockerfile` builds the frontend and serves it from the same service as the API.
