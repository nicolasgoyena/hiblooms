# Deploying the HIBLOOMS viewer

A single container serves both the web app and the API, so there is only one public link.
It runs anywhere that runs containers; these are the two likely options.

## Render (free plan, no card)

1. Push the repository to GitHub.
2. Go to <https://render.com>, create an account and click **New → Blueprint**.
3. Pick the repository: Render reads `render.yaml` and creates the service.
4. Under **Environment**, fill in:
   - `GEE_SERVICE_ACCOUNT_JSON`: the full JSON of the Earth Engine service account.
   - `HIBLOOMS_USERS`: `user:password,other:other`.
5. Wait for the build (a few minutes the first time). The link will be
   `https://hiblooms.onrender.com` or similar.

The free plan puts the service to sleep after 15 minutes without visits and it takes about
a minute to wake up. If that is a problem, the cheapest paid plan removes it.

## Own server or external infrastructure (UNAV, LifeWatch)

```bash
docker build -t hiblooms .
docker run -p 8000:7860 \
  -e GEE_SERVICE_ACCOUNT_JSON="$(cat gee_key.json)" \
  -e HIBLOOMS_USERS="nico:password" \
  -e HIBLOOMS_SECRET="a long random string" \
  -v hiblooms-data:/data \
  hiblooms
```

The `/data` volume keeps uploaded shapefiles, calibrations and caches across restarts.
Without it, that data is lost on restart.

## Environment variables

| Variable | Purpose |
|---|---|
| `GEE_SERVICE_ACCOUNT_JSON` | Earth Engine credentials (JSON or path to a file). Without it, demo mode. |
| `DATABASE_URL` | Project database, read-only user (see `DATABASE.md`). Without it, the Data tab shows simulated data. |
| `HIBLOOMS_USERS` | Login users. If missing, the site is open. |
| `HIBLOOMS_SECRET` | Session signing key. |
| `HIBLOOMS_DATA_DIR` | Instance data folder (defaults to `web/backend/_uploads`). |
| `PORT` | Listening port (7860 by default). |
| `CORS_ORIGINS` | Allowed origins if the web app is served from another domain. |

## API for workflows

Automatic documentation at `/docs`. Long operations
(`/api/monitor`, `/api/historical-context`, `/api/calibrate`) return a `job_id` that
is polled at `/api/jobs/{id}`. Adding `?wait=1` makes them answer directly, which
is convenient when calling them from a workflow.
