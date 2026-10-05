# Connecting the "Data" tab to the project database

The web tool only **reads** the database. It connects through the `DATABASE_URL`
environment variable, and on top of that it opens every session in read-only mode.
Without `DATABASE_URL`, the tab shows simulated data with the same structure.

## 1. Test locally against your database

Before deploying anything, check that it reads your real data correctly. In a terminal:

```
cd C:\Users\ngoyenaserv\hiblooms_web
pip install -r web/backend/requirements.txt
set DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/unav_water_sampling
set GEE_SERVICE_ACCOUNT_JSON=C:\path\to\gee_key.json
python -m uvicorn web.backend.server:app --reload --port 8000
```

and in another terminal `cd web\frontend` and `npm run dev`.

Open the **Data** tab. If something fails, `http://localhost:8000/api/db/status`
tells you what happened.

## 2. Upload the database to Neon

1. Create a project at <https://neon.tech> (free plan). Region: Europe (Frankfurt).
2. In its **SQL Editor**, enable PostGIS:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
3. In pgAdmin, right-click `unav_water_sampling` → **Backup…**
   - Format: **Custom**
   - Under *Data Options*, untick **Owner** and **Privileges**.
4. Restore into Neon from a terminal (Neon gives you the connection string under *Connection Details*):
   ```
   pg_restore --no-owner --no-privileges -d "postgresql://USER:PASSWORD@ep-xxxx.eu-central-1.aws.neon.tech/neondb?sslmode=require" C:\path\to\backup.backup
   ```
   Some warnings about `spatial_ref_sys` or the PostGIS extension are normal: it already exists in Neon.

Size: the free plan allows 0.5 GB. Check it first in pgAdmin with
`SELECT pg_size_pretty(pg_database_size('unav_water_sampling'));`

## 3. Read-only user for the web tool

In Neon's SQL Editor (choose a long password):

```sql
CREATE ROLE hiblooms_ro LOGIN PASSWORD 'a-long-and-different-password';
GRANT CONNECT ON DATABASE neondb TO hiblooms_ro;
GRANT USAGE ON SCHEMA public TO hiblooms_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO hiblooms_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO hiblooms_ro;
```

The web tool never uses your admin user: if someone found a flaw,
at most they could read.

## 4. Configure Render

In Render → **hiblooms** service → **Environment**, add:

```
DATABASE_URL = postgresql://hiblooms_ro:PASSWORD@ep-xxxx.eu-central-1.aws.neon.tech/neondb?sslmode=require
```

Save: Render restarts the service by itself. The Data tab will stop saying
"simulated data".

## 5. When you load new data

Data is read once and kept in memory for 10 minutes. To see it right away:
`POST /api/db/reload` (or wait 10 minutes).

When the database moves to the UNAV server, only `DATABASE_URL` needs to change
in Render. The code stays the same.
