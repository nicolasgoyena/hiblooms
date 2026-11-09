# Conectar la pestaña «Datos» a la base de datos del proyecto

La web solo **lee** la base de datos. Se conecta con la variable de entorno
`DATABASE_URL`, y además abre todas las sesiones en modo solo lectura.
Sin `DATABASE_URL`, la pestaña muestra datos simulados con la misma estructura.

## 1. Probar en local contra tu base de datos

Antes de subir nada, comprueba que lee bien tus datos reales. En la terminal:

```
cd C:\Users\ngoyenaserv\hiblooms_web
pip install SQLAlchemy psycopg2-binary
set DATABASE_URL=postgresql://postgres:TU_CLAVE@localhost:5432/hiblooms
set GEE_SERVICE_ACCOUNT_JSON=C:\ruta\a\gee_key.json
python -m uvicorn web.backend.server:app --reload --port 8000
```

y en otra terminal `cd web\frontend` y `npm run dev`.

Abre la pestaña **Datos**. Si algo falla, `http://localhost:8000/api/db/status`
te dice qué ha pasado.

## 2. Subir la base de datos a Neon

1. Crea un proyecto en <https://neon.tech> (plan gratuito). Región: Europa (Frankfurt).
2. Crea la base `hiblooms` (Neon → *Databases* → *New database*) y, en su **SQL Editor**, activa PostGIS:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
3. En pgAdmin, clic derecho sobre `hiblooms` → **Backup…**
   - Format: **Custom**
   - En *Data Options*, desmarca **Owner** y **Privileges**.
4. Restaura en Neon desde la terminal (Neon te da la cadena de conexión en *Connection Details*):
   ```
   pg_restore --no-owner --no-privileges -d "postgresql://USUARIO:CLAVE@ep-xxxx.eu-central-1.aws.neon.tech/hiblooms?sslmode=require" C:\ruta\al\backup.backup
   ```
   Algún aviso sobre `spatial_ref_sys` o la extensión PostGIS es normal: ya existe en Neon.

Tamaño: el plan gratuito admite 0,5 GB. Compruébalo antes en pgAdmin con
`SELECT pg_size_pretty(pg_database_size('hiblooms'));`

## 3. Usuario de solo lectura para la web

En el SQL Editor de Neon (elige una contraseña larga):

Las tablas están repartidas en esquemas (`catalog`, `field`, `lab`, `sampling`,
`timeseries`), así que los permisos se dan en todos. El `search_path` de la base
sirve para las consultas a mano y para el script de la sonda; la web escribe
siempre el esquema y no depende de él.

```sql
-- solo la primera vez (el rol es de todo el proyecto de Neon)
CREATE ROLE hiblooms_ro LOGIN PASSWORD 'una-contraseña-larga-y-distinta';

ALTER DATABASE hiblooms SET search_path = catalog, field, lab, sampling, timeseries, public;
GRANT CONNECT ON DATABASE hiblooms TO hiblooms_ro;
GRANT USAGE ON SCHEMA catalog, field, lab, sampling, timeseries, public TO hiblooms_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA catalog, field, lab, sampling, timeseries, public TO hiblooms_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA catalog, field, lab, sampling, timeseries, public
  GRANT SELECT ON TABLES TO hiblooms_ro;
```

La web nunca usa tu usuario de administrador: si alguien encontrara un fallo,
como mucho podría leer.

## 4. Configurar Render

En Render → servicio **hiblooms** → **Environment**, añade:

```
DATABASE_URL = postgresql://hiblooms_ro:CONTRASEÑA@ep-xxxx.eu-central-1.aws.neon.tech/hiblooms?sslmode=require
```

Guarda: Render reinicia el servicio solo. La pestaña Datos dejará de decir
«datos simulados».

## 5. Cuando cargues datos nuevos

Los datos se leen una vez y se guardan en memoria 6 horas. Para verlos al
momento: `POST /api/db/reload`.

Cuando la base se mueva al servidor de la UNAV, solo hay que cambiar
`DATABASE_URL` en Render. El código no cambia.

## Textos en dos idiomas

Los catálogos guardan el texto en inglés (`name`, `label`, `description`…) y en
español en las columnas `*_es`. La web envía los dos y muestra uno u otro según
el idioma elegido (función `loc()` de `web/frontend/src/i18n.ts`).
