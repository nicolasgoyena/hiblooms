# Desplegar el visor HIBLOOMS

Un solo contenedor sirve la web y la API, así que solo hay un enlace público.
Funciona en cualquier sitio que ejecute contenedores; aquí van las dos vías probables.

## Render (plan gratuito, sin tarjeta)

1. Sube el repositorio a GitHub.
2. Entra en <https://render.com>, crea la cuenta y pulsa **New → Blueprint**.
3. Elige el repositorio: Render leerá `render.yaml` y creará el servicio.
4. En **Environment**, rellena:
   - `GEE_SERVICE_ACCOUNT_JSON`: el JSON completo de la cuenta de servicio de Earth Engine.
   - `HIBLOOMS_USERS`: `usuario:contraseña,otro:otra`.
5. Espera a que construya (unos minutos la primera vez). El enlace será
   `https://hiblooms.onrender.com` o similar.

El plan gratuito duerme el servicio a los 15 minutos sin visitas y tarda cerca de
un minuto en despertar. Si molesta, el plan de pago más barato lo quita.

## Servidor propio o infraestructura externa (UNAV, LifeWatch)

```bash
docker build -t hiblooms .
docker run -p 8000:7860 \
  -e GEE_SERVICE_ACCOUNT_JSON="$(cat gee_key.json)" \
  -e HIBLOOMS_USERS="nico:contraseña" \
  -e HIBLOOMS_SECRET="un texto largo al azar" \
  -v hiblooms-data:/data \
  hiblooms
```

El volumen en `/data` conserva entre reinicios los shapefiles subidos, las
calibraciones y las cachés. Sin él, esos datos se pierden al reiniciar.

## Variables de entorno

| Variable | Para qué |
|---|---|
| `GEE_SERVICE_ACCOUNT_JSON` | Credenciales de Earth Engine (JSON o ruta a fichero). Sin ella, modo demo. |
| `HIBLOOMS_USERS` | Usuarios del login. Si falta, la web queda abierta. |
| `HIBLOOMS_SECRET` | Firma de las sesiones. |
| `HIBLOOMS_DATA_DIR` | Carpeta de datos de la instancia (por defecto `web/backend/_uploads`). |
| `PORT` | Puerto de escucha (7860 por defecto). |
| `CORS_ORIGINS` | Orígenes permitidos si la web se sirve desde otro dominio. |

## API para workflows

Documentación automática en `/docs`. Las operaciones largas
(`/api/monitor`, `/api/climatology`, `/api/calibrate`) devuelven un `job_id` que
se consulta en `/api/jobs/{id}`. Añadiendo `?wait=1` responden directamente, que
es lo cómodo para llamarlas desde un workflow.
