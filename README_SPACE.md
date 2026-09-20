---
title: HIBLOOMS
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# HIBLOOMS · visor de floraciones algales

Vigilancia de floraciones algales en embalses españoles con Sentinel-2.
Proyecto PID2023-153234OB-I00 · Instituto BIOMA (Universidad de Navarra)
con las Confederaciones Hidrográficas del Ebro y del Júcar.

## Secretos que hay que configurar en el Space

| Nombre | Contenido |
|---|---|
| `GEE_SERVICE_ACCOUNT_JSON` | El JSON completo de la cuenta de servicio de Google Earth Engine |
| `HIBLOOMS_USERS` | `usuario:contraseña,otro:otra` (si falta, la web queda abierta) |
| `HIBLOOMS_SECRET` | Cualquier texto largo al azar, para firmar las sesiones |

Sin `GEE_SERVICE_ACCOUNT_JSON` el servicio arranca en modo demo con datos simulados.

## API

Documentación automática en `/docs`. Las operaciones largas
(`/api/monitor`, `/api/climatology`, `/api/calibrate`) devuelven un `job_id`
y se consultan en `/api/jobs/{id}`; añadiendo `?wait=1` responden directamente,
que es lo cómodo para integrarlas en un workflow.
