# ── HIBLOOMS · visor web (React + FastAPI) en un solo contenedor ─────────────
# Un único servicio sirve la API y la web, así que solo hace falta un enlace.
# Pensado para Hugging Face Spaces (puerto 7860) pero vale para cualquier sitio
# que ejecute contenedores (Cloud Run, LifeWatch, un servidor propio).

# 1) Compilar el frontend
FROM node:20-slim AS front
WORKDIR /app
COPY web/frontend/package*.json ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build

# 2) Backend + web compilada
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY web/backend/requirements.txt web/backend/requirements.txt
RUN pip install --no-cache-dir -r web/backend/requirements.txt

# Código y datos del proyecto (shapefiles de embalses, puntos, boletín)
COPY hiblooms_core.py hiblooms_calibration.py ./
COPY shapefiles/ shapefiles/
COPY data/ data/
COPY web/backend/ web/backend/
COPY --from=front /app/dist web/frontend/dist

# Datos de la instancia (shapefiles subidos, calibraciones, cachés).
# En Hugging Face el sistema de archivos es efímero: si se reinicia, se pierden.
ENV HIBLOOMS_DATA_DIR=/data
RUN mkdir -p /data && chmod 777 /data

# Puerto por defecto; Render inyecta el suyo en PORT
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.backend.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
