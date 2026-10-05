# ── HIBLOOMS · web viewer (React + FastAPI) in a single container ────────────
# A single service serves both the API and the web app, so only one link is needed.
# Designed for Hugging Face Spaces (port 7860) but works anywhere that
# runs containers (Cloud Run, LifeWatch, your own server).

# 1) Build the frontend
FROM node:20-slim AS front
WORKDIR /app
COPY web/frontend/package*.json ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build

# 2) Backend + built web app
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg

WORKDIR /app
COPY web/backend/requirements.txt web/backend/requirements.txt
RUN pip install --no-cache-dir -r web/backend/requirements.txt

# Project code and data (reservoir shapefiles, points, bulletin)
COPY hiblooms_core.py ./
COPY shapefiles/ shapefiles/
COPY data/ data/
COPY web/backend/ web/backend/
COPY --from=front /app/dist web/frontend/dist

# Instance data (uploaded shapefiles, calibrations, caches).
# On Hugging Face the filesystem is ephemeral: it is lost on restart.
ENV HIBLOOMS_DATA_DIR=/data
RUN mkdir -p /data && chmod 777 /data

# Default port; Render injects its own in PORT
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.backend.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
