# Imagen base con Python preinstalado
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia archivos del proyecto
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Variable de entorno obligatoria para GEE
# Debes definir GEE_SERVICE_ACCOUNT_JSON como secreto en AWS y mapearlo a esta variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gee_credentials.json

# Crea el archivo JSON con el contenido de la variable de entorno en tiempo de ejecución
CMD bash -c "echo \"$GEE_SERVICE_ACCOUNT_JSON\" > /app/gee_credentials.json && streamlit run app.py --server.port 8080 --server.address 0.0.0.0"
