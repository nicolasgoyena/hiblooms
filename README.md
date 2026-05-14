# HIBLOOMS — Detection and Visualization of Cyanobacterial Blooms in Reservoirs

Official repository of the **HIBLOOMS** project.

## Contents

* **Web application** (Streamlit): Explore, analyze, and download water quality spectral indices derived from **Sentinel‑2** imagery, with comparison to *in situ* measurements. The app includes a set of predefined Spanish reservoirs but also allows users to upload their own shapefiles to visualize and analyze any reservoir.
* **REST API** (FastAPI): Asynchronous backend that handles heavy Earth Engine processing jobs, decoupled from the Streamlit frontend.
* **Auxiliary modules**: Core processing logic, calibration, database utilities, and i18n support.
* **CLI version** (`hiblooms_core.py`): Run the processing pipelines without a web interface.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-org>/hiblooms.git
   cd hiblooms
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   # Streamlit app
   pip install -r requirements.txt

   # API (only needed on the API server)
   pip install -r requirements_api.txt
   ```

## Usage

### Web Application (Streamlit)

```bash
streamlit run app.py
```

### REST API (FastAPI)

```bash
uvicorn api.main:app --reload
```

### CLI Version

```bash
python hiblooms_core.py --help
```

## Repository Structure

```
hiblooms/
├── .devcontainer/             # Optional configuration for container-based development
│   └── devcontainer.json
├── api/                       # FastAPI backend
│   ├── main.py                #   REST endpoints
│   └── worker.py              #   Asynchronous processing with Google Earth Engine
├── data/                      # Auxiliary data used by the app
│   ├── puntos_interes.csv     #   Sampling point coordinates
│   ├── clorofila_val_entero.csv
│   └── fechas_validas_*.csv   #   Precomputed valid dates by reservoir
├── images/                    # App logos and images
├── pages/                     # Additional Streamlit pages
│   ├── login.py
│   └── 2_Data_Catalog.py
├── scripts/                   # ETL and precomputation utilities
│   ├── actualizar_ficocianina.py
│   ├── descargar_cloro.py
│   ├── descargar_ficocianina.py
│   └── precalculo_fechas_optimizado.py
├── shapefiles/                # HIBLOOMS reservoir shapefile
│   ├── embalses_hiblooms.shp
│   ├── embalses_hiblooms.dbf
│   ├── embalses_hiblooms.shx
│   └── embalses_hiblooms.prj
├── app.py                     # Main Streamlit application and visualization interface
├── hiblooms_core.py           # GEE logic: indices, maps, statistics, and exports
├── hiblooms_calibration.py    # Calibration workflow and GEE-compatible map models
├── db_utils.py                # PostgreSQL connection and utilities
├── i18n.py                    # ES/EN translations
├── styles.css                 # Global app styles
├── requirements.txt           # Streamlit app dependencies
├── requirements_api.txt       # FastAPI backend dependencies
├── render.yaml                # Render.com deployment configuration
└── README.md
```

## Deployment

The app is deployed on [Render.com](https://render.com) as two separate services defined in `render.yaml`:

- **hiblooms-api**: FastAPI service running `api/main.py`
- **hiblooms-app**: Streamlit service running `app.py`

## Citation

If you use HIBLOOMS in your work, please cite the project as:

```
HIBLOOMS Project — Detection and Visualization of Cyanobacterial Blooms in Reservoirs.
University of Navarra, 2025.
GitHub repository: https://github.com/<your-org>/hiblooms
```

## License

