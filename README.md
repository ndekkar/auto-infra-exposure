# Multi-Hazard Infrastructure Exposure Pipeline

End-to-end geospatial pipeline for assessing infrastructure exposure to multiple hazards (floods, landslides, heat, wildfire, drought, cold, earthquakes). The workflow ingests AOI boundaries, infrastructure assets (points and lines), and hazard rasters, then produces exposure layers, maps, and summary statistics.

---

## Key Features

- **Multi-hazard exposure analysis** (flood, landslide, earthquake, wildfire, heat, cold)
- **Single YAML configuration** for inputs, outputs, and hazard toggles
- **Point and line infrastructure support**
- **Automated map generation** per hazard
- **Stats outputs** from either exposure shapefiles or raster overlays

---

## Supported Hazards

- Fluvial Flood
- Pluvial Flood
- Combined Flood
- Landslide
- Earthquake
- Wildfire
- Heat
- Cold

---

## Installation

### 1) Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

(or use Conda if preferred)

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit `config.yaml` to define:
- area(s) of interest (AOI),
- infrastructure input layers (points and lines),
- hazard inputs,
- output directory,
- hazards to activate or deactivate.

All notebooks rely on this configuration file.

---

## Run the Pipeline

The pipeline is designed to be executed **using Jupyter notebooks**.
No modification of the Python source code is required.

### Step 1 — Download required data (if needed)

Before running the exposure analysis, some input datasets may need to be downloaded.

Use the following notebooks as required:

- `download_dem.ipynb`  
  → Download and prepare DEM data

- `download_cold_data.ipynb`  
  → Download and prepare cold hazard input data

- `download_heat_data.ipynb`  
  → Download and prepare heat hazard input data

These notebooks usually need to be run **once per study area**, or when data updates are required.

---

### Step 2 — Run exposure analysis (quickstart / interactive)

To run an exposure analysis interactively and inspect results step by step, use:

- `run_exposure_quickstart.ipynb`

This notebook allows users to:
- run exposure analysis incrementally,
- visualize intermediate and final results,
- verify that inputs and outputs are correctly configured.

---

### Step 3 — Run full multi-hazard analysis (one-click)

To run all activated hazards automatically in a single execution, use:

- `01_run_multi_hazard_analysis.ipynb`

Simply run all cells in the notebook.
All exposure layers, maps, and statistics defined in `config.yaml` will be generated automatically.

---

## Outputs

All outputs are written to the directory defined in `config.yaml`.

Depending on the configuration, outputs may include:
- exposure vector layers (points and lines),
- hazard-specific exposure maps,
- summary statistics files.

---

## Notes

- All notebooks can be run independently.
- Notebooks can be safely re-run if needed.
- No Python programming knowledge is required.
