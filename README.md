
# Multi-Hazard Infrastructure Exposure Pipeline

End-to-end geospatial pipeline for assessing infrastructure exposure to multiple hazards(floods, landslides, heat, wildfire, drought, cold, earthquakes). The workflow ingestsAOI boundaries, infrastructure assets (points and lines), and hazard rasters, then produces exposure layers, maps, and summary statistics.

---

## Key Features

- **Multi-hazard exposure analysis** (flood, landslide, earthquake, drought, wildfire, heat, cold)
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
- Heat 
- Wildfire
- Cold

## Installation
### 1) Clone the repository
### 2) Create and activate a virtual environment
### 3) Install dependencies
```
pip install -r requirements.txt
```
## Configuration
Edit `config.yml` to point to your data inputs and outputs.


