
## Multi-Hazard Infrastructure Exposure Pipeline

This project implements a multi-hazard exposure assessment pipeline for critical infrastructure (e.g., power lines and substations). It processes geospatial data from various hazards such as floods, droughts, heatwaves, wildfires, earthquakes, and landslides, and generates both analytical outputs and maps.

---

## Features

- Multi-hazard exposure analysis (flood, landslide, earthquake, drought, wildfire, heat)
- Raster and vector data processing
- Reprojecting and harmonizing coordinate systems
- Earth Engine integration for drought (NDMI)
- NetCDF analysis for heatwave statistics
- Infrastructure exposure for both point and line features
- Automated map generation for each hazard
- Configurable via a single YAML file

---

## Supported Hazards

-Fluvial Flood
-Pluvial Flood
-Combined Flood
-Landslide
-Earthquake
-Drought (NDMI from MODIS)
-Heat (ERA5 Tmax NetCDF)
-Wildfire (centroid density from GlobFire)

## Installation
### 1. Clone the repository
### 2. Create a virtual environment
### 3. Install required packages
pip install -r requirements.txt

