"""
Configuration utilities for the multi-hazard exposure pipeline.

This module includes:
- YAML configuration loading
- Hazard-specific display specifications for plotting
"""

import yaml
from matplotlib.colors import LinearSegmentedColormap

# Custom saturated blue colormap (from light to dark)
custom_flood_cmap = LinearSegmentedColormap.from_list(
    "custom_flood", ["#eff3ff", "#6baed6", "#2171b5", "#08306b"]
)

def load_config(path):
    """
    Load a YAML configuration file.
    Parameters:
        path (str): Path to the YAML file.
    Returns:
        dict: Parsed configuration as a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_hazard_display_spec(hazard_name):
    """
    Get the plotting style and display settings for a specific hazard.
    Parameters:
        hazard_name (str): Name of the hazard (e.g. "earthquake", "landslide").
    Returns:
        dict: Plotting specification, including type, color map or palette, labels, and title.
    """
    if hazard_name == "landslide":
        return {
            "type": "continuous",
            "cmap": "YlOrRd",
            "label": "Landslide Susceptibility"
        }
    elif hazard_name == "earthquake":
        return {
            "type": "continuous",
            "cmap": "viridis",
            "label": "Seismic Hazard"
        }
    elif hazard_name in ["pluvial_flood", "fluvial_flood", "combined_flood"]:
        return {
            "type": "continuous",
            "cmap": custom_flood_cmap,
            "label": hazard_name.replace("_", " ").title()
        }
    else:
        return {
            "type": "continuous",
            "cmap": "viridis",
            "label": hazard_name.replace("_", " ").title()
        }
