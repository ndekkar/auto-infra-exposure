"""
Configuration utilities for the multi-hazard exposure pipeline.

This module includes:
- YAML configuration loading
- Hazard-specific display specifications for plotting
"""

import yaml

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
    elif hazard_name == "combined_flood":
        return {
            "type": "discrete",
            "breaks": [0, 1, 3, 6, 10],
            "palette": ['#4465891A', '#4465893f', '#4465897f', '#446589bf'],
            "labels": ["Low", "Medium", "High", "Very High"],
            "label": "Combined Flood"
        }
    elif hazard_name == "pluvial_flood":
        return {
            "type": "discrete",
            "breaks": [0, 1, 2, 4, 6],
            "palette": ['#4465891A', '#4465893f', '#4465897f', '#446589bf'],
            "labels": ["Low", "Medium", "High", "Very High"],
            "label": "Pluvial Flood"
        }
    elif hazard_name == "fluvial_flood":
        return {
            "type": "discrete",
            "breaks": [0, 1, 3, 6, 10],
            "palette": ['#4465891A', '#4465893f', '#4465897f', '#446589bf'],
            "labels": ["Low", "Medium", "High", "Very High"],
            "label": "Fluvial Flood"
        }
    else:
        return {
            "type": "continuous",
            "cmap": "viridis",
            "label": hazard_name.replace("_", " ").title()
        }
