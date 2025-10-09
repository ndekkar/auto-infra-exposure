"""
Configuration utilities for the multi-hazard exposure pipeline.

This module includes:
- YAML configuration loading
- Hazard-specific display specifications for plotting
"""
import yaml
from matplotlib.colors import ListedColormap


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

def get_hazard_display_spec(hazard_name: str):
    """
    Return a display spec dict for the given hazard.

    Keys used elsewhere:
      - type: "discrete" | "continuous"
      - palette: list of hex colors (for discrete)
      - breaks: list of class edges (for discrete)
      - labels: list of class labels (for discrete)
      - cmap: matplotlib colormap name or object (for continuous)
      - label: short name for the hazard
      - legend_title: title to show above the merged legend
    """
    hn = (hazard_name or "").lower().strip()

    # ----- DISCRETE HAZARDS -----
    # Common classes for discrete hazards
    disc_labels = ["Very low", "Low", "Medium", "High", "Very high"]

    # You can tweak these palettes to your project’s exact colors
    flood_palette = ["#f7fbff",  "#deebf7","#9ecae1", "#3182bd","#08519c"]

    if hn == "pluvial_flood":
        return {
            "type": "discrete",
            "palette": flood_palette,
            "breaks": [1, 2, 3, 4, 5, 6],   # 5 classes → 6 edges
            "labels": disc_labels,
            "label": "Pluvial Flood",
            "legend_title": "Pluvial Flood Hazard",
        }

    elif hn == "fluvial_flood":
        return {
            "type": "discrete",
            "palette": flood_palette,
            "breaks": [1, 2, 3, 4, 5, 6],
            "labels": disc_labels,
            "label": "Fluvial Flood",
            "legend_title": "Fluvial Flood Hazard",
        }

    elif hn == "combined_flood":
        return {
            "type": "discrete",
            "palette": flood_palette,
            "breaks": [1, 2, 3, 4, 5, 6],
            "labels": disc_labels,
            "label": "Combined Flood",
            "legend_title": "Combined Flood Hazard",
        }

    elif hn == "landslide":
        return {
            "type": "continuous",
            "cmap": "YlOrRd",
            "label": "Landslide",
            "legend_title": "Landslide Susceptibility",
        }

    elif hn == "earthquake":
        return {
            "type": "continuous",
            "cmap": "viridis",
            "label": "Earthquake",
            "legend_title": "Earthquake Intensity",
        }

    elif hn == "wildfire":
        return {
            "type": "continuous",
            "cmap": "Reds",
            "label": "Wildfire",
            "legend_title": "Wildfire Density",
        }
        
    elif hn == "heat":
        return {
           "type": "continuous",
           "cmap": "YlOrRd",         
           "label": "Heat",
           "legend_title": "Heat intensity ",
    }

    # ----- DEFAULT / UNKNOWN -----
    return {
        "type": "continuous",
        "cmap": "viridis",
        "label": hazard_name.replace("_", " ").title() if hazard_name else "Hazard",
        "legend_title": hazard_name.replace("_", " ").title() if hazard_name else "Hazard",
    }
