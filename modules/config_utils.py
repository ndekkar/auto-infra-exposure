import yaml

def load_config(path):
    """Load YAML configuration from a file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_hazard_display_spec(hazard_name):
    """Return plotting style and parameters for a given hazard."""
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
