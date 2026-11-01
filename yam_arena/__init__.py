"""Bimanual I2RT YAM for Isaac Lab-Arena 0.3.x (Isaac Lab 3.0, Isaac Sim 6.1).

Importing this package is cheap: nothing here imports Isaac Sim, Isaac Lab or Arena.
The Isaac-dependent modules (``robot``, ``actions``, ``embodiment``, ``scene``) must be
imported only after ``AppLauncher`` has started the app.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = PACKAGE_DIR / "assets" / "yam"
YAM_URDF = ASSET_DIR / "yam_arm.urdf"
# Output of scripts/convert_usd.py. The URDF importer writes {usd_dir}/{robot_name}/{robot_name}.usda.
YAM_USD_DIR = ASSET_DIR / "usd"
YAM_USD = YAM_USD_DIR / "yam_arm" / "yam_arm.usda"


def require_usd() -> Path:
    """Return the converted YAM USD, or explain how to make it."""
    if not YAM_USD.is_file():
        raise FileNotFoundError(f"{YAM_USD} does not exist. Convert the URDF first:\n    python scripts/convert_usd.py")
    return YAM_USD
