import json
from pathlib import Path

import numpy as np

PROJ_DIR = Path(__file__).parents[2]
DEFAULT_CALIB_PATH = PROJ_DIR / "calib" / "camera_calib.json"


def get_camera_calib(
    calib_path=DEFAULT_CALIB_PATH,
):
    with open(calib_path, "r") as f:
        camera_calib = json.load(f)
    camera_matrix = np.array(camera_calib["camera_matrix"])
    dist_coeffs = np.array(camera_calib["distortion_coefficients"])

    return camera_matrix, dist_coeffs
