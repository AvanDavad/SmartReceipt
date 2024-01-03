import json
from pathlib import Path
from typing import Tuple

import numpy as np

PROJ_DIR = Path(__file__).parents[1]
DEFAULT_CALIB_PATH = PROJ_DIR / "calib" / "camera_calib.json"


def get_camera_calib(
    calib_path=DEFAULT_CALIB_PATH,
):
    assert calib_path.is_file(), calib_path
    with open(calib_path, "r") as f:
        camera_calib = json.load(f)
    camera_matrix = np.array(camera_calib["camera_matrix"])
    dist_coeffs = np.array(camera_calib["distortion_coefficients"])

    return camera_matrix, dist_coeffs

def get_default_camera_calib(img_size_wh: Tuple[int, int]):
    w, h = img_size_wh
    fx = 1.2 * w
    fy = 1.2 * h
    cx = w / 2
    cy = h / 2
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)
    return camera_matrix, dist_coeffs
