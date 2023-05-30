import numpy as np


import json


def get_camera_calib(calib_path = "/home/avandavad/projects/receipt_extractor/calib/camera_calib.json"):
    with open(calib_path, "r") as f:
        camera_calib = json.load(f)
    camera_matrix = np.array(camera_calib["camera_matrix"])
    dist_coeffs = np.array(camera_calib["distortion_coefficients"])

    return camera_matrix, dist_coeffs