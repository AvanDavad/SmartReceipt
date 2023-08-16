import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import json
from scipy.optimize import least_squares

np.set_printoptions(suppress=True, precision=3)

PHASE_1_NUM_KEYPOINTS = 4

X0_LIST = [
    np.array([0.0, 0.0, 0.0, 0, 0, 800.0, 50.0]),
    np.array([0.0, 0.0, np.pi, 0, 0, 800.0, 50.0]),
    np.array([0.0, 0.0, 0.0, 0, 0, 100.0, 50.0]),
    np.array([0.0, 0.0, np.pi, 0, 0, 100.0, 50.0]),
]

def get_obj_points(height):
    objp = np.array(
        [[0, 0, 0], [64.0, 0, 0], [0, height, 0], [64.0, height, 0]]
    )
    return objp


def decode_x(x):
    rvec = x[0:3]
    tvec = x[3:6]
    height = x[6]
    obj_pts = get_obj_points(height)
    return rvec, tvec, height, obj_pts


def residual_function(x, gt_points, camera_matrix, dist_coeffs):
    rvec, tvec, _, obj_pts = decode_x(x)

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2)

    return (img_pts - gt_points).flatten()

def warp_perspective(img, transformation_matrix, dest_size_wh):
    out = cv2.warpPerspective(np.array(img), np.array(transformation_matrix), dest_size_wh)
    out_img = Image.fromarray(out)
    return out_img

def warp_perspective_with_nonlin_least_squares(
    img, img_pts, camera_matrix, dist_coeffs, scale_factor=10.0, verbose=True
):
    assert img_pts.shape == (PHASE_1_NUM_KEYPOINTS, 2), f"img_pts.shape is {img_pts.shape}, expected ({PHASE_1_NUM_KEYPOINTS}, 2)"

    for x0 in X0_LIST:
        print(f"Trying x0: {x0}")
        ret = least_squares(
            fun=residual_function,
            x0=x0,
            jac="2-point",
            args=(img_pts, camera_matrix, dist_coeffs),
            method="lm",
        )
        if ret.success and ret.x[-1] > 0:
            print(f"least_squares succeeded! solution: {ret.x}")
            break
        else:
            print(f"least_squares failed with message: {ret.message}")

    rvec, tvec, height, obj_pts = decode_x(ret.x)
    if verbose:
        print(f"rotation vector: {rvec}")
        print(f"translation vector: {tvec}")
        print(f"height: {height}")

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)

    dst = obj_pts[:, :2].astype(np.float32) * scale_factor
    dst += np.array([5.0, 5.0]) * scale_factor

    dst_width = int(dst[:, 0].max() + 5 * scale_factor)
    dst_height = int(dst[:, 1].max() + 20 * scale_factor)

    M = cv2.getPerspectiveTransform(img_pts[:, 0, :].astype(np.float32), dst)

    out_img = warp_perspective(img, M, (dst_width, dst_height))

    new_img_pts = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    return out_img, new_img_pts, M
