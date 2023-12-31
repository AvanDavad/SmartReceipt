import argparse
import cv2
from PIL import Image
import numpy as np
import pathlib
import json

np.set_printoptions(suppress=True, precision=3)


def main(args):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * args.grid_size
    imgpoints = []
    objpoints = []

    out_corners_dir = pathlib.Path(args.img_dir) / "corners"
    out_corners_dir.mkdir(exist_ok=True)

    for filename in pathlib.Path(args.img_dir).glob(f"*.{args.img_ext}"):
        print(filename)
        img = cv2.imread(str(filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        assert (
            ret
        ), f"failed to find corners in {filename}. remove it and start over"
        cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )
        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        Image.fromarray(img[..., ::-1]).save(out_corners_dir / filename.name)

    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
    )
    print(f"camera matrix:\n{mtx}\n")
    print(f"distortion coefficients:\n{dist}\n")

    # saving camera matrix and distortion coefficients
    camera_calib = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
    }
    filename = pathlib.Path(args.img_dir) / args.output

    with open(filename, "w") as f:
        json.dump(camera_calib, f)

    print(f"saved camera matrix and distortion coefficients to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera")
    parser.add_argument(
        "--grid_size", type=float, default=21.8, help="size of grid in mm"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="calib",
        help="directory containing calibration images",
    )
    parser.add_argument(
        "--img_ext",
        type=str,
        default="jpeg",
        help="extension of calibration images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_calib.json",
        help="output file for camera matrix and distortion coefficients",
    )
    args = parser.parse_args()

    main(args)
