import argparse
from src.camera_calib import get_camera_calib
from src.models import CNNModule6Points
from src.reader import ImageDataset, ImageReader
from pathlib import Path
import cv2
from scipy.optimize import least_squares
from PIL import Image
from PIL import ImageDraw
import numpy as np
from src.warp_perspective import warp_perspective

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")


def save_img_with_kps(
    img: Image, kps, filename, normalized=False, circle_radius=5, circle_color="blue"
):
    draw = ImageDraw.Draw(img)

    n = kps.shape[0]
    assert kps.shape == (n, 2), f"kps.shape is {kps.shape}, expected ({n}, 2)"

    for i in range(n):
        if normalized:
            kpt = (int(img.width * kps[i, 0]), int(img.height * kps[i, 1]))
        else:
            kpt = (int(kps[i, 0]), int(kps[i, 1]))

        draw.ellipse(
            (
                kpt[0] - circle_radius,
                kpt[1] - circle_radius,
                kpt[0] + circle_radius,
                kpt[1] + circle_radius,
            ),
            fill=circle_color,
        )

    img.save(filename)
    print(f"saved {filename}")


def main(args):
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True)

    version_num = args.version_num
    ckpt_name = args.ckpt_name
    ckpt_path = (
        PROJ_DIR
        / "model_checkpoints"
        / "CNNModule"
        / "lightning_logs"
        / f"version_{version_num}"
        / "checkpoints"
        / f"{ckpt_name}.ckpt"
    )
    model = CNNModule6Points().load_from_checkpoint(ckpt_path)

    img = Image.open(args.img_filename)

    # inference
    img_path = args.img_filename
    img_pts = model.inference(
        Path(img_path), ImageDataset.transforms, out_folder=out_folder
    )
    print(img_pts)

    # warping
    camera_matrix, dist_coeffs = get_camera_calib()
    img, kps = warp_perspective(img, img_pts, camera_matrix, dist_coeffs)

    save_img_with_kps(img, kps, out_folder / "warped.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run inference")
    parser.add_argument("--img_filename", type=str, help="path to image file")
    parser.add_argument(
        "--version_num", type=int, default=19, help="version number of model"
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="resnet-epoch=420-val_loss=0.00402",
        help="name of checkpoint",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="inference",
        help="folder to save inference results",
    )

    args = parser.parse_args()
    main(args)
