import argparse

from src.camera_calib import get_camera_calib
from src.draw_utils import save_img_with_kps
from pathlib import Path
from PIL import Image
import numpy as np
from src.models.phase0points_model import CNNModulePhase0Points
from src.models.phase1line_model import CNNModulePhase1Line
from src.warp_perspective import warp_perspective_with_nonlin_least_squares


PROJ_DIR = Path(__file__).parents[2]

def get_val_loss_from_ckpt_path(ckpt_path):
    try:
        val_loss = float(ckpt_path.stem.split("=")[-1])
    except:
        val_loss = np.inf
    return val_loss

def get_best_ckpt_path(checkpoint_dir):
    ckpt_path = checkpoint_dir / "lightning_logs"

    best_ckpt_path = None
    best_val_loss = np.inf

    for version_dir in ckpt_path.iterdir():
        checkpoints_dir = version_dir / "checkpoints"
        for ckpt_file in checkpoints_dir.iterdir():
            val_loss = get_val_loss_from_ckpt_path(ckpt_file)
            if (best_ckpt_path is None) or (val_loss < best_val_loss):
                best_ckpt_path = ckpt_file
                best_val_loss = val_loss

    return best_ckpt_path

def main(args):
    out_folder = PROJ_DIR / args.out_folder
    out_folder.mkdir(exist_ok=True, parents=True)
    for file in out_folder.iterdir():
        file.unlink()

    ckpt_path = get_best_ckpt_path(PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points")
    model_0 = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    img_path = Path(args.img_filename)
    img = Image.open(img_path)

    # inference
    img_pts = model_0.inference(
        img_path,
        out_folder=out_folder,
        prefix="0_inference",
    )

    # warping
    camera_matrix, dist_coeffs = get_camera_calib()
    img, line_y, M = warp_perspective_with_nonlin_least_squares(
        img, img_pts, camera_matrix, dist_coeffs
    )

    save_img_with_kps(img, line_y, out_folder / "1_warped.jpg")

    # detecting lines

    ckpt_path = get_best_ckpt_path(PROJ_DIR / "model_checkpoints" / "CNNModulePhase1Line")
    model_1 = CNNModulePhase1Line().load_from_checkpoint(ckpt_path)
    model_1.inference(img, out_folder, prefix="2_lines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run inference")
    parser.add_argument("--img_filename", type=str, default="inp.jpg", help="path to image file")
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
