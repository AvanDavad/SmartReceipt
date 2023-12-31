import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.camera_calib import get_camera_calib
from src.draw_utils import draw_borders
from src.draw_utils import draw_borders_with_chars_and_probs
from src.draw_utils import save_img_with_kps
from src.models.phase0points_model import CNNModulePhase0Points
from src.models.phase1line_model import CNNModulePhase1Line
from src.models.phase2_single_char_model import CNNModulePhase2SingleChar
from src.models.phase2char_border_model import CNNModulePhase2CharsBorder
from src.warp_perspective import warp_perspective_with_nonlin_least_squares


PROJ_DIR = Path(__file__).parents[2]


def get_val_loss_from_ckpt_path(ckpt_path):
    try:
        val_loss = float(ckpt_path.stem.split("=")[-1])
    except:
        val_loss = np.inf
    return val_loss


def get_best_ckpt_path(checkpoint_dir, version: int = -1):
    ckpt_path = checkpoint_dir / "lightning_logs"

    best_ckpt_path = None
    best_val_loss = np.inf

    for version_dir in ckpt_path.iterdir():
        if version != -1 and version != int(version_dir.stem.split("_")[-1]):
            continue

        checkpoints_dir = version_dir / "checkpoints"
        for ckpt_file in checkpoints_dir.iterdir():
            val_loss = get_val_loss_from_ckpt_path(ckpt_file)
            if (best_ckpt_path is None) or (val_loss < best_val_loss):
                best_ckpt_path = ckpt_file
                best_val_loss = val_loss

    return best_ckpt_path


def main(args):
    out_folder: Path = PROJ_DIR / args.out_folder / "run_all"
    out_folder.mkdir(exist_ok=True, parents=True)
    for file in out_folder.iterdir():
        file.unlink()

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points"
    )
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

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase1Line"
    )
    model_1 = CNNModulePhase1Line().load_from_checkpoint(ckpt_path)
    line_image_list = model_1.inference(img, out_folder, prefix="2_lines")

    # reading lines

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase2CharsBorder"
    )
    model_border = CNNModulePhase2CharsBorder().load_from_checkpoint(ckpt_path)

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase2SingleChar"
    )
    model_char = CNNModulePhase2SingleChar().load_from_checkpoint(ckpt_path)

    line_text_list = []
    for line_idx, line_img in enumerate(line_image_list):
        x_coords = model_border.inference(line_img)

        img = draw_borders(line_img, x_coords)

        filename = str(out_folder / f"line_borders_{line_idx}.jpg")
        img.save(filename)
        print(f"Saved {filename}")

        line_chars = []
        probs = []
        for char_i in range(len(x_coords) - 1):
            char_img = line_img.crop(
                (x_coords[char_i], 0, x_coords[char_i + 1], line_img.height)
            )
            pred_char, prob = model_char.inference(char_img)
            line_chars.append(pred_char)
            probs.append(prob)
        line_text = "".join(line_chars)

        img_with_chars = draw_borders_with_chars_and_probs(
            line_img, x_coords[1:], line_chars, probs
        )

        filename = str(out_folder / f"line_chars_{line_idx}.jpg")
        img_with_chars.save(filename)
        print(f"Saved {filename}")

        line_text_list.append(line_text)

    with open(out_folder / "infered_text.txt", "w") as f:
        f.write("\n".join(line_text_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run inference")
    parser.add_argument(
        "--img_filename", type=str, default="inp.jpg", help="path to image file"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="inference",
        help="folder to save inference results",
    )

    args = parser.parse_args()
    main(args)
