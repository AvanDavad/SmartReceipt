import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from src.camera_calib import get_default_camera_calib

from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.draw_utils import draw_borders, draw_borders_with_chars_and_probs, make_square_by_cropping, save_img_with_kps
from src.models.phase0_points.points_model import CNNModulePhase0Points
from src.models.phase1line_model import CNNModulePhase1Line
from src.models.phase2_single_char_model import CNNModulePhase2SingleChar
from src.models.phase2char_border_model import CNNModulePhase2CharsBorder
from src.path_utils import get_best_ckpt_path
from src.visualization.font import get_font
from src.warp_perspective import warp_perspective_with_nonlin_least_squares

PROJ_DIR = Path(__file__).parents[2]


def main(args):
    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points"
    )
    model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)
    print(f"Loading phase0 point model from {ckpt_path}")

    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()
        input_img = Image.fromarray(frame[..., ::-1])

        img = model.inference_and_visualize(input_img)

        cv2.imshow("Capturing", np.array(img)[..., ::-1])

        key = cv2.waitKey(100)
        if key != -1:
            print(key)
        if key == 13:  # Enter key
            break

    video.release()

    out_folder = Path(args.out_folder) / "run_all"
    out_folder.mkdir(parents=True, exist_ok=True)
    for item in out_folder.iterdir():
        item.unlink()

    input_img = make_square_by_cropping(input_img)
    pred_kps = model.inference(input_img)

    save_img_with_kps(input_img, pred_kps, out_folder / "0_points.jpg")

    # warping
    camera_matrix, dist_coeffs = get_default_camera_calib(input_img.size)
    ortho_img, points_on_ortho, tr_matrix = warp_perspective_with_nonlin_least_squares(
        input_img, pred_kps, camera_matrix, dist_coeffs
    )

    save_img_with_kps(ortho_img, points_on_ortho, out_folder / "1_warped.jpg")

    # detecting lines

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase1Line"
    )
    line_model = CNNModulePhase1Line().load_from_checkpoint(ckpt_path)
    line_image_list = line_model.inference(ortho_img, out_folder, prefix="2_lines")


    # reading lines

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase2CharsBorder"
    )
    border_model = CNNModulePhase2CharsBorder().load_from_checkpoint(ckpt_path)

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase2SingleChar"
    )
    char_model = CNNModulePhase2SingleChar().load_from_checkpoint(ckpt_path)

    line_text_list = []
    for line_idx, line_img in enumerate(line_image_list):
        x_coords = border_model.inference(line_img)

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
            pred_char, prob = char_model.inference(char_img)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_folder",
        type=str,
        default="inference",
        help="folder to save inference results",
    )

    args = parser.parse_args()
    main(args)
