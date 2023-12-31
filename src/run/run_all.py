import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

from src.camera_calib import get_camera_calib
from src.draw_utils import draw_borders
from src.draw_utils import draw_borders_with_chars_and_probs
from src.draw_utils import save_img_with_kps
from src.models.phase0_points.points_model import CNNModulePhase0Points
from src.models.phase1line_model import CNNModulePhase1Line
from src.models.phase2_single_char_model import CNNModulePhase2SingleChar
from src.models.phase2char_border_model import CNNModulePhase2CharsBorder
from src.path_utils import get_best_ckpt_path
from src.warp_perspective import warp_perspective_with_nonlin_least_squares


PROJ_DIR = Path(__file__).parents[2]


def run_extraction(
    img: Image.Image,
    out_folder: Optional[Path] = None,
    phase0_folder: Path = PROJ_DIR
    / "model_checkpoints"
    / "CNNModulePhase0Points",
):
    ckpt_path = get_best_ckpt_path(phase0_folder)
    model_0 = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    # inference
    img_pts = model_0.inference_and_visualize(
        img_path,
        out_folder=out_folder,
        save_filename="0_inference",
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
    img_pts = model_0.inference_and_visualize(
        img_path,
        out_folder=out_folder,
        save_filename="0_inference",
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
