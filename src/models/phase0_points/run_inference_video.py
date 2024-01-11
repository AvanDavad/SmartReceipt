import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.draw_utils import make_square_by_cropping
from src.models.phase0_points.points_model import CNNModulePhase0Points
from src.path_utils import get_best_ckpt_path

PROJ_DIR = Path(__file__).parents[3]
MODEL_CHECKPOINTS = PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points"
assert MODEL_CHECKPOINTS.is_dir(), MODEL_CHECKPOINTS


def main(args):
    ckpt_path = get_best_ckpt_path(
        MODEL_CHECKPOINTS, args.version_num, is_last=args.last
    )
    print(f"loading from checkpoint: {ckpt_path}")
    model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path).eval()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)

    out_images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_img = Image.fromarray(frame[..., ::-1])
        if args.rotate_left:
            input_img = input_img.rotate(90)
        input_img = make_square_by_cropping(input_img)

        out_img = model.inference_and_visualize(input_img)
        out_images.append(out_img)

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    filename = "output_video.mp4"
    out = cv2.VideoWriter(
        filename, fourcc, fps, (out_images[0].size[0], out_images[0].size[1])
    )

    for img in out_images:
        out.write(np.array(img)[..., ::-1])

    out.release()

    print(f"saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        "-i",
        type=str,
        default="inp.jpg",
        help="input image",
    )
    parser.add_argument(
        "--version_num",
        type=int,
        default=-1,
        help="use specified version number",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="use last version in specified version instead of best",
    )
    parser.add_argument(
        "--rotate_left",
        action="store_true",
        help="rotate the video left 90 degrees",
    )

    args = parser.parse_args()
    main(args)
