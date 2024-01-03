import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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

    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()
        img = Image.fromarray(frame[..., ::-1])

        img = model.inference_and_visualize(img)

        cv2.imshow("Capturing", np.array(img)[..., ::-1])

        key = cv2.waitKey(100)
        if key != -1:
            print(key)
        if key == 13:  # Enter key
            break

    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()
    main(args)
