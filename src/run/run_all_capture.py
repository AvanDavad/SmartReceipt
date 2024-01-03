import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw

from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.models.phase0_points.points_model import CNNModulePhase0Points
from src.path_utils import get_best_ckpt_path
from src.visualization.font import get_font

PROJ_DIR = Path(__file__).parents[2]


def make_square_by_cropping(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    elif w > h:
        margin = (w - h) // 2
        return img.crop((margin, 0, w - margin, h))
    else:
        margin = (h - w) // 2
        return img.crop((0, margin, w, h - margin))


def main(args):
    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points"
    )
    model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)
    print(f"Loading phase0 point model from {ckpt_path}")

    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()
        img = Image.fromarray(frame)
        img = make_square_by_cropping(img)
        assert img.width == img.height

        img_tensor = Phase0PointsDataset.TRANSFORMS(img)
        img_tensor = img_tensor.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            pred_kps = model(img_tensor)
        pred_kps = pred_kps.detach().cpu().numpy()[0]

        draw = ImageDraw.Draw(img)

        keypoints = []
        for i in range(pred_kps.shape[0] // 2):
            kpt = (
                int(img.width * pred_kps[2 * i]),
                int(img.height * pred_kps[2 * i + 1]),
            )
            keypoints.append(kpt)
            circle_radius = 10
            circle_color = "blue"
            draw.ellipse(
                (
                    kpt[0] - circle_radius,
                    kpt[1] - circle_radius,
                    kpt[0] + circle_radius,
                    kpt[1] + circle_radius,
                ),
                fill=circle_color,
            )

            # Draw text
            text = f"kpt_{i+1}"
            font_size = 25
            text_color = "black"
            font = get_font(font_size)
            text_position = kpt
            draw.text(text_position, text, fill=text_color, font=font)

        for i0, i1, col in [
            (0, 1, "red"),
            (0, 2, "red"),
            (1, 3, "red"),
            (2, 3, "yellow"),
        ]:
            start_point = keypoints[i0]
            end_point = keypoints[i1]
            line_width = 5
            draw.line((start_point, end_point), fill=col, width=line_width)

        cv2.imshow("Capturing", np.array(img))

        key = cv2.waitKey(100)
        if key != -1:
            print(key)
        if key == 13:  # Enter key
            break

    video.release()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    args = argparser.parse_args()
    main(args)
