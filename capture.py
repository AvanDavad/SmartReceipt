import argparse
from pathlib import Path
import cv2, time
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from src.datasets.phase0points_dataset import Phase0PointsDataset

from src.models.phase0points_model import CNNModulePhase0Points
from src.visualization.font import get_font

PROJ_DIR = Path(__file__).parent
LIGHTNING_LOGS = PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points" / "lightning_logs"

def main(args):
    ckpt_path = (
        LIGHTNING_LOGS
        / f"version_{args.version_num}"
        / "checkpoints"
        / f"{args.ckpt_name}.ckpt"
    )
    model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    video = cv2.VideoCapture(0)

    a=0

    while True:
        a=a+1

        check, frame = video.read()

        h, w, _ = frame.shape
        left = (w - h) // 2
        right = w - left
        frame = frame[:, left:right, :]

        img = Image.fromarray(frame)
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
            circle_radius = 20
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

        for i0, i1, col in [(0, 1, "red"), (0, 2, "red"), (1, 3, "red"), (2, 3, "yellow")]:
            start_point = keypoints[i0]
            end_point = keypoints[i1]
            line_width = 5
            draw.line((start_point, end_point), fill=col, width=line_width)


        cv2.imshow("Capturing", np.array(img))

        key=cv2.waitKey(1)
        if key == 27:
            break
        else:
            pass #cv2.imshow("Please press the escape(esc) key to stop the video", frame)

    print(a)

    video.release()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version_num", type=int, default=0)
    argparser.add_argument("--ckpt_name", type=str, default="resnet-epoch=00-val_loss=0.00000")

    args = argparser.parse_args()
    main(args)
