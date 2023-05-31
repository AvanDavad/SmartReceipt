import argparse

import torch
from src.camera_calib import get_camera_calib
from src.draw_utils import save_img_with_kps
from src.models import CNNModule2Points, CNNModule6Points, CNNModuleLineDetection
from src.reader import ImageDataset, ImageReader, Top2PointsDataset
from pathlib import Path
import cv2
from scipy.optimize import least_squares
from PIL import Image
from PIL import ImageDraw
import numpy as np
from src.warp_perspective import warp_perspective
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")


def main(args):
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    version_num = args.version_num
    ckpt_name = args.ckpt_name
    ckpt_path = (
        PROJ_DIR
        / "model_checkpoints"
        / "CNNModule6Points"
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
        Path(img_path), ImageDataset.transforms, out_folder=out_folder, prefix="0_inference"
    )
    print(img_pts)

    # warping
    camera_matrix, dist_coeffs = get_camera_calib()
    img, kps, M = warp_perspective(img, img_pts, camera_matrix, dist_coeffs)

    save_img_with_kps(img, kps, out_folder / "1_warped.jpg")

    # detecting lines

    version_num = 11
    ckpt_name = "resnet-epoch=00-val_loss=0.12583"
    ckpt_path = (
        PROJ_DIR
        / "model_checkpoints"
        / "CNNModuleLineDetection"
        / "lightning_logs"
        / f"version_{version_num}"
        / "checkpoints"
        / f"{ckpt_name}.ckpt"
    )
    model = CNNModuleLineDetection().load_from_checkpoint(ckpt_path)
    model.eval()
    width = img.width
    is_last = False
    kps_list = []
    offset_y = int(width * 0.1)
    idx = 0
    while not is_last:
        if offset_y >= img.height:
            break

        img_crop = img.crop((0, offset_y, width, offset_y+width))

        img_crop = np.array(img_crop)
        img_crop[int(width/6):, ...] = 128
        img_crop = Image.fromarray(img_crop)

        input_img = ImageDataset.transforms(img_crop)
        input_img = input_img.unsqueeze(0)

        kps, is_last_logit = model(input_img)

        is_last_prob = model.sigmoid(is_last_logit).item()
        is_last = is_last_prob > 0.5

        kps = kps.detach().numpy().reshape(-1, 2) * width

        save_img_with_kps(img_crop, kps, out_folder / f"line_predict_{str(idx).zfill(3)}_{is_last_prob:.3f}.jpg")

        kps_list.append(kps + np.array([0, offset_y]))

        offset_y += kps[2:, 1].mean()

        idx +=1
    
    draw = ImageDraw.Draw(img)
    for kps in kps_list:
        for i0, i1, col in [(0, 1, "red"), (2, 3, "red"), (0, 2, "yellow"), (1, 3, "yellow")]:
            start_point = tuple(kps[i0].astype(np.int32).tolist())
            end_point = tuple(kps[i1].astype(np.int32).tolist())
            line_width = 5
            draw.line((start_point, end_point), fill=col, width=line_width)
    filename = out_folder / "lines.jpg"
    img.save(filename)
    print(f"saved {filename}")

    # # extracting text
    # text_data = pytesseract.image_to_data(img, output_type=Output.DICT)

    # # normalize the image
    # img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # img = Image.fromarray(img)

    # draw = ImageDraw.Draw(img)
    # # visualizing boxes
    # n_boxes = len(text_data["text"])
    # for i in range(n_boxes):
    #     if int(text_data["conf"][i]) > 60:
    #         (x, y, w, h) = (
    #             text_data["left"][i],
    #             text_data["top"][i],
    #             text_data["width"][i],
    #             text_data["height"][i],
    #         )
    #         draw.rectangle(
    #             (x, y, x + w, y + h),
    #             outline="red",
    #             width=3,
    #         )
    # img.save(out_folder / "text_boxes.jpg")


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
