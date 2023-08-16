import argparse

from src.camera_calib import get_camera_calib
from src.draw_utils import save_img_with_kps, save_img_with_texts
from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.models import CNNModuleLineDetection
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
import numpy as np
from src.models.phase0points_model import CNNModulePhase0Points
from src.warp_perspective import warp_perspective_with_nonlin_least_squares

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
    model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    img = Image.open(args.img_filename)

    # inference
    img_path = args.img_filename
    img_pts = model.inference(
        Path(img_path),
        Phase0PointsDataset.TRANSFORMS,
        out_folder=out_folder,
        prefix="0_inference",
    )
    print(img_pts)

    # warping
    camera_matrix, dist_coeffs = get_camera_calib()
    img, kps, M = warp_perspective_with_nonlin_least_squares(
        img, img_pts, camera_matrix, dist_coeffs
    )

    save_img_with_kps(img, kps, out_folder / "1_warped.jpg")

    # detecting lines

    version_num = 16
    ckpt_name = "resnet-epoch=119-val_loss=0.04848"
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

        img_crop = img.crop((0, offset_y, width, offset_y + width))

        img_crop = np.array(img_crop)
        img_crop[int(width / 6) :, ...] = 128
        img_crop = Image.fromarray(img_crop)

        input_img = Phase0PointsDataset.TRANSFORMS(img_crop)
        input_img = input_img.unsqueeze(0)

        kps, is_last_logit = model(input_img)

        is_last_prob = model.sigmoid(is_last_logit).item()
        is_last = is_last_prob > 0.5

        kps = kps.detach().numpy().reshape(-1, 2) * width

        save_img_with_kps(
            img_crop,
            kps,
            out_folder / f"line_predict_{str(idx).zfill(3)}_{is_last_prob:.3f}.jpg",
        )

        save_img_with_texts(
            img_crop, kps, out_folder / f"text_predict_{str(idx).zfill(3)}.jpg"
        )

        kps_list.append(kps + np.array([0, offset_y]))

        offset_y += kps[:, 1].mean()

        idx += 1

    draw = ImageDraw.Draw(img)
    for kps in kps_list:
        for i0, i1, col in [
            (0, 1, "red"),
            (2, 3, "red"),
            (0, 2, "yellow"),
            (1, 3, "yellow"),
        ]:
            start_point = tuple(kps[i0].astype(np.int32).tolist())
            end_point = tuple(kps[i1].astype(np.int32).tolist())
            line_width = 5
            draw.line((start_point, end_point), fill=col, width=line_width)
    filename = out_folder / "lines.jpg"
    img.save(filename)
    print(f"saved {filename}")


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
