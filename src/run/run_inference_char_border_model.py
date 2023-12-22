"""
python -m src.run.run_inference_phase0point_model \
    --version_num 0 \
    --ckpt_name resnet-epoch=00-val_loss=0.00000

"""

import argparse
from src.draw_utils import draw_borders
from src.models.phase2char_border_model import CNNModulePhase2CharsBorder
from src.readers.image_reader import ImageReader
from src.models.phase0points_model import CNNModulePhase0Points
from src.datasets.phase0points_dataset import Phase0PointsDataset
from pathlib import Path
from PIL import Image

from src.run.run_all import get_best_ckpt_path

PROJ_DIR = Path(__file__).parent.parent.parent

def main(args):
    ckpt_path = get_best_ckpt_path(PROJ_DIR / "model_checkpoints" / "CNNModulePhase2CharsBorder")
    # ckpt_path = ckpt_path.parents[2] / f"version_8" / "checkpoints" / "charbordermodel-epoch=07-val_loss=0.27174.ckpt"
    print(f"Loading model from {ckpt_path}")
    model = CNNModulePhase2CharsBorder().load_from_checkpoint(ckpt_path)

    out_folder = PROJ_DIR / "inference" / "phase2char_border_model"
    out_folder.mkdir(exist_ok=True, parents=True)

    input_image = Image.open("line_image.jpg")
    input_image = input_image.crop(
        (0, int(input_image.height * args.top), input_image.width, int(input_image.height * args.bottom))
    )

    x_coords = model.inference(input_image)
    img = draw_borders(input_image, x_coords)

    filename = str(out_folder / "inference.jpg")
    img.save(filename)
    print(f"Saved {filename}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--top", type=float, default=0.0)
    argparser.add_argument("--bottom", type=float, default=1.0)

    args = argparser.parse_args()
    main(args)
