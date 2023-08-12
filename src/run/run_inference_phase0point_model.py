"""
python -m src.run.run_inference_phase0point_model \
    --version_num 0 \
    --ckpt_name resnet-epoch=00-val_loss=0.00000

"""

import argparse
from src.readers.image_reader import ImageReader
from src.models.phase0points_model import CNNModulePhase0Points
from src.datasets.phase0points_dataset import Phase0PointsDataset
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent.parent
LIGHTNING_LOGS = PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points" / "lightning_logs"

def main(args):
    ckpt_path = (
        LIGHTNING_LOGS
        / f"version_{args.version_num}"
        / "checkpoints"
        / f"{args.ckpt_name}.ckpt"
    )
    model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    # inference
    img_path = PROJ_DIR / "data" / "test"
    out_folder = PROJ_DIR / "inference" / f"version_{args.version_num}"
    out_folder.mkdir(exist_ok=True, parents=True)
    for img_filename in img_path.glob("*.jpg"):
        model.inference(
            img_filename,
            out_folder=out_folder,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version_num", type=int, default=0)
    argparser.add_argument("--ckpt_name", type=str, default="resnet-epoch=00-val_loss=0.00000")

    args = argparser.parse_args()
    main(args)
