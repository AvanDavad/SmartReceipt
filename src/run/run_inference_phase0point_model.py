import argparse
from pathlib import Path

from PIL import Image

from src.models.phase0_points.points_model import CNNModulePhase0Points

PROJ_DIR = Path(__file__).parent.parent.parent
LIGHTNING_LOGS = (
    PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points" / "lightning_logs"
)


def main(args):
    ckpt_path = (
        LIGHTNING_LOGS
        / f"version_{args.version_num}"
        / "checkpoints"
        / f"{args.ckpt_name}.ckpt"
    )
    CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    # inference
    PROJ_DIR / "data" / "test"
    PROJ_DIR / "data" / "train"
    out_folder_test = (
        PROJ_DIR / "inference" / f"version_{args.version_num}" / "test"
    )
    out_folder_train = (
        PROJ_DIR / "inference" / f"version_{args.version_num}" / "train"
    )


out_folder_test.mkdir(exist_ok=True, parents=True)
out_folder_train.mkdir(exist_ok=True, parents=True)

for img_filename in img_path_test.glob("*.jpg"):
    model.inference(
        img_filename,
        out_folder=out_folder_test,
    )

for img_filename in img_path_train.glob("*.jpg"):
    model.inference(
        Image.open(img_filename),
        out_folder=out_folder_train,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version_num", type=int, default=0)
    argparser.add_argument(
        "--ckpt_name", type=str, default="resnet-epoch=00-val_loss=0.00000"
    )

    args = argparser.parse_args()
    main(args)
