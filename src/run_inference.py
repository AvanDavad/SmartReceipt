from src.models import CNNModule6Points
from src.reader import ImageDataset, ImageReader
from pathlib import Path

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")


def main():
    train_reader = ImageReader(PROJ_DIR / "data" / "train")
    val_reader = ImageReader(PROJ_DIR / "data" / "val")

    train_dataset = ImageDataset(train_reader, augment=True)
    val_dataset = ImageDataset(val_reader, augment=False)

    train_dataset[0]

    version_num = 18
    ckpt_name = "resnet-epoch=65-val_loss=0.00149"
    ckpt_path = (
        PROJ_DIR
        / "lightning_logs"
        / f"version_{version_num}"
        / "checkpoints"
        / f"{ckpt_name}.ckpt"
    )
    model = CNNModule6Points().load_from_checkpoint(ckpt_path)

    # inference
    img_path = PROJ_DIR / "data" / "test"
    for img_filename in img_path.glob("*.jpg"):
        model.inference(img_filename, val_dataset.transforms)


if __name__ == "__main__":
    main()
