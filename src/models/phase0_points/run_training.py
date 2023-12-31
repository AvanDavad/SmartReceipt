import argparse
import time
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.models.phase0_points.points_model import CNNModulePhase0Points
from src.readers.image_reader import ImageReader
from src.run.run_all import get_best_ckpt_path

PROJ_DIR = Path(__file__).parents[2]
MODEL_CHECKPOINTS = PROJ_DIR / "model_checkpoints" / "CNNModulePhase0Points"

def main(args):
    train_reader = ImageReader(args.train_data)
    val_reader = ImageReader(args.val_data)

    train_dataset = Phase0PointsDataset(train_reader, augment=True)
    val_dataset = Phase0PointsDataset(val_reader, augment=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4
    )

    if args.from_scratch:
        model = CNNModulePhase0Points()
    else:
        ckpt_path = get_best_ckpt_path(MODEL_CHECKPOINTS, args.version_num)
        model = CNNModulePhase0Points().load_from_checkpoint(ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="phase0-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        default_root_dir=MODEL_CHECKPOINTS,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )
    trainer.validate(model, val_dataloader)

    print("starting training...")
    time.sleep(5)

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        default=str(PROJ_DIR / "data" / "train"),
        help="path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=str(PROJ_DIR / "data" / "val"),
        help="path to validation data",
    )
    parser.add_argument("--from_scratch", action="store_true", help="train from scratch")
    parser.add_argument("--version_num", type=int, default=-1, help="continue training from this version")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--max_epochs", type=int, default=1000, help="max epochs")

    args = parser.parse_args()

    main(args)
