import argparse
from pytorch_lightning import Trainer
from src.readers.image_reader import ImageReader
from src.models import CNNModuleLineDetection
from src.line_dataset import LineDataset
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import time


def main(args):
    train_reader = ImageReader(args.train_data)
    val_reader = ImageReader(args.val_data)

    train_dataset = LineDataset(train_reader, augment=True, shuffle=True)
    val_dataset = LineDataset(val_reader, augment=False, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4
    )

    if args.from_scratch:
        model = CNNModuleLineDetection()
    else:
        version_num = args.version_num
        ckpt_name = ""
        ckpt_path = (
            Path(args.model_checkpoints)
            / "CNNModuleLineDetection"
            / "lightning_logs"
            / f"version_{version_num}"
            / "checkpoints"
            / f"{ckpt_name}.ckpt"
        )
        model = CNNModuleLineDetection().load_from_checkpoint(ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="resnet-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        default_root_dir=Path(args.model_checkpoints)
        / "CNNModuleLineDetection",
        accelerator="gpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=3,
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
        default="/home/avandavad/projects/receipt_extractor/data/train",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="/home/avandavad/projects/receipt_extractor/data/val",
    )
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--version_num", type=int, default=0)
    parser.add_argument(
        "--ckpt_name", type=str, default="resnet-epoch=85-val_loss=0.04748"
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        default="/home/avandavad/projects/receipt_extractor/model_checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=1000)

    args = parser.parse_args()

    main(args)
