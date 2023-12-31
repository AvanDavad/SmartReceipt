"""
python -m src.run.run_training_phase1line_model --from_scratch
"""

import argparse
from pytorch_lightning import Trainer
from src.readers.image_reader import ImageReader
from src.models.phase1line_model import CNNModulePhase1Line
from src.datasets.phase1line_dataset import Phase1LineDataset
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import time

PROJ_DIR = Path(__file__).parents[2]


def main(args):
    train_reader = ImageReader(args.train_data)
    val_reader = ImageReader(args.val_data)

    train_dataset = Phase1LineDataset(train_reader, augment=True)
    val_dataset = Phase1LineDataset(val_reader, augment=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4
    )

    if args.from_scratch:
        model = CNNModulePhase1Line()
    else:
        version_num = args.version_num
        ckpt_name = args.ckpt_name
        ckpt_path = (
            Path(args.model_checkpoints)
            / "CNNModulePhase1Line"
            / "lightning_logs"
            / f"version_{version_num}"
            / "checkpoints"
            / f"{ckpt_name}.ckpt"
        )
        model = CNNModulePhase1Line().load_from_checkpoint(ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="resnet-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        default_root_dir=Path(args.model_checkpoints) / "CNNModulePhase1Line",
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
        default=PROJ_DIR / "data" / "train",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=PROJ_DIR / "data" / "val",
    )
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--version_num", type=int, default=0)
    parser.add_argument(
        "--ckpt_name", type=str, default="resnet-epoch=65-val_loss=0.00149"
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
