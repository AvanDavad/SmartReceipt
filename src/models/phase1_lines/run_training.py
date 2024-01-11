"""
python -m src.run.run_training_phase1line_model --from_scratch
"""
import argparse
import time
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import wandb
from src.datasets.phase1line_dataset import Phase1LineDataset
from src.models.phase1_lines.line_model import CNNModulePhase1Line
from src.path_utils import get_best_ckpt_path
from src.readers.image_reader import ImageReader

PROJ_DIR = Path(__file__).parents[3]
MODEL_CHECKPOINTS = PROJ_DIR / "model_checkpoints" / "CNNModulePhase1Line"
assert MODEL_CHECKPOINTS.is_dir(), MODEL_CHECKPOINTS


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
        ckpt_path = get_best_ckpt_path(
            MODEL_CHECKPOINTS, args.version_num, is_last=args.last
        )
        print(f"loading from checkpoint: {ckpt_path}")
        model = CNNModulePhase1Line().load_from_checkpoint(ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="line_model-{epoch:02d}-{val_loss:.5f}",
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
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--from_scratch", action="store_true", help="train from scratch"
    )
    parser.add_argument(
        "--version_num",
        type=int,
        default=-1,
        help="continue training from this version",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="train from last in specified version",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="max epochs"
    )
    parser.add_argument("--dropout_prob", type=float, default=0.0)

    args = parser.parse_args()

    wandb.init(
        project="phase1 (line) training",
        config={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.max_epochs,
            "dropout_prob": args.dropout_prob,
        },
    )

    main(args)

    wandb.finish()
