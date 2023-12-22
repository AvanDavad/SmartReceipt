import argparse
from pytorch_lightning import Trainer
from src.datasets.phase2char_border_dataset import Phase2CharBorderDataset
from src.datasets.phase2char_dataset import Phase2CharDataset
from src.models.phase2char_border_model import CNNModulePhase2CharsBorder
from src.models.phase2char_models import CNNModulePhase2Chars
from src.readers.char_reader import CharReader
from src.readers.image_reader import ImageReader
from src.models.phase1line_model import CNNModulePhase1Line
from src.datasets.phase1line_dataset import Phase1LineDataset
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import time

PROJ_DIR = Path(__file__).parents[2]

def main(args):

    char_reader_tr = CharReader(args.train_data)
    char_reader_val = CharReader(args.val_data)

    train_dataset = Phase2CharBorderDataset(char_reader_tr, augment=True, shuffle=True)
    val_dataset = Phase2CharBorderDataset(char_reader_val, augment=False, shuffle=False)

    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    if args.from_scratch:
        model = CNNModulePhase2CharsBorder()
        print("model initialized from scratch")
    else:
        version_num = args.version_num
        ckpt_name = args.ckpt_name
        ckpt_path = (
            Path(args.model_checkpoints)
            / "CNNModulePhase2CharsBorder"
            / "lightning_logs"
            / f"version_{version_num}"
            / "checkpoints"
            / f"{ckpt_name}.ckpt"
        )
        model = CNNModulePhase2CharsBorder().load_from_checkpoint(ckpt_path)
        print(f"model initialized from {ckpt_path}")
    model.lr = args.lr
    model.weight_decay = args.weight_decay

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="charbordermodel-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        default_root_dir=Path(args.model_checkpoints) / "CNNModulePhase2CharsBorder",
        accelerator="gpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )
    trainer.validate(model, val_dataloader)

    print("starting training...")
    time.sleep(10.0)

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
    parser.add_argument("--version_num", type=int)
    parser.add_argument(
        "--ckpt_name", type=str
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        default=PROJ_DIR / "model_checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()

    main(args)
