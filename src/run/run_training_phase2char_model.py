import argparse
from pytorch_lightning import Trainer
from src.datasets.phase2char_dataset import Phase2CharDataset
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
    image_reader_tr = ImageReader(args.train_data)
    image_reader_val = ImageReader(args.val_data)

    char_reader_tr = CharReader(image_reader_tr, w=args.w)
    char_reader_val = CharReader(image_reader_val, w=args.w)

    train_dataset = Phase2CharDataset(char_reader_tr, augment=True, shuffle=True)
    val_dataset = Phase2CharDataset(char_reader_val, augment=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    if args.from_scratch:
        model = CNNModulePhase2Chars()
    else:
        version_num = args.version_num
        ckpt_name = args.ckpt_name
        ckpt_path = (
            Path(args.model_checkpoints)
            / "CNNModulePhase2Chars"
            / "lightning_logs"
            / f"version_{version_num}"
            / "checkpoints"
            / f"{ckpt_name}.ckpt"
        )
        model = CNNModulePhase2Chars().load_from_checkpoint(ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="charmodel-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        default_root_dir=Path(args.model_checkpoints) / "CNNModulePhase2Chars",
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
    parser.add_argument("--w", type=int, default=5)
    parser.add_argument("--version_num", type=int)
    parser.add_argument(
        "--ckpt_name", type=str
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
