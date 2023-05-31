import argparse
from pytorch_lightning import Trainer
from src.models import CNNModuleLineDetection
from src.reader import ImageDataset, ImageReader, LineDataset
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import time

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")


def main(args):
    train_reader = ImageReader(PROJ_DIR / "data" / "train")
    val_reader = ImageReader(PROJ_DIR / "data" / "val")

    train_dataset = LineDataset(train_reader, augment=True, shuffle=True)
    val_dataset = LineDataset(val_reader, augment=False, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    if args.from_scratch:
        model = CNNModuleLineDetection()
    else:
        version_num = 10
        ckpt_name = "resnet-epoch=52-val_loss=0.02074"
        ckpt_path = (
            PROJ_DIR
            / "model_checkpoints"
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
        default_root_dir=PROJ_DIR / "model_checkpoints" / "CNNModuleLineDetection",
        accelerator="gpu",
        max_epochs=3000,
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
    parser.add_argument("--from_scratch", action="store_true")

    args = parser.parse_args()
    main(args)
