from pytorch_lightning import Trainer
from src.models import CNNModule6Points
from src.reader import ImageDataset, ImageReader
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import time

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")


def main():
    train_reader = ImageReader(PROJ_DIR / "data" / "train")
    val_reader = ImageReader(PROJ_DIR / "data" / "val")

    train_dataset = ImageDataset(train_reader, augment=True)
    val_dataset = ImageDataset(val_reader, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="resnet-{epoch:02d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=25000,
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
    main()
