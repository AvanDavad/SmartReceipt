from pytorch_lightning import Trainer
from src.pl_module import CNNModule
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

    train_dataset[0]

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    version_num = 15
    ckpt_name = "resnet-epoch=4558-val_loss=0.004.ckpt"
    ckpt_path = PROJ_DIR / "lightning_logs" / f"version_{version_num}" / "checkpoints" / ckpt_name
    model = CNNModule().load_from_checkpoint(ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="resnet-{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=15000,
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
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()