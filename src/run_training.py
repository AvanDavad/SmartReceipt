from pytorch_lightning import Trainer
from src.pl_module import CNNModule
from src.reader import ImageDataset, ImageReader
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")

def main():
    train_reader = ImageReader(PROJ_DIR / "data" / "train")
    val_reader = ImageReader(PROJ_DIR / "data" / "val")

    train_dataset = ImageDataset(train_reader)
    val_dataset = ImageDataset(val_reader)

    train_dataset[0]

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    version_num = 4
    ckpt_name = "resnet-epoch=80-val_loss=0.282.ckpt"
    model = CNNModule().load_from_checkpoint(PROJ_DIR / "lightning_logs" / f"version_{version_num}" / "checkpoints" / ckpt_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="resnet-{epoch:02d}-{val_loss:.3f}",
        save_top_k=2,
        mode="min",
    )

    trainer = Trainer(accelerator="gpu", max_epochs=100, callbacks=[checkpoint_callback])
    # trainer.fit(model, train_dataloader, val_dataloader)
    # trainer.validate(model, val_dataloader)

    # inference
    img_path = PROJ_DIR / "data" / "val" / "IMG_5488.jpg"
    model.inference(img_path, val_dataset.transforms)

if __name__ == "__main__":
    main()