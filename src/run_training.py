from src.pl_module import CNNModule
from src.reader import ImageDataset, ImageReader
from torch.utils.data import DataLoader
from pathlib import Path

PROJ_DIR = Path("/home/avandavad/projects/receipt_extractor")

def main():
    train_reader = ImageReader(PROJ_DIR / "data" / "train")
    val_reader = ImageReader(PROJ_DIR / "data" / "val")

    train_dataset = ImageDataset(train_reader)
    val_dataset = ImageDataset(val_reader)

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    model = CNNModule().load_from_checkpoint(PROJ_DIR / "lightning_logs" / "version_1" / "checkpoints" / "epoch=2-step=6.ckpt")

    # trainer = Trainer(max_epochs=3)
    # trainer.fit(model, train_dataloader, val_dataloader)

    # inference
    img_path = PROJ_DIR / "data" / "val" / "IMG_5488.jpg"
    model.inference(img_path, val_dataset.transforms)

if __name__ == "__main__":
    main()