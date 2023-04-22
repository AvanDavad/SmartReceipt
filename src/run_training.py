from pytorch_lightning import Trainer
from src.pl_module import CNNModule
from src.reader import ImageDataset, ImageReader
from torch.utils.data import DataLoader

def main():
    train_reader = ImageReader("/home/avandavad/projects/receipt_extractor/data/train")
    val_reader = ImageReader("/home/avandavad/projects/receipt_extractor/data/val")

    train_dataset = ImageDataset(train_reader)
    val_dataset = ImageDataset(val_reader)

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    model = CNNModule()

    trainer = Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()