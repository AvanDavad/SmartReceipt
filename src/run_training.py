print("training the AlexNet for orientation classification")
from src.data_reader import ImageDataset

def main():
    print("main")
    train_reader = ImageDataset("data", "train")
    img, lab = train_reader[0]
    print(lab)

    val_reader = ImageDataset("data", "val")
    img, lab = val_reader[0]
    print(lab)

if __name__ == "__main__":
    main()