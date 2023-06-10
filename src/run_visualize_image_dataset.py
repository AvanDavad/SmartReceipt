import argparse
from pathlib import Path
from src.image_dataset import ImageDataset
from src.image_reader import ImageReader

from src.reader import LineDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/home/avandavad/projects/receipt_extractor/data/train")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    dataset = ImageDataset(
        ImageReader(args.rootdir),
        augment=args.augment,
    )

    out_folder = Path("visualization/image_dataset")
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    for idx in range(len(dataset)):
        for repeat_idx in range(args.repeat):
            dataset.show(idx, out_folder, repeat_idx)