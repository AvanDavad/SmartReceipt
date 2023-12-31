import argparse
from pathlib import Path
from src.readers.image_reader import ImageReader

from src.line_dataset import LineDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default="/home/avandavad/projects/receipt_extractor/data/train",
    )
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--out_folder", type=str, default="visualization/line_dataset"
    )

    args = parser.parse_args()

    dataset = LineDataset(
        ImageReader(args.rootdir),
        augment=args.augment,
        shuffle=args.shuffle,
    )

    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    for idx in range(len(dataset)):
        for repeat_idx in range(args.repeat):
            dataset.show(idx, out_folder, repeat_idx=repeat_idx)
