import argparse
from pathlib import Path
from src.image_reader import ImageReader
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/home/avandavad/projects/receipt_extractor/data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_folder", type=str, default="visualization/image_reader")
    args = parser.parse_args()

    out_folder = Path(args.out_folder) / args.split
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    reader = ImageReader(Path(args.rootdir) / args.split)
    print(f"Number of samples: {len(reader)}")
    time.sleep(2)

    for idx in range(len(reader)):
        reader.show(idx, out_folder)