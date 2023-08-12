"""
examples:
python3 -m src.run_visualize_image_reader --split train
python3 -m src.run_visualize_image_reader --split val
"""

import argparse
from pathlib import Path
from src.readers.image_reader import ImageReader
import time

def remove_dir_with_all_contents(dir):
    if dir.is_dir():
        for file in dir.iterdir():
            if file.is_dir():
                remove_dir_with_all_contents(file)
            else:
                file.unlink()
        dir.rmdir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/home/avandavad/projects/receipt_extractor/data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_folder", type=str, default="visualization/image_reader")
    args = parser.parse_args()

    out_folder = Path(args.out_folder) / args.split
    remove_dir_with_all_contents(out_folder)

    out_folder.mkdir(parents=True, exist_ok=False)

    reader = ImageReader(Path(args.rootdir) / args.split)
    print(f"Number of samples: {len(reader)}")
    time.sleep(2)

    for idx in range(len(reader)):
        print(f"Processing {idx}th sample")
        reader.show(idx, out_folder)