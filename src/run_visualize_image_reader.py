import argparse
from src.image_reader import ImageReader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/home/avandavad/projects/receipt_extractor/data/train")
    args = parser.parse_args()

    reader = ImageReader(args.rootdir)
    for idx in range(len(reader)):
        reader.show(idx)