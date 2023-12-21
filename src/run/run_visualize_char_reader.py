import argparse
from pathlib import Path
from src.readers.char_reader import CharReader
from src.readers.image_reader import ImageReader
import time

from src.run.run_visualize_image_reader import remove_dir_with_all_contents

def main(args):
    out_folder: Path = Path(args.out_folder) / args.split
    remove_dir_with_all_contents(out_folder)

    out_folder.mkdir(parents=True, exist_ok=False)

    char_reader = CharReader(Path(args.rootdir) / args.split, w=args.w)

    for idx in range(args.start_idx, args.start_idx + args.max_number_of_frames):
        if idx >= len(char_reader):
            break
        print(f"Processing sample {idx} of {len(char_reader)}")
        char_reader.show(idx, out_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/home/avandavad/projects/receipt_extractor/data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--w", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_number_of_frames", type=int, default=100)
    parser.add_argument("--out_folder", type=str, default="visualization/char_reader")
    args = parser.parse_args()

    main(args)
