"""
python3 -m src.run.run_visualize_phase2char_dataset
"""
import argparse
import sys
from pathlib import Path

from src.datasets.phase2char_dataset import Phase2CharDataset
from src.readers.char_reader import CharReader

PROJ_DIR = Path(__file__).parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default=PROJ_DIR / "data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_num_chars", type=int, default=50)
    parser.add_argument("--w", type=int, default=5)
    parser.add_argument(
        "--out_folder", type=str, default="visualization/phase2char_dataset"
    )
    args = parser.parse_args()

    rootdir: Path = Path(args.rootdir) / args.split
    char_reader = CharReader(rootdir, w=args.w)
    dataset = Phase2CharDataset(
        char_reader,
        augment=args.augment,
        shuffle=args.shuffle,
    )

    out_folder: Path = Path(args.out_folder) / args.split
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    num_chars = 0
    for idx in range(len(dataset)):
        for repeat_idx in range(args.repeat):
            dataset.show(idx, out_folder, repeat_idx, verbose=True)
            num_chars += 1
            if num_chars >= args.max_num_chars:
                sys.exit(0)
