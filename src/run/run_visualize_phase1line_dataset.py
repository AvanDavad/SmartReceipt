"""
python3 -m src.run.run_visualize_phase1line_dataset --augment --repeat 3
"""
import argparse
from pathlib import Path

from src.datasets.phase1line_dataset import Phase1LineDataset
from src.readers.image_reader import ImageReader

PROJ_DIR = Path(__file__).parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default=PROJ_DIR / "data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--out_folder", type=str, default="visualization/phase1line_dataset"
    )
    args = parser.parse_args()

    rootdir = Path(args.rootdir) / args.split
    dataset = Phase1LineDataset(
        ImageReader(rootdir),
        augment=args.augment,
    )

    out_folder = Path(args.out_folder) / args.split
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    for idx in range(len(dataset)):
        for repeat_idx in range(args.repeat):
            dataset.show(idx, out_folder, repeat_idx)
