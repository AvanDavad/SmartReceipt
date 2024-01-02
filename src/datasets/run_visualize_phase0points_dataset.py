"""
python3 -m src.run.run_visualize_phase0points_dataset --augment --repeat 3
"""
import argparse
from pathlib import Path

from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.readers.image_reader import ImageReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default="data",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--color_augment_prob", type=float, default=0.5)
    parser.add_argument("--rotate_augment_prob", type=float, default=0.5)
    parser.add_argument("--perspective_augment_prob", type=float, default=0.5)
    parser.add_argument("--crop_augment_prob", type=float, default=0.5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--out_folder", type=str, default="visualization/image_dataset"
    )
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    rootdir = Path(args.rootdir) / args.split
    dataset = Phase0PointsDataset(
        ImageReader(rootdir),
        augment=args.augment,
        color_augment_prob=args.color_augment_prob,
        rotate_augment_prob=args.rotate_augment_prob,
        perspective_augment_prob=args.perspective_augment_prob,
        crop_augment_prob=args.crop_augment_prob,
    )

    out_folder = Path(args.out_folder) / args.split
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in out_folder.iterdir():
        file.unlink()

    sample_idx = 0
    for idx in range(len(dataset)):
        for repeat_idx in range(args.repeat):
            if sample_idx >= args.max_samples:
                break

            img = dataset.show(idx)

            filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"
            img.save(filename)
            print(f"saved {filename}")

            sample_idx += 1
