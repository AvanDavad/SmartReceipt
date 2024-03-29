import argparse
import json
from pathlib import Path

import numpy as np

from src.readers.char_reader import CharReader
from src.readers.image_reader import ImageReader
from src.run.run_visualize_image_reader import remove_dir_with_all_contents


def main(args):
    out_folder = Path(args.rootdir) / "char_reader"

    reader = CharReader(Path(args.rootdir))
    print(f"Found {len(reader)} samples")

    remove_dir_with_all_contents(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    assert out_folder.is_dir()

    reader = ImageReader(Path(args.rootdir))

    idx = 0
    for i in range(len(reader)):
        sample = reader[i]
        if sample.phase_2_text is None:
            continue

        ortho_img = sample.phase_1_image
        lines_y = sample.phase_1_lines

        for line_idx, (y0, y1) in enumerate(zip(lines_y[:-1], lines_y[1:])):
            line_height = y1 - y0
            line_img = ortho_img.crop(
                (0, y0 - line_height, ortho_img.width, y1 + line_height)
            )
            filename = str(out_folder / f"{str(idx).zfill(5)}.jpg")
            line_img.save(filename)
            print(f"Saved {filename}", end="\r")

            mask1 = sample.phase_2_lines > line_idx * ortho_img.width
            mask2 = sample.phase_2_lines < (line_idx + 1) * ortho_img.width
            mask = mask1 * mask2
            start_idx = np.min(np.where(mask)[0])
            end_idx = np.max(np.where(mask)[0])

            line_coords = (
                sample.phase_2_lines[start_idx : end_idx + 1]
                - line_idx * ortho_img.width
            )
            line_text = sample.phase_2_text[start_idx:end_idx]

            filename = str(out_folder / f"{str(idx).zfill(5)}.json")
            with open(filename, "w") as f:
                json.dump(
                    {
                        "line_coords": line_coords.tolist(),
                        "line_text": line_text,
                    },
                    f,
                    indent=4,
                )

            idx += 1

    print()
    reader = CharReader(Path(args.rootdir))
    print(f"Found {len(reader)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default="data/train",
    )
    args = parser.parse_args()

    main(args)
