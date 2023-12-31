import argparse
from pathlib import Path

from PIL import Image

from src.draw_utils import draw_borders_with_chars_and_probs
from src.models.phase2_single_char_model import CNNModulePhase2SingleChar
from src.models.phase2char_border_model import CNNModulePhase2CharsBorder
from src.run.run_all import get_best_ckpt_path

PROJ_DIR = Path(__file__).parents[2]


def main(args):
    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase2CharsBorder"
    )
    print(f"Loading char border model from {ckpt_path}")
    border_model = CNNModulePhase2CharsBorder().load_from_checkpoint(ckpt_path)

    ckpt_path = get_best_ckpt_path(
        PROJ_DIR / "model_checkpoints" / "CNNModulePhase2SingleChar",
        version=args.version,
    )
    print(f"Loading single char model from {ckpt_path}")
    model = CNNModulePhase2SingleChar().load_from_checkpoint(ckpt_path)

    out_folder = PROJ_DIR / "inference" / "phase2_single_char_model"
    out_folder.mkdir(exist_ok=True, parents=True)

    input_image = Image.open("line_image.jpg")
    input_image = input_image.crop(
        (
            0,
            int(input_image.height * args.top),
            input_image.width,
            int(input_image.height * args.bottom),
        )
    )

    x_coords = border_model.inference(input_image)
    pred_chars = []
    probs = []
    for x0, x1 in zip(x_coords[:-1], x_coords[1:]):
        char_image = input_image.crop((x0, 0, x1, input_image.height))
        predicted_char, prob = model.inference(char_image)
        pred_chars.append(predicted_char)
        probs.append(prob)
    pred_chars = "".join(pred_chars)

    img = draw_borders_with_chars_and_probs(
        input_image, x_coords[1:], pred_chars, probs
    )

    filename = str(out_folder / "inference.jpg")
    img.save(filename)
    print(f"Saved {filename}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=int, default=-1)
    argparser.add_argument("--top", type=float, default=0.0)
    argparser.add_argument("--bottom", type=float, default=1.0)

    args = argparser.parse_args()
    main(args)
