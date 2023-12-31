from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from PIL import Image
from PIL import ImageDraw

from src.visualization.font import get_font


def inference_and_visualize(
    self,
    img: Image.Image,
    pred_kps: List[Tuple],
    line_width: int = 5,
    circle_radius: int = 10,
    circle_color: str = "blue",
    font_size: int = 25,
    font_color: str = "black",
) -> Union[Image.Image, np.ndarray]:
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for idx, (x, y) in enumerate(pred_kps):
        draw.ellipse(
            (
                x - circle_radius,
                y - circle_radius,
                x + circle_radius,
                y + circle_radius,
            ),
            fill=circle_color,
        )

        # Draw text
        font = get_font(font_size)
        draw.text((x, y), str(idx), fill=font_color, font=font)

    for i0, i1, col in [
        (0, 1, "red"),
        (0, 2, "red"),
        (1, 3, "red"),
        (2, 3, "yellow"),
    ]:
        start_point = pred_kps[i0]
        end_point = pred_kps[i1]
        draw.line((start_point, end_point), fill=col, width=line_width)

    return img, pred_kps
