from typing import List, Optional, Tuple, Union
from PIL import Image, ImageDraw
from PIL import ImageFont
import numpy as np
import cv2

def put_stuffs_on_img(
    img: Image,
    pts: Optional[np.ndarray]=None,
    pts_normalized: bool=False,
    pts_radius: Union[int, List[int]]=5,
    pts_colors: Union[str, List[str]]="blue",
    lines: Optional[List[Tuple[int]]]=None,
    lines_width: Union[int, List[int]]=5,
    lines_colors: Union[str, List[str]]="red",
):

    img = img.copy()
    draw = ImageDraw.Draw(img)

    if pts is not None:
        n = pts.shape[0]
        assert pts.shape == (n, 2), f"pts.shape is {pts.shape}, expected ({n}, 2)"

        if isinstance(pts_colors, str):
            pts_colors = [pts_colors] * n
        elif isinstance(pts_colors, list):
            assert len(pts_colors) == n, f"len(pts_colors) is {len(pts_colors)}, expected {n}"
        else:
            raise ValueError(f"pts_colors is {pts_colors}, expected str or list")

        if isinstance(pts_radius, int):
            pts_radius = [pts_radius] * n
        elif isinstance(pts_radius, list):
            assert len(pts_radius) == n, f"len(pts_radius) is {len(pts_radius)}, expected {n}"
        else:
            raise ValueError(f"pts_radius is {pts_radius}, expected int or list")

        for i in range(n):
            if pts_normalized:
                kpt = (int(img.width * pts[i, 0]), int(img.height * pts[i, 1]))
            else:
                kpt = (int(pts[i, 0]), int(pts[i, 1]))

            draw.ellipse(
                (
                    kpt[0] - pts_radius[i],
                    kpt[1] - pts_radius[i],
                    kpt[0] + pts_radius[i],
                    kpt[1] + pts_radius[i],
                ),
                fill=pts_colors[i],
            )

    if lines is not None:

        if isinstance(lines_width, int):
            lines_width = [lines_width] * len(lines)
        elif isinstance(lines_width, list):
            assert len(lines_width) == len(lines), f"len(lines_width) is {len(lines_width)}, expected {len(lines)}"
        else:
            raise ValueError(f"lines_width is {lines_width}, expected str or list")

        if isinstance(lines_colors, str):
            lines_colors = [lines_colors] * len(lines)
        elif isinstance(lines_colors, list):
            assert len(lines_colors) == len(lines), f"len(lines_colors) is {len(lines_colors)}, expected {len(lines)}"
        else:
            raise ValueError(f"lines_colors is {lines_colors}, expected str or list")

        for idx in range(len(lines)):
            draw.line(lines[idx], fill=lines_colors[idx], width=lines_width[idx])

    return img

def draw_text_on_image(img: Image.Image, text: str, pos, font_path: str = "arial.ttf", font_size: int = 24, font_color: str = "black"):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=font_color)
    return img

def draw_vertical_line(img: Image.Image, x: int, color: str = "red", width: int = 5):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.line((x, 0, x, img.height), fill=color, width=width)
    return img

def save_img_with_kps(
    img: Image.Image, kps, filename, normalized=False, circle_radius=5, circle_color="blue"
):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    n = kps.shape[0]
    assert kps.shape == (n, 2), f"kps.shape is {kps.shape}, expected ({n}, 2)"

    for i in range(n):
        if normalized:
            kpt = (int(img.width * kps[i, 0]), int(img.height * kps[i, 1]))
        else:
            kpt = (int(kps[i, 0]), int(kps[i, 1]))

        draw.ellipse(
            (
                kpt[0] - circle_radius,
                kpt[1] - circle_radius,
                kpt[0] + circle_radius,
                kpt[1] + circle_radius,
            ),
            fill=circle_color,
        )

    img.save(filename)
    print(f"saved {filename}")

def save_img_with_texts(img, kps, filename, margin=5):

    len1 = np.linalg.norm(kps[0] - kps[1])
    len2 = np.linalg.norm(kps[2] - kps[3])
    avg_len = (len1 + len2) / 2

    height1 = np.linalg.norm(kps[0] - kps[2])
    height2 = np.linalg.norm(kps[1] - kps[3])
    avg_height = (height1 + height2) / 2

    dst = np.array([[0, 0], [avg_len, 0], [0, avg_height], [avg_len, avg_height]])
    dst += np.array([margin, margin])
    M = cv2.getPerspectiveTransform(kps.astype(np.float32), dst.astype(np.float32))

    src_img = np.array(img)
    out = cv2.warpPerspective(src_img, M, (int(avg_len + 2*margin), int(avg_height + 2*margin)))
    img = Image.fromarray(out)

    img.save(filename)
    print(f"saved {filename}")
