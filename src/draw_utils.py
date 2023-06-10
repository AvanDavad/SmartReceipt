from PIL import Image, ImageDraw
import numpy as np
import cv2
import pytesseract
from pytesseract import Output

def save_img_with_kps(
    img: Image, kps, filename, normalized=False, circle_radius=5, circle_color="blue"
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
