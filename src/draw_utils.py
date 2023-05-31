from PIL import Image, ImageDraw


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