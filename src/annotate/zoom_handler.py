

import numpy as np
from PIL import Image

ZOOM_OUT_FACTOR = 1.2
ZOOM_IN_FACTOR = 1.0 / ZOOM_OUT_FACTOR

MOVE_FACTOR = 0.1

class ImageZoomHandler:
    """
    This class handles the zooming and panning of the image.

    image and canvas:
    +===================+
    | image             |
    |                   |
    |   topleft         |
    |       +========+  |
    |       | canvas |  |
    |       |        |  |
    |       +========+  |
    |                   |
    +===================+

    the canvas shows the actual crop of the image depending on the size and
    position of the canvas.

    self.scale: the scale factor of the canvas.
    self.topleft: the top left corner of the canvas on the image
    """
    def __init__(self, img: Image, canvas_w: int, canvas_h: int, scale=1.0):
        self.img = img.copy()

        self.img_w = self.img.width
        self.img_h = self.img.height

        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        self.scale = scale
        self.topleft = np.zeros(2)

    def get_img_crop(self):
        x0 = int(self.topleft[0])
        y0 = int(self.topleft[1])
        cw = int(self.canvas_w * self.scale)
        ch = int(self.canvas_h * self.scale)

        img_np = np.array(self.img)
        img_crop_np = np.array(Image.new("RGB", (cw, ch), color="black"))

        if x0 < self.img_w and 0 <= x0 + cw and y0 < self.img_h and 0 <= y0 + ch:
            h = self.img.height
            w = self.img.width
            i0, i1 = max(0, y0), min(h, y0 + ch)
            j0, j1 = max(0, x0), min(w, x0 + cw)
            ci0, ci1 = max(0, -y0), min(ch, h-y0)
            cj0, cj1 = max(0, -x0), min(cw, w - x0)
            img_crop_np[ci0:ci1, cj0: cj1] = img_np[i0:i1, j0:j1].copy()

        return Image.fromarray(img_crop_np)

    def get_img(self):
        img_crop = self.get_img_crop()
        return img_crop.resize((self.canvas_w, self.canvas_h))

    def move_up(self):
        self.topleft[1] -= self.canvas_h * self.scale * MOVE_FACTOR

    def move_down(self):
        self.topleft[1] += self.canvas_h * self.scale * MOVE_FACTOR

    def move_left(self):
        self.topleft[0] -= self.canvas_w * self.scale * MOVE_FACTOR

    def move_right(self):
        self.topleft[0] += self.canvas_w * self.scale * MOVE_FACTOR

    def zoom_helper(self, mouse_pos, scale_factor):
        mouse_pos_img = self.transform_canvas2img(mouse_pos)
        new_topleft = self.topleft - mouse_pos_img
        new_topleft = new_topleft * scale_factor
        new_topleft = new_topleft + mouse_pos_img
        self.topleft = new_topleft
        self.scale *= scale_factor

    def zoom_in(self, mouse_pos: np.ndarray):
        self.zoom_helper(mouse_pos, ZOOM_IN_FACTOR)

    def zoom_out(self, mouse_pos: np.ndarray):
        self.zoom_helper(mouse_pos, ZOOM_OUT_FACTOR)

    def move_img_point_to_canvas_point(self, img_point, canvas_point="center"):
        if isinstance(canvas_point, str) and canvas_point == "center":
            canvas_point = np.array([self.canvas_w / 2, self.canvas_h / 2])
        else:
            canvas_point = np.array(canvas_point)
        self.topleft = img_point - self.scale * canvas_point

    def transform_img2canvas(self, points):
        points = np.asarray(points)
        return (points - self.topleft) / self.scale

    def transform_canvas2img(self, points):
        points = np.asarray(points)
        return points * self.scale + self.topleft

    def debug(self, mouse_pos):
        print("ZoomHandler debug info")
        print(f"topleft: {self.topleft}")
        print(f"scale: {self.scale}")
        print(f"mouse_pos_norm: {mouse_pos}")
        print(f"mouse_pos_img: {self.topleft + self.scale * mouse_pos}")