import abc

import numpy as np
from PIL import Image

from src.annotate.zoom_handler import ImageZoomHandler
from src.camera_calib import get_camera_calib
from src.visualization.font import get_font
from src.warp_perspective import warp_perspective
from src.warp_perspective import warp_perspective_with_nonlin_least_squares


PHASE_1_NUM_KEYPOINTS = 4
FONT = get_font(font_size=20)


BASEPOINT_CORNER_INDICES = {
    "topleft": 0,
    "topright": 1,
    "bottomleft": 2,
    "bottomright": 3,
    "top": [0, 1],
    "bottom": [2, 3],
    "left": [0, 2],
    "right": [1, 3],
}


def draw_lines_x(draw, lines_x, zoom_handler, img_height):
    for x in lines_x:
        pt0 = zoom_handler.transform_img2canvas([x, 0])
        pt1 = zoom_handler.transform_img2canvas([x, img_height])
        draw.line((pt0[0], pt0[1], pt1[0], pt1[1]), fill="blue", width=1)


def get_composite_img_lines(ortho_img, lines_y, with_underline=False):
    composite_img = np.zeros(
        (
            np.max(np.diff(lines_y).astype(np.int64)) + 1,
            ortho_img.width * (len(lines_y) - 1),
            3,
        ),
        dtype=np.uint8,
    )
    for line_idx in range(1, len(lines_y)):
        line_y0 = int(lines_y[line_idx - 1])
        line_y1 = int(lines_y[line_idx])
        img = np.array(ortho_img)[line_y0:line_y1, :, :]
        composite_img[
            : img.shape[0],
            (line_idx - 1) * ortho_img.width : line_idx * ortho_img.width,
            :,
        ] = img

    if with_underline:
        underline = np.ones_like(composite_img) * 255
        composite_img = np.concatenate([composite_img, underline], axis=0)

    composite_img = Image.fromarray(composite_img)
    return composite_img


class PhaseHandler(abc.ABC):
    @abc.abstractmethod
    def create_image_on_canvas(self, draw, mouse_pos_canvas):
        pass

    @abc.abstractmethod
    def on_left_click(self, mouse_pos_img, mouse_pos_canvas):
        pass

    @abc.abstractmethod
    def on_right_click(self, event):
        pass

    @abc.abstractmethod
    def on_text_update(self, event):
        pass


class Phase0Handler(PhaseHandler):
    def __init__(self, img, canvas_w, canvas_h, base_points=[]):
        self.base_points = base_points
        self.img = img
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        self.zoom_handler = ImageZoomHandler(
            img,
            canvas_w,
            canvas_h,
            scale=img.width / canvas_w,
        )

    def create_image_on_canvas(self, draw, mouse_pos_canvas):
        for base_point in self.base_points:
            p = self.zoom_handler.transform_img2canvas(base_point).astype(
                np.int64
            )
            draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill="red")
        # draw lines
        for p0, p1 in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            if len(self.base_points) < p1 + 1:
                break
            line_pt0 = self.zoom_handler.transform_img2canvas(
                self.base_points[p0]
            ).astype(int)
            line_pt1 = self.zoom_handler.transform_img2canvas(
                self.base_points[p1]
            ).astype(int)
            draw.line(
                (line_pt0[0], line_pt0[1], line_pt1[0], line_pt1[1]),
                fill="blue",
                width=1,
            )

    def on_left_click(self, mouse_pos_img, mouse_pos_canvas):
        next_phase = False
        if len(self.base_points) < PHASE_1_NUM_KEYPOINTS:
            self.base_points.append(mouse_pos_img.tolist())
        else:
            next_phase = True
        return next_phase

    def on_right_click(self, event):
        if self.base_points:
            self.base_points.pop()

    def on_text_update(self, event):
        pass


class Phase1Handler(PhaseHandler):
    def __init__(self, prev_handler, lines_y=[], M=None, dest_size_wh=None):
        self.prev_handler = prev_handler
        self.ortho_img, self.M = self._init_ortho(M, dest_size_wh)

        self.zoom_handler = ImageZoomHandler(
            self.ortho_img, self.canvas_w, self.canvas_h
        )
        self.lines_y = lines_y

        self.base_points_ortho = Phase1Handler.base_points_to_ortho(
            self.base_points, self.M
        )

    def _init_ortho(self, M, dest_size_wh):
        if M is None:
            camera_matrix, dist_coeffs = get_camera_calib()
            ortho_img, _, M = warp_perspective_with_nonlin_least_squares(
                self.base_img,
                np.array(self.base_points),
                camera_matrix,
                dist_coeffs,
            )
        else:
            assert dest_size_wh is not None
            M = np.array(M)
            ortho_img = warp_perspective(self.base_img, M, dest_size_wh)
        return ortho_img, M

    @property
    def base_img(self):
        return self.prev_handler.img

    @property
    def base_points(self):
        return self.prev_handler.base_points

    @property
    def canvas_w(self):
        return self.prev_handler.canvas_w

    @property
    def canvas_h(self):
        return self.prev_handler.canvas_h

    def create_image_on_canvas(self, draw, mouse_pos_canvas):
        for y in self.lines_y:
            pt0 = self.zoom_handler.transform_img2canvas([0, y])
            pt1 = self.zoom_handler.transform_img2canvas(
                [self.ortho_img.width, y]
            )
            draw.line((pt0[0], pt0[1], pt1[0], pt1[1]), fill="blue", width=1)

    @staticmethod
    def base_points_to_ortho(base_points, M):
        base_points = np.array(base_points).copy()
        base_points_hom = np.concatenate(
            [base_points, np.ones((len(base_points), 1))], axis=1
        )
        base_points_hom = base_points_hom @ M.T
        base_points = base_points_hom[:, :2] / base_points_hom[:, 2:]

        left = BASEPOINT_CORNER_INDICES["left"]
        base_points[left, 0] = np.mean(base_points[left, 0])

        right = BASEPOINT_CORNER_INDICES["right"]
        base_points[right, 0] = np.mean(base_points[right, 0])

        top = BASEPOINT_CORNER_INDICES["top"]
        base_points[top, 1] = np.mean(base_points[top, 1])

        bottom = BASEPOINT_CORNER_INDICES["bottom"]
        base_points[bottom, 1] = np.mean(base_points[bottom, 1])

        base_points_ortho = base_points
        return base_points_ortho

    def on_left_click(self, mouse_pos_img, mouse_pos_canvas):
        curr_y = mouse_pos_img[1]
        if len(self.lines_y) < 2:
            self.lines_y.append(curr_y)
        else:
            line_height = self.lines_y[-1] - self.lines_y[-2]
            num_lines = int(np.round((curr_y - self.lines_y[-1]) / line_height))
            additional_lines = np.linspace(
                self.lines_y[-1], curr_y, num_lines + 1
            )[1:]
            self.lines_y.extend(additional_lines.tolist())
        return False

    def on_right_click(self, event):
        if self.lines_y:
            self.lines_y = self.lines_y[:-1]

    def on_text_update(self, event):
        pass


class Phase2Handler(PhaseHandler):
    def __init__(self, prev_handler, lines_x=[0]):
        self.prev_handler = prev_handler

        composite_img = get_composite_img_lines(
            self.ortho_img, self.lines_y, with_underline=False
        )
        self.img = composite_img

        self.zoom_handler = ImageZoomHandler(
            self.img, self.canvas_w, self.canvas_h, scale=0.3
        )
        self.lines_x = lines_x

        x_offset = self.get_x_offset()
        self.zoom_handler.move_img_point_to_canvas_point(
            np.array([x_offset, self.img.height // 2])
        )

    @property
    def ortho_img(self):
        return self.prev_handler.ortho_img

    @property
    def lines_y(self):
        return self.prev_handler.lines_y

    @property
    def canvas_w(self):
        return self.prev_handler.canvas_w

    @property
    def canvas_h(self):
        return self.prev_handler.canvas_h

    def get_x_offset(self):
        if len(self.lines_x) < 2:
            x_offset = 10
        else:
            x_offset = self.lines_x[-1] + (self.lines_x[-1] - self.lines_x[-2])
        return x_offset

    def create_image_on_canvas(self, draw, mouse_pos_canvas):
        draw_lines_x(draw, self.lines_x, self.zoom_handler, self.img.height)
        mouse_pos_img = self.zoom_handler.transform_canvas2img(mouse_pos_canvas)
        pt0 = self.zoom_handler.transform_img2canvas([mouse_pos_img[0], 0])
        pt1 = self.zoom_handler.transform_img2canvas(
            [mouse_pos_img[0], self.img.height]
        )
        draw.line((pt0[0], pt0[1], pt1[0], pt1[1]), fill="yellow", width=1)

    def on_left_click(self, mouse_pos_img, mouse_pos_canvas):
        curr_x = mouse_pos_img[0]
        self.lines_x.append(curr_x)

        x_offset = self.get_x_offset()
        self.zoom_handler.move_img_point_to_canvas_point(
            np.array([x_offset, self.img.height // 2]),
            np.array([mouse_pos_canvas[0], self.canvas_h / 2]),
        )
        return False

    def on_right_click(self, event):
        if self.lines_x:
            self.lines_x = self.lines_x[:-1]

            x_offset = self.get_x_offset()
            self.zoom_handler.move_img_point_to_canvas_point(
                np.array([x_offset, self.img.height // 2]),
                np.array([event.x, self.canvas_h / 2]),
            )

    def on_text_update(self, event):
        pass


class Phase3Handler(PhaseHandler):
    def __init__(self, prev_handler, text, textbox):
        self.prev_handler = prev_handler

        self.textbox = textbox
        self.textbox.configure(state="normal")
        self.textbox.insert("1.0", text)

        self.text = text

        composite_img = get_composite_img_lines(
            self.ortho_img, self.lines_y, with_underline=True
        )
        self.img = composite_img

        self.zoom_handler = ImageZoomHandler(
            self.img, self.canvas_w, self.canvas_h
        )
        x_offset = self.get_x_offset()
        self.zoom_handler.move_img_point_to_canvas_point(
            np.array([x_offset, self.img.height // 2])
        )

    @property
    def ortho_img(self):
        return self.prev_handler.ortho_img

    @property
    def lines_y(self):
        return self.prev_handler.lines_y

    @property
    def canvas_w(self):
        return self.prev_handler.canvas_w

    @property
    def canvas_h(self):
        return self.prev_handler.canvas_h

    @property
    def lines_x(self):
        return self.prev_handler.lines_x

    def get_x_offset(self):
        if len(self.text) < len(self.lines_x) - 1:
            x_offset = 0.5 * (
                self.lines_x[len(self.text)] + self.lines_x[len(self.text) + 1]
            )
        else:
            x_offset = self.lines_x[-1]
        return x_offset

    def draw_rect(self, i0, i1, draw, fill):
        topleft = self.zoom_handler.transform_img2canvas(
            np.array([i0, self.img.height // 2])
        )
        bottomright = self.zoom_handler.transform_img2canvas(
            np.array([self.lines_x[i1], self.img.height])
        )
        points = topleft.tolist() + bottomright.tolist()
        draw.rectangle(points, fill=fill)

    def draw_text(self, draw):
        for x, ch in zip(self.lines_x, self.text):
            pt = self.zoom_handler.transform_img2canvas(
                [x, self.img.height // 2]
            )
            if pt[0] > -10:
                draw.text((pt[0], pt[1]), ch, fill="black", font=FONT)

    def create_image_on_canvas(self, draw, mouse_pos_canvas):
        idx_green = min(len(self.text), len(self.lines_x) - 1)
        idx_yellow = min(idx_green + 1, len(self.lines_x) - 1)

        self.draw_rect(0, idx_green, draw, "lightgreen")
        self.draw_rect(self.lines_x[idx_green], idx_yellow, draw, "yellow")

        draw_lines_x(draw, self.lines_x, self.zoom_handler, self.img.height)
        self.draw_text(draw)

    def on_text_update(self, event):
        self.text = self.textbox.get("1.0", "end").rstrip("\n")
        x_offset = self.get_x_offset()
        self.zoom_handler.move_img_point_to_canvas_point(
            np.array([x_offset, self.img.height // 2])
        )

    def on_left_click(self, mouse_pos_img, mouse_pos_canvas):
        return False

    def on_right_click(self, event):
        pass
