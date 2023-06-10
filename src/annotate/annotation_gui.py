from pathlib import Path
from src.annotate.phase_handlers import PHASE_1_NUM_KEYPOINTS, Phase0Handler, Phase1Handler, Phase2Handler, Phase3Handler
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import json

import tkinter as tk
from tkinter import filedialog, ttk

CANVAS_W = 800
CANVAS_H = 600

class AnnotationGUI:

    def __init__(self, root, args):
        self.args = args
        self.phase_handlers = []

        self._init_root(root)
        self._init_main_frame()
        self._init_textbox()
        self._init_buttons()
        self._init_canvas()

        self.choose_image_and_load()

    def _init_root(self, root):
        self.root = root
        self.root.bind("<Up>", self.move_up)
        self.root.bind("<Down>", self.move_down)
        self.root.bind("<Left>", self.move_left)
        self.root.bind("<Right>", self.move_right)

    def _init_main_frame(self):
        self.main_frame = ttk.Frame(self.root, padding=50)
        self.main_frame.pack()

    def _init_textbox(self):
        self.textbox = tk.Text(self.main_frame, width=120, height=10)
        self.textbox.configure(state="disabled")  # Initially set to non-editable
        self.textbox.grid(row=1, column=1)
        self.textbox.bind("<KeyRelease>", self.on_text_update)

    def _init_buttons(self):
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0)

        self.next_button = ttk.Button(
            self.button_frame, text="Next", command=self.next_button_pressed
        )
        self.next_button.grid(row=0, column=0)

        self.save_button = ttk.Button(self.button_frame, text="Save", command=self.save)
        self.save_button.grid(row=1, column=0)

    def _init_canvas(self):
        self.canvas_w = CANVAS_W
        self.canvas_h = CANVAS_H

        self.canvas = tk.Canvas(
            self.main_frame, bg="black", width=self.canvas_w, height=self.canvas_h
        )
        self.canvas.grid(row=0, column=1)

        self.canvas.bind("<Button-4>", self.zoom_image)
        self.canvas.bind("<Button-5>", self.zoom_image)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        self._mouse_pos_canvas = np.array([self.canvas_w/2, self.canvas_h/2])

    def choose_image_and_load(self):
        self.base_img_name = self.choose_image_file()
        if Path(self.base_img_name).is_file():
            self.base_img = Image.open(self.base_img_name)
        else:
            print("No file selected")
            sys.exit()

        annot_path = self.get_annot_path_from_img_name()
        if annot_path.is_file():
            self.process_annot_file(annot_path)
        else:
            self.to_phase_0()

        self.create_image_on_canvas()

    def get_annot_path_from_img_name(self):
        return Path(self.base_img_name).with_suffix(".json")

    def process_annot_file(self, annot_path):
        with open(annot_path, "r") as file:
            annot_dict = json.load(file)
        if annot_dict["phase_idx"] >= 0:
            self.to_phase_0(base_points=annot_dict["base_points"])
        if annot_dict["phase_idx"] >= 1:
            dest_size_wh = (annot_dict["ortho_width"], annot_dict["ortho_height"])
            self.to_phase_1(lines_y=annot_dict["lines_y"], M=annot_dict["M"], dest_size_wh=dest_size_wh)
        if annot_dict["phase_idx"] >= 2:
            self.to_phase_2(lines_x=annot_dict["lines_x"])
        if annot_dict["phase_idx"] >= 3:
            self.to_phase_3(text=annot_dict["text"])

    def choose_image_file(self):
        filename = filedialog.askopenfilename(
            initialdir=self.args.rootdir,
            title="Select image to annotate",
            filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")),
        )
        return filename

    @property
    def mouse_pos_canvas(self):
        return self._mouse_pos_canvas

    @mouse_pos_canvas.setter
    def mouse_pos_canvas(self, value):
        assert isinstance(value, np.ndarray)
        assert value.shape == (2,)
        self._mouse_pos_canvas = value

    @property
    def base_points(self):
        if self.phase_idx >= 0:
            return self.phase_handlers[0].base_points
        return []

    @property
    def phase_idx(self):
        return len(self.phase_handlers) - 1

    @property
    def zoom_handler(self):
        return self.phase_handler.zoom_handler

    @property
    def lines_y(self):
        if self.phase_idx >= 1:
            return self.phase_handlers[1].lines_y
        return []

    @property
    def M(self):
        if self.phase_idx >= 1:
            return self.phase_handlers[1].M
        return None

    @property
    def ortho_img(self):
        if self.phase_idx >= 1:
            return self.phase_handlers[1].ortho_img
        return None

    @property
    def lines_x(self):
        if self.phase_idx >= 2:
            return self.phase_handlers[2].lines_x
        return []

    @property
    def text(self):
        if self.phase_idx >= 3:
            return self.phase_handlers[3].text
        return ""

    def next_button_pressed(self):
        if self.phase_idx == 0:
            if len(self.base_points) == PHASE_1_NUM_KEYPOINTS:
                self.next_phase()
        elif self.phase_idx == 1:
            if len(self.lines_y) >= 2:
                self.next_phase()
        elif self.phase_idx == 2:
            self.next_phase()
        self.create_image_on_canvas()

    def on_left_click(self, event):
        mouse_pos_canvas = np.array([event.x, event.y])
        mouse_pos_img = self.zoom_handler.transform_canvas2img(mouse_pos_canvas)

        next_phase = self.phase_handler.on_left_click(mouse_pos_img, mouse_pos_canvas)
        if next_phase:
            self.next_phase()

        self.create_image_on_canvas()

    def on_right_click(self, event):
        self.phase_handler.on_right_click(event)
        self.create_image_on_canvas()

    def on_mouse_move(self, event):
        self.mouse_pos_canvas = np.array([event.x, event.y])
        if self.phase_idx == 2:
            self.create_image_on_canvas()

    def move_up(self, event):
        if self.phase_idx in (0, 1):
            self.zoom_handler.move_up()
            self.create_image_on_canvas()

    def move_down(self, event):
        if self.phase_idx in (0, 1):
            self.zoom_handler.move_down()
            self.create_image_on_canvas()

    def move_left(self, event):
        if self.phase_idx in (0, 1):
            self.zoom_handler.move_left()
            self.create_image_on_canvas()

    def move_right(self, event):
        if self.phase_idx in (0, 1):
            self.zoom_handler.move_right()
            self.create_image_on_canvas()

    def zoom_image(self, event):
        assert event.delta == 0, f"event.delta: {event.delta}"
        if event.num == 5:
            scroll_up = False  # wheel scrolled down
        elif event.num == 4:
            scroll_up = True  # wheel scrolled up

        # get the current coordinates of the mouse
        mouse_pos_canvas = np.array([event.x, event.y])
        self.zoom_handler.zoom_in(
            mouse_pos_canvas
        ) if scroll_up else self.zoom_handler.zoom_out(mouse_pos_canvas)

        self.create_image_on_canvas()

    def on_text_update(self, event):
        self.phase_handler.on_text_update(event)
        self.create_image_on_canvas()

    def create_image_on_canvas(self):
        img = self.zoom_handler.get_img()
        draw = ImageDraw.Draw(img)

        self.phase_handler.create_image_on_canvas(draw, self.mouse_pos_canvas)

        self.photoimage = ImageTk.PhotoImage(img)
        self.img_id = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.photoimage
        )

    def next_phase(self):
        if self.phase_idx == 0:
            self.to_phase_1()
        elif self.phase_idx == 1:
            self.to_phase_2()
        elif self.phase_idx == 2:
            self.to_phase_3()
        print(f"phase: {self.phase_idx}")

    def to_phase_0(self, base_points=[]):
        assert self.phase_idx == -1, f"phase_idx is {self.phase_idx}, expected -1"
        self.phase_handler = Phase0Handler(
            self.base_img,
            self.canvas_w,
            self.canvas_h,
            base_points=base_points,
        )
        self.phase_handlers = [self.phase_handler]

    def to_phase_1(self, lines_y=[], M=None, dest_size_wh=None):
        assert self.phase_idx == 0, f"phase_idx is {self.phase_idx}, expected 0"
        self.phase_handler = Phase1Handler(self.phase_handler, lines_y=lines_y, M=M, dest_size_wh=dest_size_wh)
        self.phase_handlers.append(self.phase_handler)

    def to_phase_2(self, lines_x=[0]):
        assert self.phase_idx == 1, f"phase_idx is {self.phase_idx}, expected 1"
        self.phase_handler = Phase2Handler(self.phase_handler, lines_x=lines_x)
        self.phase_handlers.append(self.phase_handler)

    def to_phase_3(self, text=""):
        assert self.phase_idx == 2, f"phase_idx is {self.phase_idx}, expected 2"
        self.phase_handler = Phase3Handler(self.phase_handler, text=text, textbox=self.textbox)
        self.phase_handlers.append(self.phase_handler)

    def save(self):
        annot_dict = {}
        if self.phase_idx >= 0:
            annot_dict["base_points"] = self.base_points
        if self.phase_idx >= 1:
            annot_dict["lines_y"] = self.lines_y
            annot_dict["M"] = self.M.tolist()
            annot_dict["ortho_width"] = self.ortho_img.width
            annot_dict["ortho_height"] = self.ortho_img.height
        if self.phase_idx >= 2:
            annot_dict["lines_x"] = self.lines_x
        if self.phase_idx >= 3:
            annot_dict["text"] = self.text

        annot_dict["phase_idx"] = self.phase_idx

        annot_path = self.get_annot_path_from_img_name()
        with open(annot_path, "w") as file:
            json.dump(annot_dict, file)
        print(f"Saved annotation to {annot_path}")
        sys.exit()

    def on_close(self):
        self.save()
        self.root.destroy()
