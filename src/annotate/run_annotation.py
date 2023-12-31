import argparse
import tkinter as tk

from src.annotate.annotation_gui import AnnotationGUI


def main(args):
    root = tk.Tk()
    root.title("Annotation Tool")
    _ = AnnotationGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default="/home/avandavad/projects/receipt_extractor/data",
    )
    args = parser.parse_args()

    main(args)
