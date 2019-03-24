# -*- coding: utf-8 -*-
from fragil.method_Gul2019 import Gul2019

from tkinter import filedialog
from tkinter import *

from PIL import Image
import numpy as np

from helpers.stego_metrics import Metrics


def main():
    # Gul2019 Instances
    wm = Gul2019()

    try:
        # Load watermarked image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir="static/", title="Select file",
            filetypes=(
                ("png files", "*.png"), ("jpg files", "*.jpg"),
                ("all files", "*.*")))
        watermarked_image = Image.open(root.filename)
        root.destroy()

        # Extract
        wm.extract(watermarked_image)

    except Exception as e:
        root.destroy()
        print("Error: ", e)
        print("The image file was not loaded")


if __name__ == '__main__':
    main()
