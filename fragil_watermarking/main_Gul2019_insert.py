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
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir="static/", title="Select file",
            filetypes=(
                ("png files", "*.png"), ("jpg files", "*.jpg"),
                ("all files", "*.*")))
        cover_image = Image.open(root.filename)
        root.destroy()
        watermarked_image = wm.insert(cover_image)
        watermarked_image.save("static/watermarked_image.png")

        # PSNR
        cover_array = np.asarray(Image.open(root.filename))
        red_cover_array = cover_array[:, :, 0]
        watermarked_array = np.asarray(
            Image.open("static/watermarked_image.png"))
        red_watermarked_array = watermarked_array[:, :, 0]

    except Exception as e:
        root.destroy()
        print("Error: ", e)
        print("The image file was not loaded")


if __name__ == '__main__':
    main()
