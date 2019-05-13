# -*- coding: utf-8 -*-
from fragil.Ching_Sheng_method_2019 import Ching_Sheng2019
from helpers.stego_metrics import Metrics

from tkinter import filedialog
from tkinter import *
import numpy as np


from PIL import Image


def main():
    # Instance
    wm = Ching_Sheng2019()

    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir="static/", title="Select file",
            filetypes=(
                ("bmp files", "*.bmp"),
                ("jpg files", "*.jpg"), ("png files", "*.png"),
                ("all files", "*.*")))
    except Exception as e:
        print("Error: ", e)
        print("The image file was not loaded")

    root.destroy()

    # Open image
    watermarked_image = Image.open(root.filename)
    # Instances

    answer = wm.extract(watermarked_image)

    print("\n")
    if answer == True:
        print("\nThe watermarked image is authentic\n")
    else:
        print("The watermarked image is not authentic")


if __name__ == '__main__':
    main()
