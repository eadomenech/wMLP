# -*- coding: utf-8 -*-
from fragil.method_Avila2019 import Avila2019

from tkinter import filedialog
from tkinter import *

from PIL import Image
import numpy as np

from helpers.evaluations import Evaluations


def main():
    # Avila2019 Instances
    wm = Avila2019('password')

    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir="static/", title="Select file",
            filetypes=(
                ("png files", "*.png"), ("jpg files", "*.jpg"),
                ("all files", "*.*")))
        cover_image = Image.open(root.filename).convert('RGB')
        root.destroy()
        watermarked_image = wm.insert(cover_image)
        watermarked_image.save("static/watermarked_image.png")

        # PSNR
        cover_image = Image.open(root.filename)
        stego_image = Image.open("static/watermarked_image.png").convert('RGB')
        print(Evaluations().PSNR_RGB(cover_image, stego_image))

    except Exception as e:
        root.destroy()
        print("Error: ", e)
        print("The image file was not loaded")


if __name__ == '__main__':
    main()
