# -*- coding: utf-8 -*-
from AvilaDomenech2018R import AvilaDomenech2018R

from tkinter import filedialog
from tkinter import *

from PIL import Image
import numpy as np

from evaluations.evaluations import Evaluations


def main():
    # AvilaDomenech2018R Instances
    wm = AvilaDomenech2018R('password')

    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir="static/", title="Select file",
            filetypes=(
                ("png files", "*.jpg"), ("jpg files", "*.png"),
                ("all files", "*.*")))
        cover_image = Image.open(root.filename).convert('RGB')
        root.destroy()
        watermarked_image = wm.insert(cover_image)
        watermarked_image.save("static/watermarked_image.png")

        # PSNR
        cover_image = Image.open(root.filename)
        stego_image = Image.open("static/watermarked_image.png").convert('RGB')
        print("Calculando PSNR")
        print(Evaluations().PSNR_RGB(cover_image, stego_image))

        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image)
        # Save watermark image
        dir_water_im = "watermark_" + root.filename.split("/")[-1][:-4]  + ".bmp"
        watermark_extracted.save("static/" + dir_water_im)
    except Exception as e:
        root.destroy()
        print("Error: ", e)
        print("The image file was not loaded")


if __name__ == '__main__':
    main()
