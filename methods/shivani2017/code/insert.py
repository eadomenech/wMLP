# -*- coding: utf-8 -*-
from Shivani2017R import Shivani2017R
from tkinter import filedialog
from tkinter import *
import numpy as np

from evaluations.evaluations import Evaluations


from PIL import Image


def main():
    eva = Evaluations()
    # Directory
    dir = "D:/DataSets"
    # Instance
    wm = Shivani2017R()

    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir=dir + "/", title="Select file",
            filetypes=(
                ("jpg files", "*.jpg"), ("png files", "*.png"),
                ("all files", "*.*")))
    except Exception as e:
        print("Error: ", e)
        print("The image file was not loaded")

    root.destroy()

    path = root.filename
    
    cover_image = Image.open(path)
    
    # Insert
    watermarked_image = wm.insert(cover_image)
    # Save watermarked image
    dir_water_im = "watermarked_" + path.split("/")[-1][:-4]  + ".bmp"
    watermarked_image.save("static/" + dir_water_im)

    # Show PSNR
    cover = Image.open(path)
    print(" ")
    print("PSNR: ", eva.PSNR_RGB(cover, watermarked_image))

    # Watermark extracting
    watermark_extracted = wm.extract(watermarked_image)
    # Save watermark image
    dir_water_im = "watermark_" + path.split("/")[-1][:-4]  + ".bmp"
    watermark_extracted.save("static/" + dir_water_im)


if __name__ == '__main__':
    main()
