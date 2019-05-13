# -*- coding: utf-8 -*-
from fragil.Ching_Sheng_method_2019 import Ching_Sheng2019
from helpers.stego_metrics import Metrics
from tkinter import filedialog
from tkinter import *
import numpy as np


from PIL import Image


def main():
    # Directory
    dir = "D:/DataSets"
    # Instance
    wm = Ching_Sheng2019()

    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir=dir + "/", title="Select file",
            filetypes=(
                ("bmp files", "*.bmp"),
                ("jpg files", "*.jpg"), ("png files", "*.png"),
                ("all files", "*.*")))
    except Exception as e:
        print("Error: ", e)
        print("The image file was not loaded")

    root.destroy()

    # Open image
    cover_image = Image.open(root.filename)
    # Instances
    watermarked_image = wm.insert(cover_image)
    metr = Metrics()
    # Save watermarked image
    dir_water_im = "watermarked_" + root.filename.split("/")[-1][:-4]  + ".bmp"
    watermarked_image.save("static/" + dir_water_im)

    # Show metrics
    cover = np.asarray(cover_image)
    watermark = np.asarray(watermarked_image)
    print(" ")
    print("Experimental analysis")
    print(" ")
    print("PSNR: ", metr.psnr(cover, watermark))
    print(" ")
    print("UIQI: ", metr.uiqi(cover, watermark))
    print(" ")
    print("IF: ", metr.image_fid(cover, watermark))
    print(" ")
    print("HS: ", metr.hist_sim(cover, watermark))


if __name__ == '__main__':
    main()
