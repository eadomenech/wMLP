

from helpers.utils import gaussian_noise, random_list
from helpers.blocks_image_class import BlocksImage3D
from helpers.blocks_class import BlocksImage
from tkinter import filedialog
from tkinter import *
import numpy as np
import imageio
import os


def main():
    # Directory
    dir = "D:/Anier_Matemático/Research/Papers_2019/Watermarking_Paper/Python"
    dir = "%s/src/static/Watermarked_Images" % dir
    # Initial values
    z_block = [16, 24, 32]
    x0, p, n = 0.77, 0.21, 15

    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir=dir + "/", title="Select file",
            filetypes=(
                ("bmp files", "*.bmp"),
                ("jpg files", "*.jpg"), ("png files", "*.png"),
                ("all files", "*.*")))
        cover_array = imageio.imread(root.filename)
    except Exception as e:
        print("Error: ", e)
        print("The image file was not loaded")

    root.destroy()

    # Selected blocks
    for i in range(n):
        # Instance
        j = z_block[i % 3]
        if len(cover_array.shape) == 2:
            blocks_instance = BlocksImage(cover_array, j, j)
            L = random_list(
                x0, p, list(range(blocks_instance.max_num_blocks())))
            blocks_instance.set_block(gaussian_noise(
                blocks_instance.get_block(L[i]), j, j), L[i]
            )
        else:
            blocks_instance = BlocksImage3D(cover_array, j, j)
            L = random_list(
                x0, p, list(range(blocks_instance.max_num_blocks_image_3d())))
            blocks_instance.set_block_image_3d(gaussian_noise(
                blocks_instance.get_block_image_3d(L[i]), j, j), L[i]
            )

    # Save stego image
    dir_stego_im = os.path.dirname(__file__)
    dir_stego_im = os.path.join(
        dir_stego_im,
        "static",
        "Noises",
        root.filename.split("/")[-1][:-4]
    )
    dir_stego_im = (
        "%s_gaussian_noise.%s" % (dir_stego_im, root.filename.split(".")[1])
    )
    # Save watermarked image with noise
    imageio.imsave(dir_stego_im, cover_array)
    dir_stego_im = "%s.eps" % dir_stego_im.split(".")[0]
    imageio.imsave(dir_stego_im, cover_array)

if __name__ == '__main__':
    main()
