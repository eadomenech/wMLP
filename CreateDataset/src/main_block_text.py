# -*- coding: utf-8 -*-
from helpers.utils import rgb2_black_or_white, sum_block
from helpers.blocks_class import BlocksImage
from helpers import utils, progress_bar
from PIL import Image, ImageEnhance
import numpy as np
import cv2

import os
import glob


def change_contrast(img):
    enhancer = ImageEnhance.Contrast(img)
    enhanced_im = enhancer.enhance(4.0)
    enhanced_im.save('static/binary.png')
    return enhanced_im


def main():
    paths = glob.glob('static/learn/*.png')
    progress_bar.printProgressBar(
        0, len(paths), prefix='Progress:',
        suffix='Complete', length=50)
    for num, filename in enumerate(paths):
        cover_image=change_contrast(Image.open(filename))
        # Initial Values
        blocks_with_text = []
        cad_keys = ""
        # Open image
        cover_image_array = np.asarray(cover_image)
        # Convert a rgb to grayscale (array)
        cover_array = rgb2_black_or_white(cover_image)
        # Instance
        block_instance = BlocksImage(cover_array)
        # print("\nSorting\n")
        # # Initial call to print 0% progress
        # progress_bar.printProgressBar(
        #     0, block_instance.max_num_blocks(), prefix='Progress:',
        #     suffix='Complete', length=50)
        min_pix = 5
        for i in range(block_instance.max_num_blocks()):
            sum_pix = sum_block(block_instance.get_block(i))
            if sum_pix >= min_pix and sum_pix <= (64-min_pix):
                blocks_with_text.append(i)
                cad_keys += "1"
            else:
                cad_keys += "-1"
            # progress_bar.printProgressBar(
            #     i + 1, block_instance.max_num_blocks(),
            #     prefix='Progress:', suffix='Complete', length=50)

        j = 0
        # print("\nDrawing\n")
        # Initial call to print 0% progress
        # progress_bar.printProgressBar(0, len(blocks_with_text), prefix='Progress:',
        #     suffix='Complete', length=50)
        for item in blocks_with_text:
            coord = block_instance.get_coord(item)
            cv2.rectangle(cover_image_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
            cv2.rectangle(cover_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
            j += 1
            # progress_bar.printProgressBar(j + 1, len(blocks_with_text),
            #     prefix='Progress:', suffix='Complete', length=50)

        Image.fromarray(cover_image_array).save("static/file_tampered.png")
        Image.fromarray(cover_array).save("static/file_tampered_bin.png")

        # print("\n")
        dic = dict(zip(cad_keys, list(range(block_instance.max_num_blocks()))))
        # print(" ")
        # print("Dictionary")
        # print(" ")
        # print(dic)

        # print("\n")
        # print("\nSaving blocks\n")
        cover_image = Image.open(filename)
        cover_image_array = np.asarray(cover_image)
        block_instance = BlocksImage(cover_image_array)
        # # Initial call to print 0% progress
        # progress_bar.printProgressBar(
        #     0, block_instance.max_num_blocks(), prefix='Progress:',
        #     suffix='Complete', length=50)
        # Add path if not exist
        list_dir = [
            'static/train/', 'static/valid/']
        for i in list_dir:
            try:
                os.stat(i)
            except:
                os.mkdir(i)
        list_dir = [
            'static/train/text/', 'static/train/not_text/',
            'static/valid/text/', 'static/valid/not_text/']
        for i in list_dir:
            try:
                os.stat(i)
            except:
                os.mkdir(i)
        for i in range(block_instance.max_num_blocks()):
            p = np.random.rand()
            pos = block_instance.get_coord(i)
            if p > 0.1:
                if i in blocks_with_text:
                    Image.fromarray(
                        cover_image_array[pos[0]:pos[2], pos[1]:pos[3], :]
                    ).save("static/train/text/text." + str(num) + "." + str(i) + ".png")
                else:
                    Image.fromarray(
                        cover_image_array[pos[0]:pos[2], pos[1]:pos[3], :]
                    ).save("static/train/not_text/not_text." + str(num) + "." + str(i) + ".png")
            else:
                if i in blocks_with_text:
                    Image.fromarray(
                        cover_image_array[pos[0]:pos[2], pos[1]:pos[3], :]
                    ).save("static/valid/text/text." + str(num) + "." + str(i) + ".png")
                else:
                    Image.fromarray(
                        cover_image_array[pos[0]:pos[2], pos[1]:pos[3], :]
                    ).save("static/valid/not_text/not_text." + str(num) + "." + str(i) + ".png")
            # progress_bar.printProgressBar(
            #     i + 1, block_instance.max_num_blocks(),
            #     prefix='Progress:', suffix='Complete', length=50)
        progress_bar.printProgressBar(
            num + 1, len(paths),
            prefix='Progress:', suffix='Complete', length=50)


if __name__ == '__main__':
    main()
