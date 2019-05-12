# -*- coding: utf-8 -*-
from helpers.blocks_class import BlocksImage
from helpers import utils, progress_bar
from PIL import Image
import numpy as np
import cv2


class Ching_Sheng2019():
    """
    The aim of the proposed scheme in this paper is to
    embed robust and fragile watermark into the host image at the same time
    """

    def insert(self, cover_image):
        # Initial Values
        cover_array = np.asarray(cover_image)
        # Red component
        red_cover_array = cover_array[:, :, 2]
        red_cover_array.setflags(write=1)
        # instance
        block_instace_4x4 = BlocksImage(red_cover_array, 4, 4)
        print("\n")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(
            0, block_instace_4x4.max_num_blocks(), prefix='Progress:', suffix='Complete', length=50)
        for i in range(block_instace_4x4.max_num_blocks()):
            block4x4 = block_instace_4x4.get_block(i)
            p = BlocksImage(block4x4, 1, 1)
            L = [p.get_block(j)[0][0] >> 2 for j in range(16)]
            L = [
                block_instace_4x4.get_coord(i)[0], block_instace_4x4.get_coord(i)[2]//4] + L
            fragil_watermark = utils.xor_on(utils.md5_to_list_bin(L))
            for j in range(16):
                p.set_block([(L[j] << 2) + int(fragil_watermark[j])], j)
            progress_bar.printProgressBar(
                i + 1, block_instace_4x4.max_num_blocks(),
                prefix='Progress:', suffix='Complete', length=50)

        watermarked_image = Image.fromarray(cover_array)

        return watermarked_image

    def extract(self, watermarked_image):
        # Initial Values
        modified_blocks = []
        watermarked_array = np.asarray(watermarked_image)
        # Red component
        red_watermarked_array = watermarked_array[:, :, 0]
        red_watermarked_array.setflags(write=1)
        # instance
        block_instace_4x4 = BlocksImage(red_watermarked_array, 4, 4)
        print("\n")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(
            0, block_instace_4x4.max_num_blocks(), prefix='Progress:', suffix='Complete', length=50)
        for i in range(block_instace_4x4.max_num_blocks()):
            block4x4 = block_instace_4x4.get_block(i)
            p = BlocksImage(block4x4, 1, 1)
            L = [p.get_block(j)[0][0] >> 2 for j in range(16)]
            L.extend(block_instace_4x4.get_coord(i))
            fragil_watermark = utils.xor_on(utils.md5_to_list_bin(L))
            authen = [p.get_block(j)[0][0] % 2 for j in range(16)]
            if fragil_watermark != authen:
                modified_blocks.append(i)
            progress_bar.printProgressBar(
                i + 1, block_instace_4x4.max_num_blocks(),
                prefix='Progress:', suffix='Complete', length=50)

        if modified_blocks != []:
            for item in modified_blocks:
                coord = block_instace_4x4.get_coord(item)
                cv2.rectangle(watermarked_array, (coord[2], coord[0]),
                (coord[3], coord[1]), (0, 255, 0), 1)

            Image.fromarray(watermarked_array).save("static/tampered.bmp")

        print("\n Modified blocks:",modified_blocks)

        return modified_blocks == []
