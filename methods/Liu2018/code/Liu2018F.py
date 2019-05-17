# -*- coding: utf-8 -*-
from helpers.image_class import Block_RGBImage
from helpers.blocks_class import BlocksImage
from helpers import utils
from math import floor
from PIL import Image
import numpy as np
import cv2


class Liu2019():
    """
    This scheme is a blind dual watermarking mechanism for digital color images in which invisible robust watermarks are embedded for copyright protection and fragile watermarks are embedded for image authentication.
    """

    def __init__(self):
        self.wsize = 256
        self.n = 2
        # Building the fragile watermark
        fragil_watermark_2 = utils.bin2dec(utils.md5_to_key_bin("Aslorente"))
        self.fw = utils.changeBase(fragil_watermark_2, 3 ** self.n, self.wsize)

    def insert(self, cover_image):
        # Initial Values
        i, n, block_num, block_size = 0, 2, -1, 4
        # Write permission
        cover_array = np.asarray(cover_image)
        cover_array.setflags(write=1)
        # instance
        block_instace = Block_RGBImage(cover_array, block_size, block_size)
        print("\n")
        while i < self.wsize:
            block_num += 1
            p = BlocksImage(block_instace.get_block_image(block_num), 1, 1)
            m = p.max_num_blocks() // self.n
            for j in range(m):
                pos = list(range(j * self.n, (j + 1) * self.n))
                E = sum(
                    [p.get_block(pos[k])[0][0] * 3 ** k for k in range(self.n)]
                )  % 3 ** self.n
                aux = floor((3 ** self.n - 1) / 2)
                if i < self.wsize:
                    t_value = (self.fw[i] - E + aux) % 3 ** self.n
                    i += 1
                b = utils.changeBase(t_value, 3, self.n)
                d = [k - 1 for k in b]
                for k in range(n):
                    r = n - k - 1
                    p.get_block(pos[r])[0][0] += d[k]

        watermarked_image = Image.fromarray(cover_array)

        return watermarked_image

    def extract(self, watermarked_image):
        # Initial Values
        modified_blocks = []
        extracted_fragile_w = []
        i, n, block_num, block_size = 0, 2, -1, 4
        # Write permission
        watermarked_array = np.asarray(watermarked_image)
        watermarked_array.setflags(write=1)
        # instance
        block_instace = Block_RGBImage(
            watermarked_array, block_size, block_size
        )
        print("\n")
        while i < self.wsize:
            block_num += 1
            p = BlocksImage(block_instace.get_block_image(block_num), 1, 1)
            m = p.max_num_blocks() // self.n
            for j in range(m):
                pos = list(range(j * self.n, (j + 1) * self.n))
                E = sum(
                    [p.get_block(pos[k])[0][0] * 3 ** k for k in range(self.n)]
                )  % 3 ** self.n
                extracted_fragile_w.append(E)
                i += 1

        N = self.wsize // m
        for i in range(N):
            j = i * m
            k = (i + 1) * m
            if extracted_fragile_w[j:k] != self.fw[j:k]:
                modified_blocks.append(i)

        if modified_blocks != []:
            for item in modified_blocks:
                coord = block_instace.get_coord_block_image(item)
                cv2.rectangle(watermarked_array, (coord[2], coord[0]),
                (coord[3], coord[1]), (0, 255, 0), 1)

            Image.fromarray(watermarked_array).save("static/tampered.bmp")

        print("\n Modified blocks:",modified_blocks)
        print(" ")
        print(self.fw[:self.wsize])
        print(" ")
        print(extracted_fragile_w[:self.wsize])
        print(" ")
        return self.fw[:self.wsize] == extracted_fragile_w[:self.wsize]
