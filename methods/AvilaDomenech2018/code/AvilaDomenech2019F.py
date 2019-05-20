# -*- coding: utf-8 -*-
from block_tools.blocks_class import BlocksImage
from helpers import utils

from PIL import Image
import numpy as np
from copy import deepcopy


class AvilaDomenech2019F():
    """
    Método de marca de agua digital frágil para imágenes RGB
    """
    def __init__(self, key):
        self.key = key
        # Hash of key
        self.binary_hash_key = utils.md5Binary(self.key)

    def insert(self, cover_image):
        # Image to array
        cover_array = np.asarray(cover_image)
        # Blue component
        blue_cover_array = cover_array[:, :, 2]
        blue_cover_array.setflags(write=1)
        # Dividing in 32x32 blocks
        blocks32x32 = BlocksImage(blue_cover_array, 32, 32)
        # Touring each block 32x32
        for num_block in range(blocks32x32.max_num_blocks()):
            # Copying block 32x32
            blocksCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
            # Dividing in 16x16 blocks
            blocksCopy16x16 = BlocksImage(blocksCopy32x32, 16, 16)
            # Get first block
            first_block = blocksCopy16x16.get_block(0)
            # Pariar
            for i in range(16):
                for y in range(16):
                    if (first_block[i, y] % 2) == 1:
                        first_block[i, y] -= 1
            # Hash of blocksCopy32x32 pareado
            blocksHash = utils.sha256Binary(blocksCopy32x32.tolist())
            # Insert data
            for i in range(16):
                for y in range(16):
                    first_block[i, y] += int(blocksHash[16*i + y])
            # Update block
            blocks32x32.set_block(blocksCopy32x32, num_block)
        watermarked_image = Image.fromarray(cover_array)
        return watermarked_image

    def extract(self, watermarked_image):
        import cv2
        # To array
        watermarked_array = np.asarray(watermarked_image)
        # Blue component
        blue_watermarked_array = watermarked_array[:, :, 2]
        blue_watermarked_array.setflags(write=1)
        blue_watermarked_array_noise = blue_watermarked_array.copy()
        # Dividing in 32x32 blocks
        blocks32x32 = BlocksImage(blue_watermarked_array_noise, 32, 32)
        # Touring each block 32x32
        modifiedBlocks = []
        for num_block in range(blocks32x32.max_num_blocks()):
            # Copying block 32x32
            blockCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
            # Dividing in 16x16 blocks
            blocksCopy16x16 = BlocksImage(blockCopy32x32, 16, 16)
            # Get first block
            first_block = blocksCopy16x16.get_block(0)
            # Watermark
            w = ''
            # Pariar
            for i in range(16):
                for y in range(16):
                    if (first_block[i, y] % 2) == 1:
                        first_block[i, y] -= 1
                        w += '1'
                    else:
                        w += '0'
            # Hash of blocksCopy32x32 pareado
            blocksHash = utils.sha256Binary(blockCopy32x32.tolist())
            if w != blocksHash:
                modifiedBlocks.append(num_block)
        print(modifiedBlocks)
        for item in modifiedBlocks:
            coord = blocks32x32.get_coord(item)
            cv2.rectangle(
                watermarked_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
        return Image.fromarray(watermarked_array)
