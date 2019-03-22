# -*- coding: utf-8 -*-
from helpers.blocks_class import BlocksImage
from helpers import utils

from PIL import Image
import numpy as np
from copy import deepcopy


class Gul2019():
    """
    Método de marca de agua digital frágil para imágenes
    """

    def insert(self, cover_image):
        cover_array = np.asarray(cover_image)
        # Red component
        red_cover_array = cover_array[:, :, 0]
        red_cover_array.setflags(write=1)
        # Dividing in 32x32 blocks
        blocks32x32 = BlocksImage(red_cover_array, 32, 32)
        # Touring each block 32x32
        for num_block in range(blocks32x32.max_num_blocks()):
            # Copying block 32x32
            blocksCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
            # Dividing in 16x16 blocks
            blocksCopy16x16 = BlocksImage(blocksCopy32x32, 16, 16)
            # Add key in 4th block, meanwhile I take a ones 16x16 matrix
            blocksCopy16x16.set_block(np.ones((16, 16)), 3)
            # Hash of blocks32x32 whit key
            blocksHash = utils.sha256Binary(blocksCopy16x16.get().tolist())
            # Binary block to insert
            binaryBlock = np.asarray(
                utils.bString2bIntlist(blocksHash)).reshape((16, 16))
            # Get 4th component of blocks32x32
            components = BlocksImage(blocks32x32.get_block(num_block), 16, 16)
            component4 = components.get_block(3)
            for i in range(16):
                for y in range(16):
                    if ((component4[i, y] + binaryBlock[i, y]) % 2) == 1:
                        component4[i, y] -= 1

        watermarked_image = Image.fromarray(cover_array)
        return watermarked_image

    def extract(self, watermarked_image):
        import cv2
        watermarked_array = np.asarray(watermarked_image)
        # Red component
        red_watermarked_array = watermarked_array[:, :, 0]
        red_watermarked_array.setflags(write=1)
        red_watermarked_array_noise = red_watermarked_array.copy()
        # Dividing in 32x32 blocks
        blocks32x32 = BlocksImage(red_watermarked_array_noise, 32, 32)
        originalblocks32x32 = BlocksImage(red_watermarked_array, 32, 32)
        # Touring each block 32x32
        modifiedBlocks = []
        for num_block in range(blocks32x32.max_num_blocks()):
            # Copying block 32x32
            blockCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
            # Dividing in 16x16 blocks
            blocksCopy16x16 = BlocksImage(blockCopy32x32, 16, 16)
            # Add key in 4th block, meanwhile I take a ones 16x16 matrix
            blocksCopy16x16.set_block(np.ones((16, 16), dtype=np.uint8), 3)
            # Hash of blocks32x32 whit key
            blocksHash = utils.sha256Binary(blocksCopy16x16.get().tolist())
            # Binary block to insert
            binaryBlock = np.asarray(
                utils.bString2bIntlist(blocksHash)).reshape((16, 16))
            # Get 4th component of blocks32x32
            components = BlocksImage(
                blocks32x32.get_block(num_block), 16, 16)
            component4 = components.get_block(3)
            band = True
            for i in range(16):
                for y in range(16):
                    if ((component4[i, y] + binaryBlock[i, y]) % 2) != 0:
                        band = False
            if not band:
                modifiedBlocks.append(num_block)
        print(modifiedBlocks)
        for item in modifiedBlocks:
            coord = blocks32x32.get_coord(item)
            cv2.rectangle(
                watermarked_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
        Image.fromarray(watermarked_array).save("static/tampered.png")
