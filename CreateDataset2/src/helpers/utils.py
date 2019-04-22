

import cv2
from PIL import Image, ImageChops, ImageOps, ImageEnhance
import numpy as np


def rgb2_black_or_white(cover):
    if cover:
        modified_cover = ImageChops.invert(cover)
        cover = ImageChops.subtract(modified_cover, cover)
        # Convert to grayscale
        cover = cover.convert('L')
        cover = ImageChops.invert(cover)
        cover = ImageEnhance.Brightness(cover).enhance(2.0)
        cover = ImageOps.colorize(cover, (0, 0, 0), (255, 255, 255))
    # Convert to grayscale
    cover = cover.convert('L')
    # Convert to array
    cover_array = np.asarray(cover)
    cover_array.setflags(write=1)
    # convert to binary cover array
    ret, binary_cover_array = cv2.threshold(
        cover_array,127,255,cv2.THRESH_BINARY)
    return binary_cover_array


def sum_block(block):
    L = []
    n = len(block)
    aux = [[(1 if block[i][j] == 0 else 0) for i in range(n)] for j in range(n)]
    for i in range(n):
        L.extend(aux[i])
    return sum(L)
