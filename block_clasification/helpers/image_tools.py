#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image


class ImageTools():
    """Esta clase implementa las herramientas utiles en el tratamiento de
    imagenes.
    """

    def rgb2ycbcr(self, im):
        xform = np.array(
            [[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
        ycbcr = np.dot(im, xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return ycbcr

    def ycbcr2rgb(self, im):
        xform = np.array(
            [[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = im.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = np.dot(rgb, xform.T)
        rgb = self.modelo_entero(rgb)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return np.uint8(rgb)

    def modelo_entero(self, array):
        for fila in range(len(array)):
            for columna in range(len(array[0])):
                for pos in range(3):
                    array[fila][columna][pos] = round(array[fila][columna][pos])
        return array
