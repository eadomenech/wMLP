#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from scipy.fftpack import dct

import math


class DCT:
    """
    Trnasformadas discretas de coseno necesarias para el procesamiento de señales
    """

    def dct2(self, signal):
        return dct(signal, norm="ortho")

    def idct2(self, transformed):
        idct_block = dct(transformed, type=3, norm="ortho")
        return idct_block

    def dct2_int(self, signal):
        dct_block = dct(signal, norm="ortho")
        for x in range(len(dct_block)):
            for y in range(len(dct_block[0])):
                if (dct_block[x, y] - int(dct_block[x, y])) < 0.5:
                    dct_block[x, y] = int(dct_block[x, y])
                else:
                    dct_block[x, y] = int(dct_block[x, y]) + 1
        return dct_block

    def idct2_int(self, transformed):
        idct_block = dct(transformed, type=3, norm="ortho")
        for x in range(len(idct_block)):
            for y in range(len(idct_block[0])):
                if (idct_block[x, y] - int(idct_block[x, y])) < 0.5:
                    idct_block[x, y] = int(idct_block[x, y])
                else:
                    idct_block[x, y] = int(idct_block[x, y]) + 1
        return idct_block
