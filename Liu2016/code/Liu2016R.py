# -*- coding: utf-8 -*-


from helpers.blocks_class import BlocksImage
from helpers.ImageTools import ImageTools
from helpers import utils

from PIL import Image
import numpy as np
import pywt


class Liu2016R():
    """
    Método de marca de agua digital robusta
    """
    def __init__(self, key, k=1.0):
        self.key = key
        # Hash of key
        self.binary_hash_key = utils.md5Binary(self.key)

        # Strength of the watermark
        self.k = k

        # Quantification matrix
        self.Q = np.asarray([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

        # Cargando watermark
        watermark = Image.open("static/Watermarking.png").convert("1")

        # obteniendo array de la watermark
        watermark_array = np.asarray(watermark)            

        # Calculando e imprimeindo datos iniciales
        self.len_of_watermark = watermark_array.size
        print('Cantidad de bit a insertar: ', self.len_of_watermark)

        # Datos de la watermark como lista
        self.list_bit_of_watermark = watermark_array.reshape((1, self.len_of_watermark))[0]

    def insert(self, cover_image):
        
        # Instancia
        itools = ImageTools()

        # Convirtiendo a modelo de color YCbCr
        cover_ycbcr_array = itools.rgb2ycbcr(cover_image)

        # Obteniendo componente Y
        cover_array = cover_ycbcr_array[:, :, 0]

        # DWT
        LL, [LH, HL, HH] = pywt.dwt2(cover_array, 'haar')
        
        # Dividiendo LL en bloques de 8x8
        bt_of_LL = BlocksImage(LL)
        bt_of_HH = BlocksImage(HH)

        for i in range(bt_of_LL.max_num_blocks()):
            # Cuantificado
            QLL = utils.quantification(bt_of_LL.get_block(i), self.Q)
            # replaced directly by the resulting Q-LL
            bt_of_HH.set_block(QLL, i)
        for i in range(self.len_of_watermark + 1):
            colums = len(LL[0])
            # Marcado
            if i > 0:
                px = i // colums
                py = i % colums
                LL[px, py] += self.list_bit_of_watermark[i-1] * self.k
        
        cover_array = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        image_rgb_array = itools.ycbcr2rgb(cover_ycbcr_array)                
        
        return Image.fromarray(image_rgb_array)

    def extract(self, watermarked_image):
        pass
        return watermarked_image
