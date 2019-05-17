# -*- coding: utf-8 -*-
from image_tools.ImageTools import ImageTools
from block_tools.blocks_class import BlocksImage
from PIL import Image
from scipy import misc
import numpy as np
import pywt
import math
import random


class Shivani2017R():
    """
    The aim of the proposed scheme in this paper is to
    embed robust watermark into the host image.
    """

    def __init__(self):

        # Cargando watermark
        self.watermark = Image.open("static/Watermarking.png").convert("1")
        self.watermark_list = []
        self.len_watermark_list = 0
        self.cant_posibilidades = None
        self.v = []
    
    def watermark_proccesing(self, tupla):
        '''Pre-procesa la marca de agua'''

        self.watermark_list = []
        
        x = tupla[0]
        y = tupla[1]

        m = min(x, y)

        self.watermark = self.watermark.resize((m, m))
        
        # Obteniendo array de la watermark
        watermark_array = np.asarray(self.watermark)

        # Datos de la watermark como lista
        watermark_as_list = watermark_array.reshape(
            (1, watermark_array.size))[0]
        
        # Tomando solo los valores correspondientes a los datos
        for p in watermark_as_list:
            if p:
                self.watermark_list.append(255)
            else:
                self.watermark_list.append(0)
        
        # Calculando e imprimeindo datos iniciales
        self.len_watermark_list = len(self.watermark_list)

        # Actualizando cant_posibilidades
        self.cant_posibilidades = x*y

    def insert(self, cover_image):
        # Instancias necesarias
        itools = ImageTools()

        # Embedding of Copyright Information (robust watermarking)
        # Convirtiendo a modelo de color YCbCr
        cover_ycbcr_array = itools.rgb2ycbcr(cover_image)

        # Tomando componente Y
        cover_array = cover_ycbcr_array[:, :, 0]

        # Wavelet Transform
        LL, [LH, HL, HH] = pywt.dwt2(cover_array, 'haar')

        # Watermark resize
        self.watermark_proccesing(LL.shape)

        colums = len(LL[0])

        print("Creando posisciones random")
        val = [x for x in range(self.cant_posibilidades)]
        random.shuffle(val)
        self.v = val[:self.len_watermark_list]
        
        print("Insertando")
        # Embedding
        for i in range(self.len_watermark_list):
            px = self.v[i] // colums
            py = self.v[i] - (px * colums)
            if self.watermark_list[i] == 255:
                if HL[px, py] <= LH[px, py]:
                    T = abs(LH[px, py])-abs(HL[px, py])
                    A3w = T + HL[px][py]
                    HL[px, py] = A3w + (LH[px, py]+HL[px, py])/2.0
            else:
                if LH[px, py] <= HL[px, py]:
                    T = abs(HL[px, py])-abs(LH[px, py])
                    A2w = T + LH[px, py]
                    LH[px, py] = A2w + (LH[px, py]+HL[px, py])/2.0
        
        # Inverse transform
        cover_ycbcr_array[:, :, 0] = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

        image_rgb_array = itools.ycbcr2rgb(cover_ycbcr_array)
        
        # Generating the watermarked image
        watermarked_image = Image.fromarray(image_rgb_array)

        return watermarked_image

    def extract(self, watermarked_image):
        # Instancias necesarias
        itools = ImageTools()

        # Embedding of Copyright Information (robust watermarking)
        # Convirtiendo a modelo de color YCbCr
        watermarked_ycbcr_array = itools.rgb2ycbcr(watermarked_image)

        # Tomando componente Y
        watermarked_array = watermarked_ycbcr_array[:, :, 0]

        # Wavelet Transform
        LL, [LH, HL, HH] = pywt.dwt2(watermarked_array, 'haar')

        colums = len(LL[0])

        extract = []
        
        # Extracting
        for i in range(self.len_watermark_list):
            px = self.v[i] // colums
            py = self.v[i] - (px * colums)
            if HL[px, py] >= LH[px, py]:
                extract.append(1)
            else:
                extract.append(0)
        
        wh = int(math.sqrt(self.len_watermark_list))
        extract_image1 = Image.new("1", (wh, wh), 255)
        array_extract_image = misc.fromimage(extract_image1)

        for i in range(wh):
            for y in range(wh):
                if extract[wh*i+y] == 0:
                    array_extract_image[i, y] = 0

        watermark_extracted = misc.toimage(array_extract_image)  
        
        return watermark_extracted
