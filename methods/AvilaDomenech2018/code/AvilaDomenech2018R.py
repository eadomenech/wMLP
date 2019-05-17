# -*- coding: utf-8 -*-
from block_tools.blocks_class import BlocksImage
from image_tools.ImageTools import ImageTools
from helpers import utils
from qr_tools.MyQR62 import MyQR62

from transforms.DAT import DAT
from transforms.DqKT import DqKT

from PIL import Image
from pathlib import Path
from scipy import misc
import numpy as np
import random
import math


class AvilaDomenech2018R():
    """
    Método de marca de agua digital robusta
    """
    def __init__(self, key):
        
        self.key = key
        
        # Hash of key
        self.binary_hash_key = utils.md5Binary(self.key)

        # Cargando watermark
        self.watermark = Image.open("static/Watermarking.png").convert("1")

        # Utilizando Arnold Transforms
        for i in range(20):
            self.watermark = DAT().dat2(self.watermark)

        # Obteniendo array de la watermark
        watermark_array = np.asarray(self.watermark)

        # Datos de la watermark como lista
        watermark_as_list = watermark_array.reshape(
            (1, watermark_array.size))[0]
        
        # Tomando solo los valores correspondientes a los datos
        self.watermark_list = []
        for p in watermark_as_list:
            if p:
                self.watermark_list.append(0)
            else:
                self.watermark_list.append(255)
        
        # Calculando datos iniciales
        self.len_watermark_list = len(self.watermark_list)

        # Posiciones seleccionadas
        self.pos = []
    
    def generar(self, maximo):
        '''Genera posiciones a utilizar en el marcado'''
        assert self.len_watermark_list <= maximo
        p = [i for i in range(maximo)]
        random.shuffle(p)        
        self.pos = p[:self.len_watermark_list]

        # # Utilizar Bloques segun key
        # dic = {'semilla': 0.00325687, 'p': 0.22415897}
        # valores = []
        # cantidad = bt_of_cover.max_blocks()
        # for i in range(cantidad):
        #     valores.append(i)
        # v = pwlcm.mypwlcm_limit(dic, valores, len_of_watermark)
    
    def zigzag(self, n):
        indexorder = sorted(((x, y) for x in range(n) for y in range(n)), key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1]) % 2 else s[1]))
        return {index: n for n, index in enumerate(indexorder)}


    def get_indice(self, m):
        zarray = self.zigzag(8)
        px = -1
        py = -1
        n = int(len(zarray) ** 0.5 + 0.5)
        for x in range(n):
            for y in range(n):
                if zarray[(x, y)] == m:
                    px = x
                    py = y
        return (px, py)

    def insert(self, cover_image):
        
        print("...Proceso de insercion...")

        # Instancias necesarias
        itools = ImageTools()

        print("Convirtiendo a YCbCr")
        # Convirtiendo a modelo de color YCbCr
        cover_ycbcr_array = itools.rgb2ycbcr(cover_image)

        # Obteniendo componente Y
        cover_array = cover_ycbcr_array[:, :, 0]

        # Objeto de la clase Bloque
        bt_of_cover = BlocksImage(cover_array)

        # Generando bloques a marcar
        print("Generando bloques a marcar")
        self.generar(bt_of_cover.max_num_blocks())

        print("Marcando")
        # Marcar los self.len_watermark_list bloques
        for i in range(self.len_watermark_list):
            block = bt_of_cover.get_block(self.pos[i])

            # Valores de coeficiente y delta optimo para el bloque
            (c, delta) = (19, 128) 

            # Calculando Krawchout Transform
            dqkt_block = DqKT().dqkt2(block)

            negative = False
            (px, py) = self.get_indice(c)
            if dqkt_block[px, py] < 0:
                negative = True

            if self.watermark_list[i % self.len_watermark_list] == 0:
                # Bit a insertar 0
                dqkt_block[px, py] = 2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) + delta/2.0
            else:
                # Bit a insertar 1
                dqkt_block[px, py] = 2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) - delta/2.0

            if negative:
                dqkt_block[px, py] *= -1
            idqkt_block = DqKT().idqkt2(dqkt_block)
            inv = idqkt_block
            for x in range(8):
                for y in range(8):
                    if (inv[x, y] - int(inv[x, y])) < 0.5:
                        inv[x, y] = int(inv[x, y])
                    else:
                        inv[x, y] = int(inv[x, y]) + 1
                    if inv[x, y] > 255:
                        inv[x, y] = 255
                    if inv[x, y] < 0:
                        inv[x, y] = 0
            bt_of_cover.set_block(idqkt_block, self.pos[i])

        print("Convirtiendo a RGB")
        image_rgb_array = itools.ycbcr2rgb(cover_ycbcr_array)

        return Image.fromarray(image_rgb_array)

    def extract(self, watermarked_image):

        print("...Proceso de extraccion...")

        # Instancias necesarias
        itools = ImageTools()

        print("Convirtiendo a YCbCr")
        # Convirtiendo a modelo de color YCbCr
        watermarked_ycbcr_array = itools.rgb2ycbcr(watermarked_image)

        # Tomando componente Y
        watermarked_array = watermarked_ycbcr_array[:, :, 0]

        bt_of_watermarked_Y = BlocksImage(watermarked_array)

        extract = []

        print("Extrayendo")
        # Recorrer todos los bloques de la imagen
        for i in range(self.len_watermark_list):
            block = bt_of_watermarked_Y.get_block(self.pos[i])
            # Valores de coeficiente y delta optimo para el bloque
            (c, delta) = (19, 128)
            dqkt_block = DqKT().dqkt2(np.array(block, dtype=np.float32))
            
            negative = False
            
            (px, py) = self.get_indice(c)           
            if dqkt_block[px, py] < 0:
                negative = True
            
            C1 = (2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) + delta/2.0) - abs(dqkt_block[px, py])
            
            C0 = (2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) - delta/2.0) - abs(dqkt_block[px, py])

            if negative:
                C1 *= -1
                C0 *= -1
            if C0 < C1:
                extract.append(0)
            else:
                extract.append(1)

        wh = int(math.sqrt(self.len_watermark_list))
        extract_image = Image.new("1", (wh, wh), 255)
        array_extract_image1 = misc.fromimage(extract_image)

        for i in range(wh):
            for y in range(wh):
                if extract[wh*i+y] == 0:
                    array_extract_image1[i, y] = 0
        
        # myqr1 = MyQR62()
        
        watermark_extracted = misc.toimage(array_extract_image1)
        for i in range(10):
            watermark_extracted = DAT().dat2(watermark_extracted)
        
        # # Utilizacion de caracteristica de QR code
        
        # array = misc.fromimage(watermark_extracted)
        # watermark_extracted = misc.toimage(
        #     myqr1.get_resconstructed(array))

        # b = BlocksImage(misc.fromimage(watermark_extracted), 2, 2)
        # for m in range(b.max_num_blocks()):
        #     b.set_color(m)
        # watermark_extracted = misc.toimage(b.get())
        
        return watermark_extracted
