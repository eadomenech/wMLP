# -*- coding: utf-8 -*-
# from helpers.image_class import Block_RGBImage
from block_tools.blocks_class import BlocksImage
from helpers.utils import md5Binary, base2decimal, decimal2base
from math import floor
from PIL import Image
from scipy import misc
import numpy as np
import cv2


class Liu2018F():
    """
    This scheme is a blind dual watermarking mechanism for digital color images in which invisible robust watermarks are embedded for copyright protection and fragile watermarks are embedded for image authentication.
    """

    def __init__(self, key, n=2):

        # Variables
        self.n = n
        self.wsize = 256        
        
        # Building the fragile watermark
        # self.fw_binary = md5Binary(str(key))
        self.fw_binary = [1, 1, 1, 1, 1, 1] 
        self.fw_decimal = base2decimal(self.fw_binary, 2)
        self.fw_3n = decimal2base(self.fw_decimal, 3 ** self.n)
        # self.fw = self.changeBase(
        #     fragil_watermark_2, 3 ** self.n, self.wsize)
        # print(self.fw)
    
    def calculate_E(self, U, s):
        for i in range(len(U)):
            suma = (3 ** i) * U[i]
        return suma % (3 ** self.n)

    def calculate_t(self, s, E):
        return (s - E + (3 ** self.n - 1)/2) % (3 ** self.n)
    
    def insertarEnComponente(self, component_image):
        '''Insertar en una componente'''
        # Datos como array
        array = misc.fromimage(component_image)
        # Datos como lista
        lista = array.reshape((1, array.size))[0]
        # Recorriendo los U
        # for i in range(len(lista)//self.n):
        for i in range(1000):
            print("{} de {}".format(i, len(lista)//self.n))
            s = self.fw_3n[i % len(self.fw_3n)]
            E = self.calculate_E(lista[i*self.n:self.n*(i+1)], s)
            t = self.calculate_t(s, E)
            t_3 = decimal2base(t, 3)
            t_3 = [(t_3[k] - 1) for k in range(len(t_3))]
            while(len(t_3) < self.n):
                t_3.insert(0, 0)
            for l in range(self.n):
                    lista[i*self.n + l] += t_3[(l*-1)-1]

    def insert(self, cover_image):
        # Dividiendo en componentes RGB
        r, g, b = cover_image.split()
        
        # Marcando cada componente
        self.insertarEnComponente(r)
        self.insertarEnComponente(g)
        self.insertarEnComponente(b)

        return Image.merge("RGB", (r, g, b))

    def extract(self, watermarked_image):
        return watermarked_image
