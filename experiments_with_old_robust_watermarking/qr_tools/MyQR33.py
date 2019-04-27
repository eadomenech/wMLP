# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
from scipy import misc


class MyQR:

    def __init__(self, d):
        self.myqr = np.ndarray(shape=(33, 33))
        for x in range(33):
            for y in range(33):
                self.myqr[x, y] = 100

    def get_data(self):    # Devuelve una lista con los datos

        return None

    def set_data(self, lista):    # Asigna los datos correspondientes a los valores de lista

        return None

    def get_qr(self):  # Devuelve array correspondiente a la imagen del QR Code
        # Franja blanca de arriba y abajo
        for i in range(4):
            for y in range(33):
                self.myqr[i, y] = 255
                self.myqr[29+i, y] = 255

        # Franja blanca de la izquierda y derecha
        for i in range(33):
            for y in range(4):
                self.myqr[i, y] = 255
                self.myqr[i, 32-y] = 255

        # Cuadro superior izquierdo
        self.myqr[3:12, 3:12] = 255  # Blanco
        self.myqr[4:11, 4:11] = 0  # Negro
        self.myqr[5, 5:10] = 255  # Blanco
        self.myqr[9, 5:10] = 255  # Blanco
        self.myqr[6:9, 5] = 255  # Blanco
        self.myqr[6:9, 9] = 255  # Blanco

        # Cuadro superior derecho
        self.myqr[3:12, 21:30] = 255  # Blanco
        self.myqr[4:11, 22:29] = 0  # Negro
        self.myqr[5, 23:28] = 255  # Blanco
        self.myqr[9, 23:28] = 255  # Blanco
        self.myqr[6:9, 23] = 255  # Blanco
        self.myqr[6:9, 27] = 255  # Blanco

        # Cuadro inferiro izquierdo
        self.myqr[21:30, 3:12] = 255  # Blanco
        self.myqr[22:29, 4:11] = 0  # Negro
        self.myqr[23, 5:10] = 255  # Blanco
        self.myqr[27, 5:10] = 255  # Blanco
        self.myqr[24:27, 5] = 255  # Blanco
        self.myqr[24:27, 9] = 255  # Blanco

        # Cuadro pequeno inferior derecha
        self.myqr[20:25, 20:25] = 0  # Negro
        self.myqr[21, 21:24] = 255  # Blanco
        self.myqr[23, 21:24] = 255  # Blanco
        self.myqr[22, 21] = 255  # Blanco
        self.myqr[22, 23] = 255  # Blanco

        # Puntos de alineamiento
        self.myqr[12, 10] = 0
        self.myqr[13, 10] = 255
        self.myqr[14, 10] = 0
        self.myqr[15, 10] = 255
        self.myqr[16, 10] = 0
        self.myqr[17, 10] = 255
        self.myqr[18, 10] = 0
        self.myqr[19, 10] = 255
        self.myqr[20, 10] = 0

        self.myqr[10, 12] = 0
        self.myqr[10, 13] = 255
        self.myqr[10, 14] = 0
        self.myqr[10, 15] = 255
        self.myqr[10, 16] = 0
        self.myqr[10, 17] = 255
        self.myqr[10, 18] = 0
        self.myqr[10, 19] = 255
        self.myqr[10, 20] = 0

        return self.myqr
