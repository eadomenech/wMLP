# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
from scipy import misc


class MyQR62:

    def __init__(self):
        self.myqr = np.ndarray(shape=(62, 62))
        for x in range(62):
            for y in range(62):
                self.myqr[x, y] = 100

    def get_data(self):    # Devuelve una lista con los datos

        return None

    def set_data(self, lista):    # Asigna los datos correspondientes a los valores de lista

        return None

    def get_resconstructed(self, qr_array):
        qr_estandar = self.get_qr()
        for i in range(62):
            for j in range(62):
                if qr_estandar[i, j] == 100:
                    qr_estandar[i, j] = qr_array[i, j]
        return qr_estandar

    def get_qr(self):  # Devuelve array correspondiente a la imagen del QR Code
        # Franja blanca de arriba y abajo
        for i in range(6):
            for y in range(62):
                self.myqr[i, y] = 255
                self.myqr[56+i, y] = 255

        # Franja blanca de la izquierda y derecha
        for i in range(62):
            for y in range(6):
                self.myqr[i, y] = 255
                self.myqr[i, 61-y] = 255

        # Cuadro superior izquierdo
        self.myqr[6:22, 6:22] = 255  # Blanco
        self.myqr[6:20, 6:20] = 0  # Negro
        self.myqr[8:18, 8:18] = 255  # Blanco
        self.myqr[10:16, 10:16] = 0  # Negro

        # Cuadro superior derecho
        self.myqr[6:22, 40:56] = 255  # Blanco
        self.myqr[6:20, 42:56] = 0  # Negro
        self.myqr[8:18, 44:54] = 255  # Blanco
        self.myqr[10:16, 46:52] = 0  # Negro

        # Cuadro inferiro izquierdo
        self.myqr[40:56, 6:22] = 255  # Blanco
        self.myqr[42:56, 6:20] = 0  # Negro
        self.myqr[44:54, 8:18] = 255  # Blanco
        self.myqr[46:52, 10:16] = 0  # Blanco

        # Cuadro pequeno inferior derecha
        self.myqr[38:48, 38:48] = 0  # Negro
        self.myqr[40:46, 40:46] = 255  # Blanco
        self.myqr[42:44, 42:44] = 0  # Negro

        # Puntos de alineamiento
        self.myqr[22:24, 18:20] = 0
        self.myqr[24:26, 18:20] = 255
        self.myqr[26:28, 18:20] = 0
        self.myqr[28:30, 18:20] = 255
        self.myqr[30:32, 18:20] = 0
        self.myqr[32:34, 18:20] = 255
        self.myqr[34:36, 18:20] = 0
        self.myqr[36:38, 18:20] = 255
        self.myqr[38:40, 18:20] = 0

        self.myqr[18:20, 22:24] = 0
        self.myqr[18:20, 24:26] = 255
        self.myqr[18:20, 26:28] = 0
        self.myqr[18:20, 28:30] = 255
        self.myqr[18:20, 30:32] = 0
        self.myqr[18:20, 32:34] = 255
        self.myqr[18:20, 34:36] = 0
        self.myqr[18:20, 36:38] = 255
        self.myqr[18:20, 38:40] = 0

        return self.myqr
