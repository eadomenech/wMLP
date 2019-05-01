# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
from scipy import misc


class MyQR:

    def __init__(self):
        self.myqr = np.ndarray(shape=(124, 124))
        for x in range(124):
            for y in range(124):
                self.myqr[x, y] = 100

    def get_resconstructed(self, qr_array):
        qr_estandar = self.get_qr()
        for i in range(124):
            for j in range(124):
                if qr_estandar[i, j] == 100:
                    qr_estandar[i, j] = qr_array[i, j]
        return qr_estandar

    def get_qr(self):  # Devuelve array correspondiente a la imagen del QR Code
        # Franja blanca de arriba y abajo
        for i in range(12):
            for y in range(124):
                self.myqr[i, y] = 255
                self.myqr[112+i, y] = 255

        # Franja blanca de la izquierda y derecha
        for i in range(124):
            for y in range(12):
                self.myqr[i, y] = 255
                self.myqr[i, 123-y] = 255

        # Cuadro superior izquierdo
        self.myqr[12:44, 12:44] = 255  # Blanco
        self.myqr[12:40, 12:40] = 0  # Negro
        self.myqr[16:36, 16:36] = 255  # Blanco
        self.myqr[20:32, 20:32] = 0  # Negro

        # Cuadro superior derecho
        self.myqr[12:44, 80:116] = 255  # Blanco
        self.myqr[12:40, 84:112] = 0  # Negro
        self.myqr[16:36, 88:108] = 255  # Blanco
        self.myqr[20:32, 92:104] = 0  # Negro

        # Cuadro inferior izquierdo
        self.myqr[80:116, 12:44] = 255  # Blanco
        self.myqr[84:112, 12:40] = 0  # Negro
        self.myqr[88:108, 16:36] = 255  # Blanco
        self.myqr[92:104, 20:32] = 0  # Negro

        # Cuadro pequeno inferior derecha
        self.myqr[76:96, 76:96] = 0  # Negro
        self.myqr[80:92, 80:92] = 255  # Blanco
        self.myqr[84:88, 84:88] = 0  # Negro

        # Puntos de alineamiento
        self.myqr[44:48, 36:40] = 0
        self.myqr[48:52, 36:40] = 255
        self.myqr[52:56, 36:40] = 0
        self.myqr[56:60, 36:40] = 255
        self.myqr[60:64, 36:40] = 0
        self.myqr[64:68, 36:40] = 255
        self.myqr[68:72, 36:40] = 0
        self.myqr[72:76, 36:40] = 255
        self.myqr[76:80, 36:40] = 0

        self.myqr[36:40, 44:48] = 0
        self.myqr[36:40, 48:52] = 255
        self.myqr[36:40, 52:56] = 0
        self.myqr[36:40, 56:60] = 255
        self.myqr[36:40, 60:64] = 0
        self.myqr[36:40, 64:68] = 255
        self.myqr[36:40, 68:72] = 0
        self.myqr[36:40, 72:76] = 255
        self.myqr[36:40, 76:80] = 0

        return self.myqr
