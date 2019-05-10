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
    
    def get_coord(self, pos):
        # Se cuentan las posiciones a partir de 0
        if pos < 62*62:
            x = pos // 62
            y = pos % 62
            return (x, y)
        raise Exception("There is no such block")
    
    def get_pos(self):
        '''Devuelve las posiciones donde se encuentran los datos'''
        pos = []
        pos += [396 + i for i in range(16)]
        pos += [458 + i for i in range(16)]
        pos += [520 + i for i in range(16)]
        pos += [582 + i for i in range(16)]
        pos += [644 + i for i in range(16)]
        pos += [706 + i for i in range(16)]
        pos += [768 + i for i in range(16)]
        pos += [830 + i for i in range(16)]
        pos += [892 + i for i in range(16)]
        pos += [954 + i for i in range(16)]
        pos += [1016 + i for i in range(16)]
        pos += [1078 + i for i in range(16)]

        pos += [1264 + i for i in range(16)]
        pos += [1326 + i for i in range(16)]
        pos += [1388 + i for i in range(16)]
        pos += [1450 + i for i in range(16)]

        pos += [1494 + i for i in range(12)]
        pos += [1508 + i for i in range(36)]
        pos += [1556 + i for i in range(12)]
        pos += [1570 + i for i in range(36)]
        pos += [1618 + i for i in range(12)]
        pos += [1632 + i for i in range(36)]
        pos += [1680 + i for i in range(12)]
        pos += [1694 + i for i in range(36)]
        pos += [1742 + i for i in range(12)]
        pos += [1756 + i for i in range(36)]
        pos += [1804 + i for i in range(12)]
        pos += [1818 + i for i in range(36)]
        pos += [1866 + i for i in range(12)]
        pos += [1880 + i for i in range(36)]
        pos += [1928 + i for i in range(12)]
        pos += [1942 + i for i in range(36)]
        pos += [1990 + i for i in range(12)]
        pos += [2004 + i for i in range(36)]
        pos += [2052 + i for i in range(12)]
        pos += [2066 + i for i in range(36)]
        pos += [2114 + i for i in range(12)]
        pos += [2128 + i for i in range(36)]
        pos += [2176 + i for i in range(12)]
        pos += [2190 + i for i in range(36)]
        pos += [2238 + i for i in range(12)]
        pos += [2252 + i for i in range(36)]
        pos += [2300 + i for i in range(12)]
        pos += [2314 + i for i in range(36)]

        pos += [2362 + i for i in range(12)]
        pos += [2376 + i for i in range(18)]
        pos += [2404 + i for i in range(8)]
        pos += [2424 + i for i in range(12)]
        pos += [2438 + i for i in range(18)]
        pos += [2466 + i for i in range(8)]

        pos += [2504 + i for i in range(14)]
        pos += [2528 + i for i in range(8)]
        pos += [2566 + i for i in range(14)]
        pos += [2590 + i for i in range(8)]
        pos += [2628 + i for i in range(14)]
        pos += [2652 + i for i in range(8)]
        pos += [2690 + i for i in range(14)]
        pos += [2714 + i for i in range(8)]
        pos += [2752 + i for i in range(14)]
        pos += [2776 + i for i in range(8)]
        pos += [2814 + i for i in range(14)]
        pos += [2838 + i for i in range(8)]
        pos += [2876 + i for i in range(14)]
        pos += [2900 + i for i in range(8)]
        pos += [2938 + i for i in range(14)]
        pos += [2962 + i for i in range(8)]

        pos += [3000 + i for i in range(32)]
        pos += [3062 + i for i in range(32)]
        pos += [3124 + i for i in range(32)]
        pos += [3186 + i for i in range(32)]
        pos += [3248 + i for i in range(32)]
        pos += [3310 + i for i in range(32)]
        pos += [3372 + i for i in range(32)]
        pos += [3434 + i for i in range(32)]

        return pos

    def get_data(self):
        '''Devuelve una lista con los datos'''
        lista = []
        for i in self.get_pos():
            coord = self.get_coord(i)
            lista.append(self.myqr[coord[0], coord[1]])
        return lista

    def set_data(self, lista):
        '''Asigna los datos correspondientes a los valores de lista'''
        assert len(lista) == 1444

        for item, value in enumerate(self.get_pos()):
            coord = self.get_coord(value)
            if lista[item]:
                self.myqr[coord[0], coord[1]] = 0
            else:
                self.myqr[coord[0], coord[1]] = 255
        
        return lista

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

        # Formato
        self.myqr[6:8, 22:24] = 255
        self.myqr[8:18, 22:24] = 0
        self.myqr[20:22, 22:24] = 255
        self.myqr[22:24, 6:10] = 255
        self.myqr[22:24, 10:12] = 0
        self.myqr[22:24, 12:16] = 255
        self.myqr[22:24, 16:24] = 0

        self.myqr[22:24, 40:42] = 0
        self.myqr[22:24, 42:44] = 255
        self.myqr[22:24, 44:54] = 0
        self.myqr[22:24, 54:56] = 255

        self.myqr[40:46, 22:24] = 0
        self.myqr[46:50, 22:24] = 255
        self.myqr[50:52, 22:24] = 0
        self.myqr[52:56, 22:24] = 255
        
        return self.myqr
