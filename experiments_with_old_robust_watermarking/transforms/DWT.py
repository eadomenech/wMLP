# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import pywt


class DWT:  # DWT necesarias para el procesamiento de señales

    def dwt2(self, matrix):    # DWT Obtener los coeficientes 2D en el 1er nivel
        coeffs = pywt.dwt2(matrix, 'haar')
        ll = coeffs[0].copy(order='C')
        lh = coeffs[1][0].copy(order='C')
        hl = coeffs[1][1].copy(order='C')
        hh = coeffs[1][2].copy(order='C')
        return [ll, [lh, hl, hh]]

    def idwt2(self, coefficientes):  # IDWT Función inversa 2D en el 1er nivel
        inv = pywt.idwt2(coefficientes, 'haar')
        return inv

    def idwt2_int(self, coefficientes):  # IDWT Función inversa 2D en el 1er nivel
        inv = pywt.idwt2(coefficientes, 'haar')
        for x in range(len(inv)):
            for y in range(len(inv[0])):
                if (inv[x, y] - int(inv[x, y])) < 0.5:
                    inv[x, y] = int(inv[x, y])
                else:
                    inv[x, y] = int(inv[x, y]) + 1
        return inv

    # WaveletPacket2D
    #   a - LL, low-low coefficients
    #   h - LH, low-high coefficients
    #   v - HL, high-low coefficients
    #   d - HH, high-high coefficients

    def wp(self, matrix):    # WaveletPacket2D Obtener los coeficientes 2D
        wp = pywt.WaveletPacket2D(data=matrix, wavelet='db1', mode='sym')
        return wp

    def iwp(self, a, h, v, d):  # WaveletPacket2D Función inversa 2D
        new_wp['a'] = a
        new_wp['h'] = h
        new_wp['v'] = v
        new_wp['d'] = d
        new_wp.reconstruct(update=False)
        return new_wp
