#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import empty, arange, exp, real, imag, pi
from numpy.fft import rfft, irfft

import math


class DCT:
    """
    Trnasformadas discretas de coseno necesarias para el procesamiento de señales
    """

    def dct(self, y):
        """
        1D DCT
        """
        N = len(y)
        N = len(y)
        y2 = empty(2*N, float)
        y2[:N] = y[:]
        y2[N:] = y[::-1]
        c = rfft(y2)
        phi = exp(-1j*pi*arange(N)/(2*N))
        return real(phi*c[:N])

    def idct(self, a):   # 1D inverse DCT
        N = len(a)
        c = empty(N+1, complex)
        phi = exp(1j*pi*arange(N)/(2*N))
        c[:N] = phi*a
        c[N] = 0.0
        return irfft(c)[:N]

    def dct2(self, y):   # 2D DCT
        M = y.shape[0]
        N = y.shape[1]
        a = empty([M, N], int)
        b = empty([M, N], int)
        for i in range(M):
            a[i, :] = self.dct(y[i, :])
            for j in range(N):
                b[:, j] = self.dct(a[:, j])
        return b

    def idct2(self, b):  # 2D inverse DCT
        M = b.shape[0]
        N = b.shape[1]
        a = empty([M, N], float)
        y = empty([M, N], float)
        a1 = empty([M, N], int)
        y1 = empty([M, N], int)
        for i in range(M):
            a[i, :] = self.idct(b[i, :])
        for j in range(N):
            y[:, j] = self.idct(a[:, j])
        for i in range(M):
            for p in range(N):
                if (y[i, p] - int(y[i, p])) < 0.5:
                    y1[i, p] = int(y[i, p])
                else:
                    y1[i, p] = int(y[i, p]) + 1
        return y1
