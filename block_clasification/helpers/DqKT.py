#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np


class DqKT:
    """
    Transformada Discreta de Krawtchouk
    """

    def __init__(self):
        coef0 = np.array([0.0147885, -0.059767, 0.158129, -0.311834, 0.476334, -0.563607, 0.497054, -0.286974])
        coef1 = np.array([0.059767, -0.192251, 0.378225, -0.488673, 0.353587, 0.0464855, -0.45096, 0.497054])
        coef2 = np.array([0.158129, -0.378225, 0.474875, -0.223586, -0.252437, 0.41582, 0.0464855, -0.563607])
        coef3 = np.array([0.311834, -0.488673, 0.223586, 0.298509, -0.330481, -0.252437, 0.353587, 0.476334])
        coef4 = np.array([0.476334, -0.353587, -0.252437, 0.330481, 0.298509, -0.223586, -0.488673, -0.311834])
        coef5 = np.array([0.563607, 0.0464855, -0.41582, -0.252437, 0.223586, 0.474875, 0.378225, 0.158129])
        coef6 = np.array([0.497054, 0.45096, 0.0464855, -0.353587, -0.488673, -0.378225, -0.192251, -0.059767])
        coef7 = np.array([0.286974, 0.497054, 0.563607, 0.476334, 0.311834, 0.158129, 0.059767, 0.0147885])

        self.coef = np.array([coef0, coef1, coef2, coef3, coef4, coef5, coef6, coef7])

    def dqkt2(self, array):
        """
        2D DKT
        """
        dkt = np.ndarray(shape=(8, 8))
        for x in range(8):
            for y in range(8):
                dkt[x, y] = self.M(array, x, y)
        return dkt

    def idqkt2(self, array):
        """
        2D IDKT
        """
        idkt = np.ndarray(shape=(8, 8))
        for x in range(8):
            for y in range(8):
                idkt[x, y] = self.IM(array, x, y)
        return idkt

    def M(self, A, m, n):
        M = 0
        for x in range(8):
            for y in range(8):
                M += ((self.coef[x, m]*self.coef[y, n]))*float(A[x, y])
        return M

    def IM(self, A, m, n):
        IM = 0
        for x in range(8):
            for y in range(8):
                IM += float(A[x, y])*self.coef[m, x]*self.coef[n, y]
        return IM
