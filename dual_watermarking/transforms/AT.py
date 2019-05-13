#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from cvxopt import matrix


class AT:
    """
    Transformada de Arnold
    """
    def __init__(self, ite):
        self.iter = ite
        self.ma = np.array([[1, 1], [1, 2]])

    def at2(self, array):
        """
        2D AT
        """
        n = len(array)
        out = np.zeros_like(array)

        for a in range(self.iter):
            for b in range(n):
                for c in range(n):
                    p = self.ma * [b, c]
                    out[(p[1] % n), (p[0] % n)] = array[b, c]
            array = out

        return out


    # def at2(self, array):
    #     """
    #     2D AT
    #     """
