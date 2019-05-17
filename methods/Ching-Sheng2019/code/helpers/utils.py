# -*- coding: utf-8 -*-
import numpy as np
import hashlib
import math


def md5_to_list_int(L):
    hexa_data = hashlib.md5(np.asarray(L)).hexdigest()
    return [int(format(ord(x), '08b'), 2) for x in hexa_data][:16]


def md5_to_list_bin(L):
    hexa_data = hashlib.md5(np.asarray(L)).hexdigest()
    return "".join(format(ord(x), '08b') for x in hexa_data)[:128]


def xor(x, y):
    if len(x) != len(y):
        print("The operation can not be performed")
    else:
        return "".join(str(int(x[i]) ^ int(y[i])) for i in range(len(x)))


def xor_on(x):
    cad = x[:16]
    for i in range(1, 8):
        j = i * 16
        k = (i + 1) * 16
        cad = xor(cad, x[j:k])
    return cad


def pwlcm(x, p):
    if x >= 0 and x < p:
        return x / p
    elif x >= p and x < 0.5:
        return (x - p) / (0.5 - p)
    elif x >= 0.5 and x < 1:
        return pwlcm(1 - x, p)


def chaotic_map(x, p, n):
    L = []
    for i in range(n):
        x = pwlcm(x, p)
        L.append(int(math.floor(x * 10 ** (14) % n)))
    return L


def perm(L, ind):
    pos = []
    n = len(ind)
    for i in range(n):
        pos.append(L[ind[i]])
    return pos


def set_diff(L1, L2):
    n = len(L2)
    for i in range(n):
        L1.remove(L2[i])
    return L1


def list_reduced(L):
    R = []
    for i in L:
        if i not in R:
            R.append(i)
    return R


def random_list(x, p, L):
    pos = []
    ind = []
    ind = list_reduced(chaotic_map(x, p, len(L)))
    pos = perm(L, ind)
    if len(pos) == len(L):
        return pos
    elif len(pos) == 1:
        return L
    else:
        return pos + random_list(x, p, set_diff(L, pos))
    return pos


def gaussian_noise(B, m=8, n=8):
    import random
    if len(B.shape) == 2:
        for i in range(m):
            for j in range(n):
                altera_valor= random.uniform(1, 15)
                suma_or_resta = random.randint(0, 1)
                if suma_or_resta == 0:
                    B[i][j] = int(B[i][j]+altera_valor)
                else:
                    B[i][j] = int(B[i][j]-altera_valor)
    else:
        for i in range(m):
            for j in range(n):
                [r, g, b] = B[i][j][:]
                altera_valor= random.uniform(1, 15)
                suma_or_resta = random.randint(0, 1)
                if suma_or_resta == 0:
                    B[i][j][:] = [
                        int(r+altera_valor),
                        int(g+altera_valor),
                        int(b+altera_valor)
                    ]
                else:
                    B[i][j][:] = [
                        int(r-altera_valor),
                        int(g-altera_valor),
                        int(b-altera_valor)
                    ]
    return B


def sp_noise(B, prob, m=8, n=8):
    import random
    if len(B.shape) == 2:
        for i in range(m):
            for j in range(n):
                if random.random() < prob:
                    sal_p = random.randint(0, 1)
                    if sal_p == 0:
                        sal_p = 0
                    else:
                        sal_p = 255
                    B[i][j] = sal_p
    else:
        for i in range(m):
            for j in range(n):
                [r, g, b] = B[i][j][:]
                if random.random() < prob:
                    sal_p = random.randint(0, 1)
                    if sal_p == 0:
                        sal_p = 0
                    else:
                        sal_p = 255
                    B[i][j][:] = [sal_p, sal_p, sal_p]
    return B


def cropping_noise(B, m=8, n=8):
    if len(B.shape) == 2:
        for i in range(m):
            for j in range(n):
                B[i][j] = 0
    else:
        for i in range(m):
            for j in range(n):
                B[i][j][:] = 0
    return B
