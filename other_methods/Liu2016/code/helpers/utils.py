# -*- coding: utf-8 -*-
import hashlib
import numpy as np


def md5Binary(key):
    hexa_data = hashlib.md5(key.encode('utf-8')).hexdigest()
    return bin(int(hexa_data, 16))[2:].zfill(8)


def sha256Binary(key):
    hexa_data = hashlib.sha256(repr(key).encode('utf-8')).hexdigest()
    sha_data = ("{0:8b}".format(int(hexa_data, 16)))
    for v in range(256-(len(sha_data))):
        sha_data = '0' + sha_data
    return sha_data


def bString2bIntlist(string):
    """
    Convierte una string correspondiente a un binario en una lista de int
    """
    return list(map(int, string))


def quantification(matrix, Q):
    '''Cuantificacion de matrix de 8x8'''
    assert len(matrix) == 8
    assert len(matrix[0]) == 8
    assert len(Q) == 8
    assert len(Q[0]) == 8
    q = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            q[i, j] = matrix[i, j]/Q[i, j]
    return q


def changeBase(entrada, base):    
    r = []
    while entrada > 0:
        pe = entrada // base
        r.insert(0, entrada % base)
        entrada = pe
    return r

