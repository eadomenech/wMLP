# -*- coding: utf-8 -*-
import glob


def crear_lista():
    lista = []

    for c in range(34):
        coef = c + 16
        for d in range(100):
            delta = d + 30
            lista.append(str(coef)+'_'+str(delta))
    
    return lista