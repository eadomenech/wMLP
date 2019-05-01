# -*- coding: utf-8 -*-
# For the cluster

from PIL import Image

import os
import random
import glob

def crear_lista():
    lista = []

    for c in range(34):
        coef = c + 16
        for d in range(100):
            delta = d + 30
            lista.append(str(coef)+'_'+str(delta))
    
    return lista



def main():
    # Creando lista de direcciones posibles
    lista = crear_lista()
    # Totalizando    
    dic_cantidades = {}
    for clase in lista:
        paths = glob.glob('result/' + clase + '/*.png')
        dic_cantidades[clase] = len(paths)
    
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(16, 50, 1)
    y = np.arange(30, 130, 1)
    X, Y = np.meshgrid(x, y)
    ellipses = []
    for c in range(64):
        lista_temp = []
        for d in range(130):
            try:
                lista_temp.append(dic_cantidades[str(c)+'_'+str(d)])
            except:
                lista_temp.append(0)
        ellipses.append(lista_temp)
    plt.imshow(ellipses)
    plt.colorbar()
    plt.show()


    


if __name__ == '__main__':
    main()
