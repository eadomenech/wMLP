# -*- coding: utf-8 -*-
# For the cluster

from PIL import Image

import os
import random
import glob
import util

'''
Encargada de graficar el mapa de color teniendo en cuenta la frecuencia,
el valor delta y la cantidad de imagenes donde el optimo se obtiene con estos valores
'''
def main():
    # Creando lista de direcciones posibles
    lista = util.crear_lista()
    # Totalizando    
    dic_cantidades = {}
    for clase in lista:
        paths = glob.glob('join/' + clase + '/*.png')
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
    plt.colorbar(orientation='horizontal')
    plt.xlabel('Embedding strength', size=18)
    plt.ylabel('Coefficient', size=18)
    plt.show()


    


if __name__ == '__main__':
    main()
