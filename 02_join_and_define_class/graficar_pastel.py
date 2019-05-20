# -*- coding: utf-8 -*-
# For the cluster

from PIL import Image

import os
import random
import glob
import util

'''
Encargada de graficar barras correspondientes a la cantidad de imagenes
donde se alcanza el optimo en la frecuencia dada
'''
def main():
    # Creando lista de direcciones posibles
    lista = util.crear_lista()
    # Totalizando    
    dic_cantidades = {}
    frec = None
    for clase in lista:
        if not frec == clase[:2]:
            frec = clase[:2]
            dic_cantidades[frec] = 0
        paths = glob.glob('join/' + clase + '/*.png')
        dic_cantidades[frec] += len(paths)
    
    import matplotlib.pyplot as plt
    import numpy as np
    x = [dic_cantidades['16'], dic_cantidades['19']]
    suma = 0
    for d in dic_cantidades.keys():
        if not ((d == '16') or (d == '19')):
            suma += dic_cantidades[d]  

    x.append(suma)
    dic_cantidades.values()
    labels = ['16', '19', 'others']
    plt.pie(x, labels=labels, autopct='%1.1f%%')
    plt.show()    


if __name__ == '__main__':
    main()
