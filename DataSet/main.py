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
    # Creando carpeta result si no existe
    try:
        os.stat('result/')
    except:
        os.mkdir('result/')
    # Creando subcarpetas en result correspondientes a las variantes
    for path in lista:
        try:
            os.stat('result/' + path + '/')
        except:
            os.mkdir('result/' + path + '/')

    original_folders = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    
    for folder in original_folders:
        print("Procesando caperta: " + folder)
        for clase in lista:
            path = 'original/' + folder + '/' + clase + '/'
            try:
                os.stat(path)
            except:
                os.mkdir(path)
            paths = glob.glob(path + '*.png')
            for num, path in enumerate(paths):
                Image.open(path).save(
                    'result/' + clase + '/' + folder + '.' + str(num) + '.png')
    
    dic_cantidades = {}
    for clase in lista:
        paths = glob.glob('result/' + clase + '/*.png')
        dic_cantidades[clase] = len(path)
    
    print(dic_cantidades)


    


if __name__ == '__main__':
    main()
