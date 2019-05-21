# -*- coding: utf-8 -*-
# For the cluster

from PIL import Image

import os
import random
import glob

'''
Une todos los bloques de una misma clases sin importar la imagen de la que provenga. Necesita la clasificacion por clases definidas realizada anteriormente.
'''
def main():
    # Lista de clases definidas
    lista = [
        '16_130', '19_67', '19_73', '19_78', '19_82', '19_85',
        '19_90', '19_98', '19_115']
    # Creando carpeta result si no existe
    try:
        os.stat('join/')
    except:
        os.mkdir('join/')
    # Creando subcarpetas en result correspondientes a las variantes
    for path in lista:
        try:
            os.stat('join/' + path + '/')
        except:
            os.mkdir('join/' + path + '/')

    original_folders = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    
    for folder in original_folders:
        print("Procesando caperta: " + folder)
        for clase in lista:
            path = 'classified/' + folder + '/' + clase + '/'
            try:
                os.stat(path)
            except:
                os.mkdir(path)
            paths = glob.glob(path + '*.png')
            for num, path in enumerate(paths):
                Image.open(path).save(
                    'join/' + clase + '/' + folder + '.' + str(num) + '.png')
    
    dic_cantidades = {}
    for clase in lista:
        paths = glob.glob('join/' + clase + '/*.png')
        dic_cantidades[clase] = len(paths)    


if __name__ == '__main__':
    main()
