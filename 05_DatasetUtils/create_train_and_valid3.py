# -*- coding: utf-8 -*-
from helpers.utils import rgb2_black_or_white, sum_block
from helpers.blocks_class import BlocksImage
from helpers import utils, progress_bar
from PIL import Image, ImageEnhance
import numpy as np
import cv2

import os
import glob

'''Lo mismo que el 2 pero pone solo 6000 por clase'''


def main():
    lista = [
        '16_100', '19_67', '19_73', '19_78', '19_82', '19_85',
        '19_90', '19_98', '19_115', '19_128'
        ]
    
    # Create folders
    list_dir = ['static/train/', 'static/valid/', 'static/organized/']
    for i in list_dir:
        try:
            os.stat(i)
        except:
            os.mkdir(i)
            for y in lista:
                try:
                    os.stat(i+y)
                except:
                    os.mkdir(i+y)
    
    original_folders = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    
    for folder in original_folders:
        print("Organized: Procesando caperta " + folder)
        for clase in lista:
            path = 'static/original/' + folder + '/' + clase
            try:
                os.stat(path)
            except:
                os.mkdir(path)
            paths = glob.glob(path + '/**/*.png', recursive=True)
            for num, path in enumerate(paths):
                Image.open(path).save(
                    'static/organized/' + clase + '/' + clase + '.' + str(num) + '_' + folder + '.png')
    
    
    # print("Train and Valid: Procesando caperta ")
    # for clase in lista:
    #     path = 'static/organized/' + clase
    #     paths = glob.glob(path + '/**/*.png', recursive=True)
    #     import random
    #     random.shuffle(paths)
    #     cant_mult_1000 = len(paths) // 1000
    #     cant = cant_mult_1000 * 1000
    #     if cant_mult_1000 > 10:
    #         d = (cant_mult_1000 // 10) * 1000
    #     else:
    #         d = 1000
    #     for num in range(cant):
    #         if num >= d:
    #             Image.open(paths[num]).save(
    #                 'static/train/' + clase + '/' + clase + '.' + str(num) + '.png')
    #         else:
    #             Image.open(paths[num]).save(
    #                 'static/valid/' + clase + '/' + clase + '.' + str(num) + '.png')

    print("Train and Valid: Procesando caperta ")
    for clase in lista:
        path = 'static/organized/' + clase
        paths = glob.glob(path + '/*.png')
        import random
        random.shuffle(paths)
        cant_mult_1000 = len(paths) // 1000
        cant = cant_mult_1000 * 1000
        if cant > 6000:
            cant = 6000
        d = cant / 10
        for num in range(cant):
            if num >= d:
                Image.open(paths[num]).save(
                    'static/train/' + clase + '/' + clase + '.' + str(num) + '.png')
            else:
                Image.open(paths[num]).save(
                    'static/valid/' + clase + '/' + clase + '.' + str(num) + '.png')


if __name__ == '__main__':
    main()
