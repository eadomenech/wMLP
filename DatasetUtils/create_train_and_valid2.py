# -*- coding: utf-8 -*-
from helpers.utils import rgb2_black_or_white, sum_block
from helpers.blocks_class import BlocksImage
from helpers import utils, progress_bar
from PIL import Image, ImageEnhance
import numpy as np
import cv2

import os
import glob


def main():
    lista = [
        '16_100', '19_52', '19_54', '19_56', '19_61', '19_69', '19_76',
        '19_84', '19_93', '19_123', '28_90', '28_94', '28_97', '28_120',
        '34_130']
    
    # Create folders
    list_dir = ['static/train/', 'static/valid/']
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
        print("Procesando caperta " + folder)
        for clase in lista:
            path = 'static/original/' + folder + '/' + clase
            try:
                os.stat(path)
            except:
                os.mkdir(path)
            paths = glob.glob(path + '/**/*.png', recursive=True)
            for num, path in enumerate(paths):
                p = np.random.rand()
                if p > 0.1:
                    Image.open(path).save(
                        'static/train/' + clase + '/' + clase + '.' + str(num) + '_' + folder + '.png')
                else:
                    Image.open(path).save(
                        'static/valid/' + clase + '/' + clase + '.' + str(num) + '_' + folder + '.png')


if __name__ == '__main__':
    main()
