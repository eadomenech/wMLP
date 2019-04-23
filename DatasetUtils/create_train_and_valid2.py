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
        '16_120', '17_90', '19_53', '19_60', '19_130', '20_130', '28_94',
        '28_120', '34_130']
    
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
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    
    for folder in original_folders:
        for clase in lista:
            path = 'static/original/' + folder + '/' + clase
            try:
                os.stat(path)
            except:
                os.mkdir(path)
            paths = glob.glob(path + '/**/*.png', recursive=True)
            for num, path in enumerate(paths):
                Image.open(path).save(
                    'static/train/' + clase + '/' + clase + '.' + str(num) + '.png')


if __name__ == '__main__':
    main()
