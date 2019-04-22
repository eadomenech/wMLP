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
    for clase in lista:
        paths = glob.glob(
            'static/original/' + clase + '/**/*.png', recursive=True)
        for num, path in enumerate(paths):
            Image.open(path).save(
                'static/train/' + clase + '/' + clase + '.' + str(num) + '.png')


if __name__ == '__main__':
    main()
