# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import pywt


def main():
    data = np.asarray([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])
    print(data)
    # DWT
    LL, [LH, HL, HH] = pywt.dwt2(data, 'haar')

    print(LH)
    
    # Working
    rows, colums = len(LL), len(LL[0])
    print(rows)
    print(colums)

    cant =  LL.size

    for i in range(cant):
        px = i // colums
        py = i - (px * colums)
        LH[px, py] = LH[px, py] + 1000
    
    print(LH)
    
    # IDWT
    data1 = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    print(data1)
    
    


if __name__ == '__main__':
    main()
