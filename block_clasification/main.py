# -*- coding: utf-8 -*-
from tkinter import filedialog
from tkinter import *

from PIL import Image
import numpy as np

from pyjaya.clasic import JayaClasic
from pyjaya.utils import FloatRange, IntRange, BinaryRange
from helpers.blocks_class import BlocksImage
from helpers.image_tools import ImageTools
from helpers.DqKT import DqKT
from helpers.evaluations import Evaluations


clases = []


# def jaya(block_array):
#
#     def function(solution):
#         block_array = Image.open(block_path)
#         return sum(np.asarray(solution)**2)
#
#     print("RUN: JayaClasic")
#     listVars = [IntRange(1, 64), FloatRange(0.0, 200.0)]
#     ja = JayaClasic(20, listVars, function)
#     ja.toMaximize()
#     print(ja.run(5000))
#     print("--------------------------------------------------------------")

def zigzag(n):
    indexorder = sorted(
        ((x, y) for x in range(n) for y in range(n)), key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1]) % 2 else s[1]))
    return {index: n for n, index in enumerate(indexorder)}


def get_indice(m):
    zarray = zigzag(8)
    indice = []
    n = int(len(zarray) ** 0.5 + 0.5)
    for x in range(n):
        for y in range(n):
                if zarray[(x, y)] == m:
                    indice.append(x)
                    indice.append(y)
    return indice


def clasificar(block_path):
    block_image = Image.open(block_path)
    block_array = np.asarray(block_image)
    # Convirtiendo a modelo de color YCbCr
    block_ycbcr_array = ImageTools().rgb2ycbcr(block_array)
    # Y component
    Y_array = block_ycbcr_array[:, :, 0]
    # Krawtchouk transform
    dqkt_block = DqKT().dqkt2(np.array(Y_array, dtype=np.float32))
    # Bit a marcar
    watermark_bit = 1
    result = {'c': 0, 'delta': 1}
    psnr = 0.0
    for c in range(64):
        for d in range(20):
            delta = d + 1
            negative = False
            if dqkt_block[get_indice(c)[0], get_indice(c)[1]] < 0:
                negative = True

            if watermark_bit == 0:
                # Bit a insertar 0
                dqkt_block[get_indice(c)[0], get_indice(c)[1]] = 2*delta*round(abs(dqkt_block[get_indice(c)[0], get_indice(c)[1]])/(2.0*delta)) - delta/2.0
            else:
                # Bit a insertar 1
                dqkt_block[get_indice(c)[0], get_indice(c)[1]] = 2*delta*round(abs(dqkt_block[get_indice(c)[0], get_indice(c)[1]])/(2.0*delta)) + delta/2.0

            if negative:
                dqkt_block[get_indice(c)[0], get_indice(c)[1]] *= -1
            Y_array = DqKT().idqkt2(dqkt_block)
            block_ycbcr_array[:, :, 0] = Y_array
            # Convirtiendo a modelo de color RGB
            block_rgb_array = ImageTools().ycbcr2rgb(block_ycbcr_array)
            psnr_aux = Evaluations().PSNR_RGB(
                Image.open(block_path), Image.fromarray(block_rgb_array))
            if psnr_aux > psnr:
                psnr = psnr_aux
                result = {'c': c, 'delta': delta}
                watermarked_path = block_path[:-4] + 'w.png'
                Image.fromarray(block_rgb_array).save(watermarked_path)
    print(result)


def main():
    try:
        # Load cover image
        root = Tk()
        root.filename = filedialog.askopenfilename(
            initialdir="static/", title="Select file",
            filetypes=(
                ("png files", "*.png"), ("jpg files", "*.jpg"),
                ("all files", "*.*")))
        cover_image = Image.open(root.filename).convert('RGB')
        root.destroy()

    except Exception as e:
        root.destroy()
        print("Error: ", e)
        print("The image file was not loaded")

    # Instance a la clase Bloque
    cover_array = np.asarray(cover_image)
    blocks = BlocksImage(cover_array)

    # for i in range(blocks.max_num_blocks()):
    for i in range(10):
        block_array = blocks.get_block(i)
        # Save block image
        block_image = Image.fromarray(block_array, 'RGB')
        block_path = 'static/' + str(i) + '.png'
        block_image.save(block_path)
        # Clasificacion del block
        clasificador = clasificar(block_path)
        if clasificador not in clases:
            clases.append(clasificador)


if __name__ == '__main__':
    main()
