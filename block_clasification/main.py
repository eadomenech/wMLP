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

def marcar(block_path, watermark, coef, delta):
    # Cargando imagen
    block_image = Image.open(block_path)
    # Convirtiendola a ndarray
    block_array = np.asarray(block_image)
    # Convirtiendo a modelo de color YCbCr
    block_ycbcr_array = ImageTools().rgb2ycbcr(block_array)
    # Y component
    Y_array = block_ycbcr_array[:, :, 0]
    # Krawtchouk transform
    dqkt_block = DqKT().dqkt2(np.array(Y_array, dtype=np.float32))
    
    negative = False
    if dqkt_block[get_indice(coef)[0], get_indice(coef)[1]] < 0:
        negative = True

    if watermark == 0:
        # Bit a insertar 0
        dqkt_block[get_indice(coef)[0], get_indice(coef)[1]] = 2*delta*round(abs(dqkt_block[get_indice(coef)[0], get_indice(coef)[1]])/(2.0*delta)) - delta/2.0
    else:
        # Bit a insertar 1
        dqkt_block[get_indice(coef)[0], get_indice(coef)[1]] = 2*delta*round(abs(dqkt_block[get_indice(coef)[0], get_indice(coef)[1]])/(2.0*delta)) + delta/2.0

    if negative:
        dqkt_block[get_indice(coef)[0], get_indice(coef)[1]] *= -1
    
    Y_array = DqKT().idqkt2(dqkt_block)
    block_ycbcr_array[:, :, 0] = Y_array
    # Convirtiendo a modelo de color RGB
    block_rgb_array = ImageTools().ycbcr2rgb(block_ycbcr_array)
    watermarked_image_without_noise = Image.fromarray(block_rgb_array)
    return watermarked_image_without_noise

def extraer(block_path, coef, delta):
    # Cargando imagen
    block_image = Image.open(block_path)
    # Convirtiendola a ndarray
    block_array = np.asarray(block_image)
    # Convirtiendo a modelo de color YCbCr
    block_ycbcr_array = ImageTools().rgb2ycbcr(block_array)
    # Y component
    Y_array = block_ycbcr_array[:, :, 0]
    # Krawtchouk transform
    dqkt_block = DqKT().dqkt2(np.array(Y_array, dtype=np.float32))
    
    negative = False
    if dqkt_block[get_indice(coef)[0], get_indice(coef)[1]] < 0:
        negative = True
    
    C1 = (2*delta*round(abs(dqkt_block[get_indice(coef)[0], get_indice(coef)[1]])/(2.0*delta)) + delta/2.0) - abs(dqkt_block[get_indice(coef)[0], get_indice(coef)[1]])
    C0 = (2*delta*round(abs(dqkt_block[get_indice(coef)[0], get_indice(coef)[1]])/(2.0*delta)) - delta/2.0) - abs(dqkt_block[get_indice(coef)[0], get_indice(coef)[1]])

    if negative:
        C1 *= -1
        C0 *= -1
    if C0 < C1:
        return 0
    else:
        return 1


def clasificar(block_path):
    
    # Bit a marcar
    if np.random.rand() > 0.5:
        watermark_bit = 1
    else:
        watermark_bit = 0
    result = {'c': 0, 'delta': 1}
    score = 0.0
    for c in range(40):
        coef = c + 10        
        for d in range(100):
            delta = d + 1
            # Marcando el bloque
            watermarked_image_without_noise = marcar(block_path, watermark_bit, coef, delta)
            # Calculando el PSNR
            psnr_img_watermarked_without_noise = Evaluations().PSNR_RGB(
                Image.open(block_path), watermarked_image_without_noise)
            print("The PSNR with c: %d and delta: %d is: %f" %(coef, delta, psnr_img_watermarked_without_noise))
            
            # Aplicando ruido
            watermarked_with_noise_path = block_path[:-4] + 'wnoised.jpg'
            watermarked_image_without_noise.save(
                watermarked_with_noise_path, quality=25, optimice=True)
            
            # Extrayendo watermark bit
            extract = extraer(watermarked_with_noise_path, coef, delta)

            if watermark_bit == extract:
                ber_with_noise = 1
            else:
                ber_with_noise = 0

            # Score
            score_aux = (psnr_img_watermarked_without_noise/100 + ber_with_noise)/2
            if score_aux > score:
                score = score_aux
                result['c'] = coef
                result['delta'] = delta
                result['psnr'] = psnr_img_watermarked_without_noise
                if ber_with_noise:
                    result['extract_true'] = True
                else:
                    result['extract_true'] = False
                result['score'] = score
                watermarked_path = block_path[:-4] + 'w.png'
                watermarked_image_without_noise.save(watermarked_path)
    return result


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
    for i in range(20):
        block_array = blocks.get_block(i)
        # Save block image
        block_image = Image.fromarray(block_array, 'RGB')
        block_path = 'static/original_blocks/' + str(i) + '.png'
        block_image.save(block_path)
        # Clasificacion del block
        clasificador = clasificar(block_path)
        print(clasificador)
        if clasificador not in clases:
            clases.append(clasificador)


if __name__ == '__main__':
    main()
