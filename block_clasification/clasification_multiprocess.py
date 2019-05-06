# -*- coding: utf-8 -*-
# For the cluster

import os

from PIL import Image
import numpy as np

from helpers.blocks_class import BlocksImage
from helpers.image_tools import ImageTools
from helpers.DqKT import DqKT
from helpers.evaluations import Evaluations

import random

import glob

from multiprocessing import Pool


clases = {}

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

def marcar(block_array, watermark, coef, delta):
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
    
    return block_rgb_array

def extraer(block_array, coef, delta):
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


def procesar(block_path, watermark_bit, coef, delta):
    # Cargando imagen
    block_image = Image.open(block_path)
    # Convirtiendola a ndarray
    block_array = np.asarray(block_image)
    block_rgb_array = marcar(block_array, watermark_bit, coef, delta)
    watermarked_image_without_noise = Image.fromarray(block_rgb_array)
    # Calculando el PSNR
    psnr_img_watermarked_without_noise = Evaluations().PSNR_RGB(
        Image.open(block_path), watermarked_image_without_noise)
    
    # Extrayendo watermark bit
    # Convirtiendola a ndarray
    block_array = np.asarray(watermarked_image_without_noise)
    extract_without_noise = extraer(block_array, coef, delta)

    if watermark_bit == extract_without_noise:
        ber_without_noise = 1
    else:
        ber_without_noise = 0

    # Aplicando ruido
    watermarked_with_noise_path = block_path[:-4] + 'wnoised.jpg'
    watermarked_image_without_noise.save(
        watermarked_with_noise_path, quality=20, optimice=True)
    
    # Extrayendo watermark bit
    watermarked_image_with_noise = Image.open(
        watermarked_with_noise_path)
    # Convirtiendola a ndarray
    block_array = np.asarray(watermarked_image_with_noise)
    extract_with_noise = extraer(block_array, coef, delta)

    if watermark_bit == extract_with_noise:
        ber_with_noise = 1
    else:
        ber_with_noise = 0
    
    return {
        'psnr_img_watermarked_without_noise': psnr_img_watermarked_without_noise,
        'ber_without_noise': ber_without_noise,
        'ber_with_noise': ber_with_noise}


def clasificar(block_path):
    
    result = {'c': 0, 'delta': 1}
    score = 0.0
    for c in range(34):
        coef = c + 16        
        for d in range(100):
            delta = d + 30
            # Marcando el bloque
            result0 = procesar(block_path, 0, coef, delta)
            result1 = procesar(block_path, 1, coef, delta)

            # Promedio de datos
            psnr_img_watermarked_without_noise = (
                result0['psnr_img_watermarked_without_noise'] + result0['psnr_img_watermarked_without_noise'])/2
            ber_without_noise = (
                result0['ber_without_noise']+result1['ber_without_noise'])/2.0
            ber_with_noise = (
                result0['ber_with_noise']+result1['ber_with_noise'])/2.0

            # Score
            score_aux = (
                psnr_img_watermarked_without_noise/160 + ber_without_noise + ber_with_noise)/3
            if score_aux > score:
                score = score_aux
                result['c'] = coef
                result['delta'] = delta
                result['psnr'] = psnr_img_watermarked_without_noise
                if ber_without_noise == 1.0:
                    result['extract_without_noise_true'] = True
                else:
                    result['extract_without_noise_true'] = False
                if ber_with_noise == 1.0:
                    result['extract_with_noise_true'] = True
                else:
                    result['extract_with_noise_true'] = False
                result['score'] = score
                if result['extract_without_noise_true'] and  result['extract_with_noise_true']:
                    break
    return result


def is_in_clases(lista):
    for item in clases:
        if lista == clases[item]:
            return item
    return None


def sprint(path):
    cover_image = Image.open(path).convert('RGB')
    # Creando path para almacenar los bloques de esta imagen
    b_path = 'static/' + path.split('/')[-1][:-4]+'/'
    try:
        os.stat(b_path)
    except Exception:
        os.mkdir(b_path)

    # Instance a la clase Bloque
    cover_array = np.asarray(cover_image)
    blocks = BlocksImage(cover_array)
    random_blocks = [i for i in range(blocks.max_num_blocks())]
    random.shuffle(random_blocks)
    for i in range(blocks.max_num_blocks()):
        block_array = blocks.get_block(random_blocks[i])
        # Save block image
        block_image = Image.fromarray(block_array, 'RGB')
        block_path = b_path + str(random_blocks[i]) + '.png'
        try:
            os.stat(block_path)
            print("Ya existe: ", block_path)
        except Exception:
            block_image.save(block_path)
            # Clasificacion del block
            clasificador = clasificar(block_path)
            # print(clasificador)
            class_path = b_path + str(clasificador['c']) + '_' + str(clasificador['delta']) + '/'
            if len(clases) == 0:
                # Add como clase 1            
                os.mkdir(class_path)
                clases['1'] = [clasificador['c'], clasificador['delta']]
                block_image = Image.open(block_path).save(class_path + str(random_blocks[i]) + '.png') 
            elif is_in_clases([clasificador['c'], clasificador['delta']]):
                # Add a la clase correspondiente
                block_image = Image.open(block_path).save(class_path + str(random_blocks[i]) + '.png')
            else:
                # Add clase
                clases[len(clases)+1] = [clasificador['c'], clasificador['delta']]
                os.mkdir(class_path)
                block_image = Image.open(block_path).save(class_path + str(random_blocks[i]) + '.png')


def main():
    
    # Load cover images
    paths = glob.glob('static/Dataset/*.bmp')   

    pool = Pool(processes=8)

    pool.map(sprint, paths)

    


if __name__ == '__main__':
    main()
