# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
from PIL import Image

import numpy as np
from scipy import misc

from evaluations.evaluations import Evaluations
from block_tools.blocks_class import BlocksImage
from AvilaDomenech2019R import AvilaDomenech2019R
import pwlcm
import math
from pathlib import Path
import glob


def image_name(path):
    a = path.split('/')[-1]
    return a[:-4]


def run_main():
    eva = Evaluations()
    wm = AvilaDomenech2019R('password')

    image_paths = glob.glob('static/dataset/*.jpg')

    # Fichero a guardar resultados
    f_psnr = open("psnr.txt", "w")
    f_psnr.close()
    f_ber25 = open("ber25.txt", "w")
    f_ber25.close()
    f_ber50 = open("ber50.txt", "w")
    f_ber50.close()
    f_ber75 = open("ber75.txt", "w")
    f_ber75.close()
    f_berWithout = open("berWithout.txt", "w")
    f_berWithout.close()
    f_berGueztli = open("berGuetzli.txt", "w")
    f_berGueztli.close()
    for cover_image_url in image_paths:
        name = image_name(cover_image_url)
        f_psnr = open("psnr.txt", "a+")
        f_ber25 = open("ber25.txt", "a+")
        f_ber50 = open("ber50.txt", "a+")
        f_ber75 = open("ber75.txt", "a+")
        f_berWithout = open("berWithout.txt", "a+")
        f_berGuetzli = open("berGuetzli.txt", "a+")

        # Set
        print("Iniciando...")
        # Cargando imagen
        cover_image = Image.open(cover_image_url)

        # Insert
        watermarked_image_without_noise = wm.insert(cover_image)
        
        # Almacenando imagen marcada
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + name + "_without_noise.png")

        # Calculando PSNR
        print("Calculando PSNR...")
        watermarked_image_without_noise = Image.open(
            "static/experimento/watermarked_" + name + "_without_noise.png")
        cover_image = Image.open(cover_image_url)

        psnr_img_watermarked_without_noise = eva.PSNR_RGB(
            cover_image, watermarked_image_without_noise)
        f_psnr.write("%f," % (psnr_img_watermarked_without_noise))

        # Aplicando ruido JPEG 25, 50, 75 % y Guetzli
        print("Aplicando ruido JPEG20, JPEG50 JPEG75 y Guetzli")
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + name + "_with_jpeg25.jpg",
            quality=25, optimice=True)
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + name + "_with_jpeg50.jpg",
            quality=50, optimice=True)
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + name + "_with_jpeg75.jpg",
            quality=75, optimice=True)

        # Falta guetzli

        ##########################################################
        # Extrayendo de Without
        print("Extrayendo de Without")
        watermarked_image_with_noise = Image.open(
            "static/experimento/watermarked_" + name + "_without_noise.png")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + name + "_without_noise.png")

        print("Calculando BER without noise")
        # Cargando watermark
        watermark = wm.watermark
        ber_without_noise= eva.BER_A(
            misc.fromimage(watermark),
            misc.fromimage(watermark_extracted))
        f_berWithout.write("%f," % (ber_without_noise))

        ##########################################################
        # Extrayendo de jpeg25
        print("Extrayendo de JPEG25")
        watermarked_image_with_noise = Image.open(
            "static/experimento/watermarked_" + name + "_with_jpeg25.jpg")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + name + "_with_jpeg25.png")

        print("Calculando BER with noise JPEG25")
        # Cargando watermark
        watermark = wm.watermark
        ber_with_noise_jpeg25 = eva.BER_A(
            misc.fromimage(watermark),
            misc.fromimage(watermark_extracted))
        f_ber25.write("%f," % (ber_with_noise_jpeg25))
        
        ################################################################
        # Extrayendo de jpeg50
        print("Extrayendo de JPEG50")
        watermarked_image_with_noise = Image.open(
            "static/experimento/watermarked_" + name + "_with_jpeg50.jpg")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + name + "_with_jpeg50.png")

        print("Calculando BER with noise JPEG50")
        # Cargando watermark
        watermark = wm.watermark
        ber_with_noise_jpeg50 = eva.BER_A(
            misc.fromimage(watermark),
            misc.fromimage(watermark_extracted))
        f_ber50.write("%f," % (ber_with_noise_jpeg50))

        #############################################################
        # Extrayendo de jpeg75
        print("Extrayendo de JPEG75")
        watermarked_image_with_noise = Image.open(
            "static/experimento/watermarked_" + name + "_with_jpeg75.jpg")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + name + "_with_jpeg75.png")

        print("Calculando BER with noise JPEG75")
        # Cargando watermark
        watermark = wm.watermark
        ber_with_noise_jpeg75 = eva.BER_A(
            misc.fromimage(watermark),
            misc.fromimage(watermark_extracted))
        f_ber75.write("%f," % (ber_with_noise_jpeg75))

        print('******************************************************')

        f_psnr.close()
        f_ber25.close()
        f_ber50.close()
        f_ber75.close()
        f_berWithout.close()
        f_berGuetzli.close()


if __name__ == "__main__":
    run_main()
