# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
from PIL import Image

import numpy as np
from scipy import misc

from evaluations.evaluations import Evaluations
from block_tools.blocks_class import BlocksImage
from Shivani2017R import Shivani2017R
import pwlcm
import math
from pathlib import Path


def run_main():
    eva = Evaluations()
    wm = Shivani2017R()

    db_images = [
        'csg562-003.jpg', 'csg562-004.jpg', 'csg562-005.jpg', 'csg562-006.jpg',
        'csg562-007.jpg', 'csg562-008.jpg', 'csg562-009.jpg', 'csg562-010.jpg',
        'csg562-011.jpg', 'csg562-012.jpg', 'csg562-013.jpg', 'csg562-014.jpg',
        'csg562-015.jpg', 'csg562-016.jpg', 'csg562-017.jpg', 'csg562-018.jpg',
        'csg562-019.jpg', 'csg562-020.jpg', 'csg562-021.jpg', 'csg562-022.jpg',
        'csg562-023.jpg', 'csg562-024.jpg', 'csg562-025.jpg', 'csg562-026.jpg',
        'csg562-027.jpg', 'csg562-028.jpg', 'csg562-029.jpg', 'csg562-030.jpg',
        'csg562-031.jpg', 'csg562-032.jpg', 'csg562-033.jpg', 'csg562-034.jpg',
        'csg562-035.jpg', 'csg562-036.jpg', 'csg562-037.jpg', 'csg562-038.jpg',
        'csg562-039.jpg', 'csg562-040.jpg', 'csg562-041.jpg', 'csg562-042.jpg',
        'csg562-043.jpg', 'csg562-044.jpg', 'csg562-045.jpg', 'csg562-046.jpg',
        'csg562-047.jpg', 'csg562-048.jpg', 'csg562-049.jpg', 'csg562-050.jpg',
        'csg562-054.jpg', 'csg562-055.jpg', 'csg562-056.jpg', 'csg562-057.jpg',
        'csg562-058.jpg', 'csg562-059.jpg', 'csg562-060.jpg', 'csg562-061.jpg',
        'csg562-062.jpg', 'csg562-063.jpg', 'csg562-064.jpg', 'csg562-065.jpg'
    ]

    carpeta = 'static/dataset/'
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
    for db_img in db_images:
        f_psnr = open("psnr.txt", "a+")
        f_ber25 = open("ber25.txt", "a+")
        f_ber50 = open("ber50.txt", "a+")
        f_ber75 = open("ber75.txt", "a+")
        f_berWithout = open("berWithout.txt", "a+")
        f_berGuetzli = open("berGuetzli.txt", "a+")
        
        cover_image_url = carpeta + db_img

        # Set
        print("Iniciando...")
        # Cargando imagen
        cover_image = Image.open(cover_image_url)

        # Insert
        watermarked_image_without_noise = wm.insert(cover_image)
        
        # Almacenando imagen marcada
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + db_img[:10] + "_without_noise.png")

        # Calculando PSNR
        print("Calculando PSNR...")
        watermarked_image_without_noise = Image.open(
            "static/experimento/watermarked_" + db_img[:10] + "_without_noise.png")
        cover_image = Image.open(cover_image_url)

        psnr_img_watermarked_without_noise = eva.PSNR_RGB(
            cover_image, watermarked_image_without_noise)
        f_psnr.write("%f," % (psnr_img_watermarked_without_noise))

        # Aplicando ruido JPEG 25, 50, 75 % y Guetzli
        print("Aplicando ruido JPEG20, JPEG50 JPEG75 y Guetzli")
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg25.jpg",
            quality=25, optimice=True)
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg50.jpg",
            quality=50, optimice=True)
        watermarked_image_without_noise.save(
            "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg75.jpg",
            quality=75, optimice=True)

        # Falta guetzli

        ##########################################################
        # Extrayendo de Without
        print("Extrayendo de Without")
        watermarked_image_with_noise = Image.open(
            "static/experimento/watermarked_" + db_img[:10] + "_without_noise.png")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + db_img[:10] + "_without_noise.png")

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
            "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg25.jpg")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + db_img[:10] + "_with_jpeg25.png")

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
            "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg50.jpg")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + db_img[:10] + "_with_jpeg50.png")

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
            "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg75.jpg")
        
        # Watermark extracting
        watermark_extracted = wm.extract(watermarked_image_with_noise)

        watermark_extracted.save(
            "static/experimento/watermark_" + db_img[:10] + "_with_jpeg75.png")

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
