# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
from PIL import Image

import numpy as np
from scipy import misc

from evaluations.evaluations import Evaluations

from transforms.DqKT import DqKT
from transforms.DAT import DAT

from block_tools.blocks_class import BlocksImage

from qr_tools.MyQR62 import MyQR62

import pwlcm

import math

#PyTorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from pathlib import Path


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 8 * 8, 5000)
        self.fc2 = nn.Linear(5000, 9)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Clasification():
    
    def __init__(self):
        self.model = Net()
        checkpoint = torch.load(Path('data/fnn600_with_jpeg_20.pt'), map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    
    def predict(self, image_array):
        image = Image.fromarray(image_array)
        image_tensor = self.data_transforms(image)

        # PyTorch pretrained models expect the Tensor dims to be
        # (num input imgs, num color channels, height, width).
        # Currently however, we have (num color channels, height, width);
        # let's fix this by inserting a new axis.
        image_tensor = image_tensor.unsqueeze(0)

        output = self.model(Variable(image_tensor))

        return np.argmax(output.detach().numpy())


def binary2int(binary):
    # Devuelve el entero correspondiente a una lista de binarios
    n = len(binary)
    v = 0
    for i in range(n):
        v += (2**(n-i-1))*binary[i]
    return v


def get_dwt(chromosome):
    """
    Devuelve la subbanda de la DWT a utilizar (0, 1, 2, 3) -> (LL, LH, HL, HH)
    """
    return binary2int(chromosome[0:2])


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


def run_main():
    from image_tools.ImageTools import ImageTools
    dqkt = DqKT()
    eva = Evaluations()
    myqr = MyQR62()
    dat = DAT()
    itools = ImageTools()

    clasification = Clasification()

    delta = 128

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
    f_ber20 = open("ber20.txt", "w")
    f_ber20.close()
    f_ber50 = open("ber50.txt", "w")
    f_ber50.close()
    f_ber75 = open("ber75.txt", "w")
    f_ber75.close()
    f_berGueztli = open("berGuetzli.txt", "w")
    f_berGueztli.close()
    for db_img in db_images:
        f_psnr = open("psnr.txt", "a+")
        f_ber20 = open("ber20.txt", "a+")
        f_ber50 = open("ber50.txt", "a+")
        f_ber75 = open("ber75.txt", "a+")
        f_berGuetzli = open("berGuetzli.txt", "a+")
        cover_image_url = carpeta + db_img

        c = [1, 19]

        if (c[1]-c[0]) != 0:
            # Set
            print("Iniciando...")
            # Cargando imagen
            cover_image = Image.open(cover_image_url)
            # Convirtiendo a modelo de color YCbCr
            cover_ycbcr_array = itools.rgb2ycbcr(cover_image)

            cover_array = cover_ycbcr_array[:, :, 0]
            # Cargando watermark
            watermark = Image.open("static/Watermarking.png").convert("1")

            # Utilizando Arnold Transforms
            for i in range(20):
                watermark = dat.dat2(watermark)

            # obteniendo array de la watermark
            watermark_array = np.asarray(watermark)  # Array of watermark

            # Instance a la clase Bloque
            bt_of_cover = BlocksImage(cover_array)

            bt_of_rgb_cover = BlocksImage(misc.fromimage(cover_image))

            # Calculando e imprimeindo datos iniciales
            len_of_watermark = watermark_array.size
            print('Cantidad de bit a insertar: ', len_of_watermark)
            print('Cantidad de bloques del cover: ', bt_of_cover.max_num_blocks())

            # Datos de la watermark como lista
            list_bit_of_watermark = watermark_array.reshape((1, len_of_watermark))[0]

            print('Marca de agua como lista:')
            print(list_bit_of_watermark)

            print("Seleccionando bloques...")

            # # Utilizar Bloques segun key
            # dic = {'semilla': 0.00325687, 'p': 0.22415897}
            # valores = []
            # cantidad = bt_of_cover.max_blocks()
            # for i in range(cantidad):
            #     valores.append(i)
            # v = pwlcm.mypwlcm_limit(dic, valores, len_of_watermark)

            # Simulando pwlcm para AG
            import random
            v = []
            # cantidad = bt_of_cover.max_num_blocks()            
            # x, y = cover_image.size
            # izq_min = x//64
            # der_max = (x//8 - x//64)
            # arr_min = y//64
            # aba_max = (y//8 - y//64)
            # while len(v) < len_of_watermark:
            #     val = random.randrange(cantidad)
            #     columna = val//(x//8)
            #     fila = val - columna*(x//8)
            #     if val not in v:
            #         if columna > izq_min and columna < der_max and fila > arr_min and fila < aba_max:
            #             v.append(val)
            p = [i for i in range(bt_of_cover.max_num_blocks())]
            random.shuffle(p)        
            v = p[:len_of_watermark]

            print("Insertando...")
            # Marcar los self.len_of_watermark bloques
            for i in range(len_of_watermark):
                block = bt_of_cover.get_block(v[i])
                # Predict
                p = clasification.predict(bt_of_rgb_cover.get_block(v[i]))
                if p == 1:
                    c[1] = 17
                    delta = 90
                elif p == 4:
                    c[1] = 19
                    delta = 60
                elif p == 5:
                    c[1] = 20
                    delta = 130
                elif p == 7:
                    c[1] = 28
                    delta = 94
                elif p == 8:
                    c[1] = 34
                    delta = 130
                else:
                    c[1] = 19
                    delta = 130

                dqkt_block = dqkt.dqkt2(block)

                negative = False
                if dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] < 0:
                    negative = True

                if list_bit_of_watermark[i % len_of_watermark] == 0:
                    # Bit a insertar 0
                    dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] = 2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) - delta/2.0
                else:
                    # Bit a insertar 1
                    dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] = 2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) + delta/2.0

                if negative:
                    dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] *= -1
                idqkt_block = dqkt.idqkt2(dqkt_block)
                inv = idqkt_block
                for x in range(8):
                    for y in range(8):
                        if (inv[x, y] - int(inv[x, y])) < 0.5:
                            inv[x, y] = int(inv[x, y])
                        else:
                            inv[x, y] = int(inv[x, y]) + 1
                        if inv[x, y] > 255:
                            inv[x, y] = 255
                        if inv[x, y] < 0:
                            inv[x, y] = 0
                bt_of_cover.set_block(idqkt_block, v[i])
            
            cover_marked_ycbcr_array = cover_ycbcr_array

            image_rgb_array = itools.ycbcr2rgb(cover_marked_ycbcr_array)
            watermarked_image_without_noise = Image.fromarray(image_rgb_array)

            # Almacenando imagen marcada
            watermarked_image_without_noise.save(
                "static/experimento/watermarked_" + db_img[:10] + "_without_noise.jpg")

            # Calculando PSNR
            print("Calculando PSNR...")
            watermarked_image_without_noise = Image.open(
                "static/experimento/watermarked_" + db_img[:10] + "_without_noise.jpg")
            cover_image = Image.open(cover_image_url)
            # cover_array = misc.fromimage(cover_image)
            psnr_img_watermarked_without_noise = eva.PSNR_RGB(
                cover_image, watermarked_image_without_noise)
            f_psnr.write("%f," % (psnr_img_watermarked_without_noise))

            # Aplicando ruido JPEG 20, 50, 75 % y Guetzli
            print("Aplicando ruido JPEG20, JPEG50 JPEG75 y Guetzli")
            watermarked_image_without_noise.save(
                "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg20.jpg",
                quality=25, optimice=True)
            watermarked_image_without_noise.save(
                "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg50.jpg",
                quality=50, optimice=True)
            watermarked_image_without_noise.save(
                "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg75.jpg",
                quality=75, optimice=True)

            # Falta guetzli

            # Extrayendo de jpeg20
            print("Extrayendo de JPEG20")
            watermarked_image_with_noise = Image.open(
                "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg20.jpg")
            bt_of = BlocksImage(misc.fromimage(watermarked_image_with_noise))    

            # Convirtiendo a modelo de color YCbCr
            watermarked_ycbcr_image_with_noise = itools.rgb2ycbcr(
                watermarked_image_with_noise)
            watermarked_image_with_noise_array = watermarked_ycbcr_image_with_noise[:, :, 0]

            bt_of_watermarked_image_with_noise = BlocksImage(
                watermarked_image_with_noise_array)

            extract = []

            # for i in range(bt.max_blocks()):  # Recorrer todos los bloques de la imagen
            for i in range(len(list_bit_of_watermark)):  # Recorrer los primeros len(list_bit_of_watermark) bloques
                block = bt_of_watermarked_image_with_noise.get_block(v[i])
                # Predict
                p = clasification.predict(bt_of.get_block(v[i]))
                if p == 1:
                    c[1] = 17
                    delta = 90
                elif p == 4:
                    c[1] = 19
                    delta = 60
                elif p == 5:
                    c[1] = 20
                    delta = 130
                elif p == 7:
                    c[1] = 28
                    delta = 94
                elif p == 8:
                    c[1] = 34
                    delta = 130
                else:
                    c[1] = 19
                    delta = 130
                dqkt_block = dqkt.dqkt2(np.array(block, dtype=np.float32))
                negative = False
                if dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] < 0:
                    negative = True

                C1 = (2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) + delta/2.0) - abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])
                C0 = (2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) - delta/2.0) - abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])

                if negative:
                    C1 *= -1
                    C0 *= -1
                if C0 < C1:
                    extract.append(0)
                else:
                    extract.append(1)

            wh = int(math.sqrt(len_of_watermark))
            extract_image1 = Image.new("1", (wh, wh), 255)
            array_extract_image1 = misc.fromimage(extract_image1)

            for i in range(wh):
                for y in range(wh):
                    if extract[wh*i+y] == 0:
                        array_extract_image1[i, y] = 0

            myqr1 = MyQR62()

            watermark_array_image = misc.toimage(array_extract_image1)
            for i in range(10):
                watermark_array_image = dat.dat2(watermark_array_image)
            # array = misc.fromimage(watermark_array_image)
            # watermark_extracted = misc.toimage(
            #     myqr1.get_resconstructed(array))

            watermark_extracted = watermark_array_image

            watermark_extracted.save(
                "static/experimento/watermark_" + db_img[:10] + "_with_jpeg20.png")

            # b = BlocksImage(misc.fromimage(watermark_extracted), 2, 2)
            # for m in range(b.max_num_blocks()):
            #     b.set_color(m)
            # misc.toimage(b.get()).save(
            #     "static/experimento/watermark_" + db_img[:10] + "_with_jpeg20_resconstructed.png")

            # watermark_extracted_reconstructed = Image.open(
            #     "static/experimento/watermark_" + db_img[:10] + "_with_jpeg20_resconstructed.png")

            # Temporal
            watermark_extracted_reconstructed = watermark_extracted

            print("Calculando BER with noise JPEG20")
            # Cargando watermark
            watermark = Image.open("static/Watermarking.png").convert("1")
            ber_with_noise_jpeg20 = eva.BER_A(
                misc.fromimage(watermark),
                misc.fromimage(watermark_extracted_reconstructed))
            f_ber20.write("%f," % (ber_with_noise_jpeg20))

            # Extrayendo de jpeg50
            print("Extrayendo de JPEG50")
            watermarked_image_with_noise = Image.open(
                "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg50.jpg")
            bt_of = BlocksImage(misc.fromimage(watermarked_image_with_noise))    

            # Convirtiendo a modelo de color YCbCr
            watermarked_ycbcr_image_with_noise = itools.rgb2ycbcr(
                watermarked_image_with_noise)
            watermarked_image_with_noise_array = watermarked_ycbcr_image_with_noise[:, :, 0]

            bt_of_watermarked_image_with_noise = BlocksImage(
                watermarked_image_with_noise_array)

            extract = []

            # for i in range(bt.max_blocks()):  # Recorrer todos los bloques de la imagen
            for i in range(len(list_bit_of_watermark)):  # Recorrer los primeros len(list_bit_of_watermark) bloques

                block = bt_of_watermarked_image_with_noise.get_block(v[i])
                # Predict
                p = clasification.predict(bt_of.get_block(v[i]))
                if p == 1:
                    c[1] = 17
                    delta = 90
                elif p == 4:
                    c[1] = 19
                    delta = 60
                elif p == 5:
                    c[1] = 20
                    delta = 130
                elif p == 7:
                    c[1] = 28
                    delta = 94
                elif p == 8:
                    c[1] = 34
                    delta = 130
                else:
                    c[1] = 19
                    delta = 130
                dqkt_block = dqkt.dqkt2(np.array(block, dtype=np.float32))
                negative = False
                if dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] < 0:
                    negative = True

                C1 = (2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) + delta/2.0) - abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])
                C0 = (2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) - delta/2.0) - abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])

                if negative:
                    C1 *= -1
                    C0 *= -1
                if C0 < C1:
                    extract.append(0)
                else:
                    extract.append(1)

            wh = int(math.sqrt(len_of_watermark))
            extract_image1 = Image.new("1", (wh, wh), 255)
            array_extract_image1 = misc.fromimage(extract_image1)

            for i in range(wh):
                for y in range(wh):
                    if extract[wh*i+y] == 0:
                        array_extract_image1[i, y] = 0

            myqr1 = MyQR62()

            watermark_array_image = misc.toimage(array_extract_image1)
            for i in range(10):
                watermark_array_image = dat.dat2(watermark_array_image)
            # array = misc.fromimage(watermark_array_image)
            # watermark_extracted = misc.toimage(
            #     myqr1.get_resconstructed(array))

            watermark_extracted.save(
                "static/experimento/watermark_" + db_img[:10] + "_with_jpeg50.png")

            # b = BlocksImage(misc.fromimage(watermark_extracted), 2, 2)
            # for m in range(b.max_num_blocks()):
            #     b.set_color(m)
            # misc.toimage(b.get()).save(
            #     "static/experimento/watermark_" + db_img[:10] + "_with_jpeg50_resconstructed.png")

            # watermark_extracted_reconstructed = Image.open(
            #     "static/experimento/watermark_" + db_img[:10] + "_with_jpeg50_resconstructed.png")

            # Temporal
            watermark_extracted_reconstructed = watermark_extracted

            print("Calculando BER with noise JPEG50")
            # Cargando watermark
            watermark = Image.open("static/Watermarking.png").convert("1")
            ber_with_noise_jpeg50 = eva.BER_A(
                misc.fromimage(watermark),
                misc.fromimage(watermark_extracted_reconstructed))
            f_ber50.write("%f," % (ber_with_noise_jpeg50))

            # Extrayendo de jpeg75
            print("Extrayendo de JPEG75")
            watermarked_image_with_noise = Image.open(
                "static/experimento/watermarked_" + db_img[:10] + "_with_jpeg75.jpg")
            bt_of = BlocksImage(misc.fromimage(watermarked_image_with_noise))

            # Convirtiendo a modelo de color YCbCr
            watermarked_ycbcr_image_with_noise = itools.rgb2ycbcr(
                watermarked_image_with_noise)
            watermarked_image_with_noise_array = watermarked_ycbcr_image_with_noise[:, :, 0]

            bt_of_watermarked_image_with_noise = BlocksImage(
                watermarked_image_with_noise_array)

            extract = []

            for i in range(len(list_bit_of_watermark)):  # Recorrer los primeros len(list_bit_of_watermark) bloques

                block = bt_of_watermarked_image_with_noise.get_block(v[i])
                # Predict
                p = clasification.predict(bt_of.get_block(v[i]))
                if p == 1:
                    c[1] = 17
                    delta = 90
                elif p == 4:
                    c[1] = 19
                    delta = 60
                elif p == 5:
                    c[1] = 20
                    delta = 130
                elif p == 7:
                    c[1] = 28
                    delta = 94
                elif p == 8:
                    c[1] = 34
                    delta = 130
                else:
                    c[1] = 19
                    delta = 130
                dqkt_block = dqkt.dqkt2(np.array(block, dtype=np.float32))
                negative = False
                if dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]] < 0:
                    negative = True

                C1 = (2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) + delta/2.0) - abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])
                C0 = (2*delta*round(abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])/(2.0*delta)) - delta/2.0) - abs(dqkt_block[get_indice(c[1])[0], get_indice(c[1])[1]])

                if negative:
                    C1 *= -1
                    C0 *= -1
                if C0 < C1:
                    extract.append(0)
                else:
                    extract.append(1)

            wh = int(math.sqrt(len_of_watermark))
            extract_image1 = Image.new("1", (wh, wh), 255)
            array_extract_image1 = misc.fromimage(extract_image1)

            for i in range(wh):
                for y in range(wh):
                    if extract[wh*i+y] == 0:
                        array_extract_image1[i, y] = 0

            myqr1 = MyQR62()

            watermark_array_image = misc.toimage(array_extract_image1)
            for i in range(10):
                watermark_array_image = dat.dat2(watermark_array_image)
            # array = misc.fromimage(watermark_array_image)
            # watermark_extracted = misc.toimage(
            #     myqr1.get_resconstructed(array))

            watermark_extracted.save(
                "static/experimento/watermark_" + db_img[:10] + "_with_jpeg75.png")

            # b = BlocksImage(misc.fromimage(watermark_extracted), 2, 2)
            # for m in range(b.max_num_blocks()):
            #     b.set_color(m)
            # misc.toimage(b.get()).save(
            #     "static/experimento/watermark_" + db_img[:10] + "_with_jpeg75_resconstructed.png")

            # watermark_extracted_reconstructed = Image.open(
            #     "static/experimento/watermark_" + db_img[:10] + "_with_jpeg75_resconstructed.png")

            # Temporal
            watermark_extracted_reconstructed = watermark_extracted

            print("Calculando BER with noise JPEG75")
            # Cargando watermark
            watermark = Image.open("static/Watermarking.png").convert("1")
            ber_with_noise_jpeg75 = eva.BER_A(
                misc.fromimage(watermark),
                misc.fromimage(watermark_extracted_reconstructed))
            f_ber75.write("%f," % (ber_with_noise_jpeg75))
            print('*********************************************************')

        f_psnr.close()
        f_ber20.close()
        f_ber50.close()
        f_ber75.close()
        f_berGuetzli.close()


if __name__ == "__main__":
    run_main()
