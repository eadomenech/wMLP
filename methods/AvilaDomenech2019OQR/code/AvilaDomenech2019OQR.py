# -*- coding: utf-8 -*-
from block_tools.blocks_class import BlocksImage
from image_tools.ImageTools import ImageTools
from helpers import utils
from qr_tools.MyQR62 import MyQR62

from transforms.DAT import DAT
from transforms.DqKT import DqKT

from PIL import Image
from pathlib import Path
from scipy import misc
import numpy as np
import random
import math

#PyTorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable


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


class AvilaDomenech2019OQR():
    """
    Método de marca de agua digital robusta que utiliza las caracteristicas de los QR code para obtener mejores valores de BER. Se utiliza un MLP para clasificar los bloques y obtener los valores coeficiente y delta con los que se obtienen los mejores valores segun metricas utilizadas 
    """
    def __init__(self, key):
        
        self.key = key
        
        # Hash of key
        self.binary_hash_key = utils.md5Binary(self.key)

        # Cargando watermark
        self.watermark = Image.open("static/Watermarking.png").convert("1")

        # Obteniendo array de la watermark
        watermark_array = np.asarray(self.watermark)

        # Datos de la watermark como lista
        watermark_as_list = watermark_array.reshape(
            (1, watermark_array.size))[0]

        # Instancia a QR62
        myqr = MyQR62()
        
        # Tomando solo los valores correspondientes a los datos
        self.watermark_list = []
        for p in myqr.get_pos():
            if watermark_as_list[p]:
                self.watermark_list.append(0)
            else:
                self.watermark_list.append(255)
        for p in range(8):
            self.watermark_list.append(0)
        
        # Convertir a matriz cuadrada
        true_watermark_array = np.reshape(
            np.asarray(self.watermark_list), (38, 38))

        # Obtener imagen correspondiente
        true_watermark_image = misc.toimage(true_watermark_array)

        # Utilizando Arnold Transforms
        # Para imagen 38x38 el periodo es 40
        for i in range(30):
            true_watermark_image = DAT().dat2(true_watermark_image)

        true_watermark_array_scambred = np.asarray(true_watermark_image)

        # Datos de la watermark como lista
        self.len_watermark_list = true_watermark_array_scambred.size
        self.watermark_list = true_watermark_array_scambred.reshape((1, self.len_watermark_list))[0]

        # Posiciones seleccionadas
        self.pos = []
    
    def generar(self, maximo):
        '''Genera posiciones a utilizar en el marcado'''
        assert self.len_watermark_list <= maximo
        p = [i for i in range(maximo)]
        random.shuffle(p)        
        self.pos = p[:self.len_watermark_list]

        # # Utilizar Bloques segun key
        # dic = {'semilla': 0.00325687, 'p': 0.22415897}
        # valores = []
        # cantidad = bt_of_cover.max_blocks()
        # for i in range(cantidad):
        #     valores.append(i)
        # v = pwlcm.mypwlcm_limit(dic, valores, len_of_watermark)
    
    def clasificar(self, num_block, cover_image):
        # Instancia necesaria
        clasification = Clasification()
        bt_of_rgb_cover = BlocksImage(misc.fromimage(cover_image))
        # Predict
        p = clasification.predict(
            bt_of_rgb_cover.get_block(num_block))
        if p == 1:
            return (17, 90)
        elif p == 4:
            return (19, 60)
        elif p == 5:
            return (20, 130)
        elif p == 7:
            return (28, 94)
        elif p == 8:
            return (34, 130)
        else:
            return (19, 130)
    
    def zigzag(self, n):
        indexorder = sorted(((x, y) for x in range(n) for y in range(n)), key=lambda s: (s[0]+s[1], -s[1] if (s[0]+s[1]) % 2 else s[1]))
        return {index: n for n, index in enumerate(indexorder)}


    def get_indice(self, m):
        zarray = self.zigzag(8)
        px = -1
        py = -1
        n = int(len(zarray) ** 0.5 + 0.5)
        for x in range(n):
            for y in range(n):
                if zarray[(x, y)] == m:
                    px = x
                    py = y
        return (px, py)

    def insert(self, cover_image):
        
        print("...Proceso de insercion...")

        # Instancias necesarias
        itools = ImageTools()

        print("Convirtiendo a YCbCr")
        # Convirtiendo a modelo de color YCbCr
        cover_ycbcr_array = itools.rgb2ycbcr(cover_image)

        # Obteniendo componente Y
        cover_array = cover_ycbcr_array[:, :, 0]

        # Objeto de la clase Bloque
        bt_of_cover = BlocksImage(cover_array)

        # Generando bloques a marcar
        print("Generando bloques a marcar")
        self.generar(bt_of_cover.max_num_blocks())

        print("Marcando")
        # Marcar los self.len_watermark_list bloques
        for i in range(self.len_watermark_list):
            block = bt_of_cover.get_block(self.pos[i])

            # Valores de coeficiente y delta optimo para el bloque
            (c, delta) = self.clasificar(self.pos[i], cover_image) 

            # Calculando Krawchout Transform
            dqkt_block = DqKT().dqkt2(block)

            negative = False
            (px, py) = self.get_indice(c)
            if dqkt_block[px, py] < 0:
                negative = True

            if self.watermark_list[i % self.len_watermark_list] == 0:
                # Bit a insertar 0
                dqkt_block[px, py] = 2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) + delta/2.0
            else:
                # Bit a insertar 1
                dqkt_block[px, py] = 2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) - delta/2.0

            if negative:
                dqkt_block[px, py] *= -1
            idqkt_block = DqKT().idqkt2(dqkt_block)
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
            bt_of_cover.set_block(idqkt_block, self.pos[i])

        print("Convirtiendo a RGB")
        image_rgb_array = itools.ycbcr2rgb(cover_ycbcr_array)

        return Image.fromarray(image_rgb_array)

    def extract(self, watermarked_image):

        print("...Proceso de extraccion...")

        # Instancias necesarias
        itools = ImageTools()

        print("Convirtiendo a YCbCr")
        # Convirtiendo a modelo de color YCbCr
        watermarked_ycbcr_array = itools.rgb2ycbcr(watermarked_image)

        # Tomando componente Y
        watermarked_array = watermarked_ycbcr_array[:, :, 0]

        bt_of_watermarked_Y = BlocksImage(watermarked_array)

        extract = []

        print("Extrayendo")
        # Recorrer todos los bloques de la imagen
        for i in range(self.len_watermark_list):
            block = bt_of_watermarked_Y.get_block(self.pos[i])
            # Valores de coeficiente y delta optimo para el bloque
            (c, delta) = self.clasificar(self.pos[i], watermarked_image)
            dqkt_block = DqKT().dqkt2(np.array(block, dtype=np.float32))
            
            negative = False
            
            (px, py) = self.get_indice(c)           
            if dqkt_block[px, py] < 0:
                negative = True
            
            C1 = (2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) + delta/2.0) - abs(dqkt_block[px, py])
            
            C0 = (2*delta*round(abs(dqkt_block[px, py])/(2.0*delta)) - delta/2.0) - abs(dqkt_block[px, py])

            if negative:
                C1 *= -1
                C0 *= -1
            if C0 < C1:
                extract.append(1)
            else:
                extract.append(0)

        # Creando una imagen cuadrada con valores 255
        wh = int(math.sqrt(self.len_watermark_list))
        extract_image = Image.new("1", (wh, wh), 255)
        array_extract_image1 = misc.fromimage(extract_image)

        for i in range(wh):
            for y in range(wh):
                if extract[wh*i+y] == 0:
                    array_extract_image1[i, y] = 0
        
        watermark_extracted = misc.toimage(array_extract_image1)
        for i in range(10):
            watermark_extracted = DAT().dat2(watermark_extracted)
        
        array = misc.fromimage(watermark_extracted)

        array_as_list = array.reshape((1, self.len_watermark_list))[0]

        myqr1 = MyQR62()
        
        # Insertando datos al QR code
        myqr1.set_data(array_as_list)
        
        watermark_extracted = misc.toimage(myqr1.get_qr())

        b = BlocksImage(misc.fromimage(watermark_extracted), 2, 2)
        for m in range(b.max_num_blocks()):
            b.set_color(m)

        watermark_extracted = misc.toimage(b.get())
        
        return watermark_extracted
