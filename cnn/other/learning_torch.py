#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt


def main(args):
    # # Cargando Lenna en escala grises
    # lenna = Image.open("static/Lenna.jpg")
    # # Convirtiendo la imagen en nd array
    # lenna_nd = np.array(lenna)
    # lenna_tensor = torch.from_numpy(lenna_nd)
    # # imprimiendo datos del tensor
    # print("Tensor:")
    # print(lenna_tensor)
    #
    # # Cargando varias imagenes. Todas las que se ubiquen en static y sean jpg
    # images = glob.glob("static/*.jpg")
    # # Imprimiendo lista de direcciones de imagenes jpg en static
    # print("Lista de rutas de imagenes:")
    # print(images)
    #
    # # Convirtiendo imagenes en ndarray
    # imagenes_array = np.array(
    #     [np.array(Image.open(img).resize((224, 224))) for img in images[:64]])
    # print("Arrays de las imagenes:")
    # print(imagenes_array)
    #
    # images = imagenes_array.reshape(-1, 224, 224, 3)
    # images_tensor = torch.from_numpy(images)
    # print("Tensores de las imagenes:")
    # print(images_tensor)

    # # Multiplicacion de tensores con CPU y GPU
    # a = torch.rand(1000, 1000)
    # b = torch.rand(1000, 1000)
    # print("Con CPU")
    # print(a.matmul(b))
    #
    # if torch.cuda.is_available():
    #     print("Con GPU")
    #     a = a.cuda()
    #     b = b.cuda()
    #     print(a.matmul(b))

    # # example where we create variables and check the gradients
    # from torch.autograd import Variable
    # x = Variable(torch.ones(2, 2), requires_grad=True)
    # print(x)
    # y = x.mean()
    # print(y)
    # y.backward()
    # print(x.grad)
    # print(x.data)

    # # Creating data for our neural network
    # from torch.autograd import Variable
    #
    # def get_data():
    #     train_X = np.asarray([
    #         3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
    #         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    #     train_Y = np.asarray([
    #         1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
    #         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    #     dtype = torch.FloatTensor
    #     X = Variable(
    #         torch.from_numpy(train_X).type(dtype), requires_grad=False
    #     ).view(17, 1)
    #     y = Variable(
    #         torch.from_numpy(train_Y).type(dtype), requires_grad=False)
    #     return X, y
    #
    # def get_weights():
    #     w = Variable(torch.randn(1), requires_grad=True)
    #     b = Variable(torch.randn(1), requires_grad=True)
    #     return w, b
    #
    # def simple_network(x):
    #     w, b = get_weights()
    #     y_pred = torch.matmul(x, w) + b
    #     return y_pred
    #
    # def loss_fn(y, y_pred):
    #     loss = (y_pred - y).pow(2).sum()
    #     w, b = get_weights()
    #     for param in [w, b]:
    #         if param.grad:
    #             param.grad.data.zero_()
    #     loss.backward()
    #     return loss.data
    #
    # def optimize(learning_rate):
    #     w.data -= learning_rate * w.grad.data
    #     b.data -= learning_rate * b.grad.data
    #
    # x, y = get_data()
    # y_pred = simple_network(x)
    # print(loss_fn(y, y_pred))

    # from torch.autograd import Variable
    # from torch.nn import Linear
    #
    # inp = Variable(torch.randn(1, 10))
    # myLayer = Linear(in_features=10, out_features=5, bias=True)
    # myLayer(inp)
    #
    # print(myLayer.weight)

    # # PyTorch non-linear activations
    # from torch.autograd import Variable
    # from torch.nn import ReLU
    #
    # sample_data = Variable(torch.Tensor([1, 2, -1, -1]))
    # myRelu = ReLU()
    # print(myRelu(sample_data))

    # from torch.autograd import Variable
    # from torch.nn import MSELoss
    #
    # loss = MSELoss()
    # input = Variable(torch.randn(3, 5), requires_grad=True)
    # target = Variable(torch.randn(3, 5))
    # output = loss(input, target)
    # output.backward()

    # # For classification
    # from torch.autograd import Variable
    # from torch.nn import CrossEntropyLoss
    #
    # loss = CrossEntropyLoss()
    # input = Variable(torch.randn(3, 5), requires_grad=True)
    # target = Variable(torch.LongTensor(3).random_(5))
    # output = loss(input, target)
    # output.backward()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
