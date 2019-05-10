from PIL import Image
import math

from scipy import signal


class Evaluations:

    def SIM_1_A(self, array1, array2):

        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for i in range(len(array1)):
            for j in range(len(array1[0])):
                if array1[i, j] != 0:
                    array1[i, j] = 1
                if array2[i, j] != 0:
                    array2[i, j] = 1
                s1 += array1[i, j]*array2[i, j]
                s2 += array1[i, j]**2
                s3 += array2[i, j]**2

        return s1/(math.sqrt(s2)*math.sqrt(s3))

    # Working with images
    def PSNR_L(self, img1, img2):
        MSE = self.MSE_L(img1, img2)
        if MSE != 0:
            return 10*(math.log10(255**2/MSE))
        else:
            return 100

    def PSNR_RGB(self, img1, img2):
        MSE = self.MSE_RGB(img1, img2)
        if MSE != 0:
            return 10*(math.log10(255**2/MSE))
        else:
            return 100

    def MSE_L(self, img1, img2):
        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        if (w1 != w2) or (h1 != h2):
            return Exception
        mse = 0
        MSE = 0
        pixels1 = img1.load()  # create the pixel map
        pixels2 = img2.load()
        for i in range(w1):    # for every pixel:
            for j in range(h1):
                mse += math.fabs(pixels1[i, j]-pixels2[i, j])**2
        MSE = mse / (w1*h1)

        return MSE

    def MSE_RGB(self, img1, img2):
        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        if (w1 != w2) or (h1 != h2):
            return Exception
        mse = [0, 0, 0]
        MSE = 0
        pixels1 = img1.load()  # create the pixel map
        pixels2 = img2.load()
        a = []
        b = []
        for i in range(w1):    # for every pixel:
            for j in range(h1):
                b.append(abs(pixels1[i, j][0]-pixels2[i, j][0]))
                mse[0] = mse[0] + (math.fabs(pixels1[i, j][0]-pixels2[i, j][0]))**2
                mse[1] = mse[1] + (math.fabs(pixels1[i, j][1]-pixels2[i, j][1]))**2
                mse[2] = mse[2] + (math.fabs(pixels1[i, j][2]-pixels2[i, j][2]))**2
            a.append(b)
        mse = (mse[0] + mse[1] + mse[2]) / 3
        MSE = mse / (w1*h1)

        return MSE

    # Working with arrays
    def MSE_L_A(self, array1, array2):
        w1 = len(array1)
        h1 = len(array1[0])
        w2 = len(array2)
        h2 = len(array2[0])
        if (w1 != w2) or (h1 != h2):
            return Exception
        mse = 0
        MSE = 0
        for i in range(w1):    # for every pixel:
            for j in range(h1):
                mse += math.fabs(array2[i, j]-array1[i, j])**2
        MSE = mse / (w1*h1)

        return MSE

    def PSNR_L_A(self, array1, array2):
        MSE = self.MSE_L_A(array1, array2)
        if MSE != 0:
            return 10*(math.log10(255**2/MSE))
        else:
            return -1

    def BER_A(self, array1, array2):
        S = 0
        B = array1.size
        for i in range(len(array1)):
            for j in range(len(array1[0])):
                if array1[i, j] != array2[i, j]:
                    S += 1
        BER = float(S)/B
        return BER

    def NC_A(self, array1, array2): # Esto solo para imagenes en escalas de grises, no para binarias
        original = 0
        temp = 0
        for i in range(len(array1)):
            for j in range(len(array1[0])):
                temp += (int(array1[i, j])*int(array2[i, j]))
                original += (int(array1[i, j])**2)
        return float(temp)/original
