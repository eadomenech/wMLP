
from math import log10, log
import cv2


class Metrics:
    def mse(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        diff_pow = (C - S) ** 2
        return sum(sum(sum(diff_pow))) / (m * n * dims_cover[2])

    def mse_L(self, cover_array, stego_array):
        """
        MSE para imagenes en escala de grises
        """
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        diff_pow = (C - S) ** 2
        print(sum(sum(diff_pow)))
        return sum(sum(diff_pow)) / (m * n)

    def psnr(self, cover_array, stego_array):
        RMSE = self.mse(cover_array, stego_array)
        if RMSE != 0:
            return 10 * log10(255 ** 2 / RMSE)
        else:
            return 100

    def psnr_L(self, cover_array, stego_array):
        RMSE = self.mse_L(cover_array, stego_array)
        if RMSE != 0:
            return 10 * log10(255 ** 2 / RMSE)
        else:
            return 100

    def uiqi(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        C_m = sum(sum(sum(C))) / (m * n * dims_cover[2])
        S_m = sum(sum(sum(S))) / (m * n * dims_cover[2])
        diff_pow = (C - C_m) ** 2
        sigma2c = sum(sum(sum(diff_pow))) / (m * n * dims_cover[2] - 1)
        diff_pow = (S - S_m) ** 2
        sigma2s = sum(sum(sum(diff_pow))) / (m * n * dims_cover[2] - 1)
        prod_diff = (C - C_m) * (S - S_m)
        sigmacs = sum(sum(sum(prod_diff))) / (m * n * dims_cover[2] - 1)
        aux = 4 * sigmacs * C_m * S_m
        return aux / ((sigma2c + sigma2s) * (C_m ** 2 + S_m ** 2))

    def image_fid(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        return 1 - sum(sum(sum((C - S) ** 2))) / sum(sum(sum(C ** 2)))

    def hist_sim(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        histc_1 = cv2.calcHist([cover_array], [0], None, [256], [0, 256])
        histc_2 = cv2.calcHist([cover_array], [1], None, [256], [0, 256])
        histc_3 = cv2.calcHist([cover_array], [2], None, [256], [0, 256])
        hists_1 = cv2.calcHist([stego_array], [0], None, [256], [0, 256])
        hists_2 = cv2.calcHist([stego_array], [1], None, [256], [0, 256])
        hists_3 = cv2.calcHist([stego_array], [2], None, [256], [0, 256])
        hs = 0
        for i in range(256):
            aux_1 = histc_1[i][0] + histc_2[i][0] + histc_3[i][0]
            aux_2 = hists_1[i][0] + hists_2[i][0] + hists_3[i][0]
            diff = (aux_1 - aux_2) / (m * n * dims_cover[2])
            hs += abs(diff)
        return hs

    def rel_entropy(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        histc_1 = cv2.calcHist([cover_array], [0], None, [256], [0, 256])
        histc_2 = cv2.calcHist([cover_array], [1], None, [256], [0, 256])
        histc_3 = cv2.calcHist([cover_array], [2], None, [256], [0, 256])
        hists_1 = cv2.calcHist([stego_array], [0], None, [256], [0, 256])
        hists_2 = cv2.calcHist([stego_array], [1], None, [256], [0, 256])
        hists_3 = cv2.calcHist([stego_array], [2], None, [256], [0, 256])
        rel_ent = 0
        for i in range(256):
            aux_1 = histc_1[i][0] + histc_2[i][0] + histc_3[i][0]
            aux_2 = hists_1[i][0] + hists_2[i][0] + hists_3[i][0]
            sum_all = sum(histc_1)[0] + sum(histc_2)[0] + sum(histc_3)[0]
            prob_cover = aux_1 / sum_all
            prob_stego = aux_2 / sum_all
            if prob_cover != 0 and prob_stego != 0:
                rel_ent += prob_cover * abs(log(prob_cover / prob_stego))
        return rel_ent

    def ber(self, binary_seq_1, binary_seq_2):
        s = 0
        for i in range(len(binary_seq_1)):
            if binary_seq_1[i] != binary_seq_2[i]:
                s += 1
        return float(s)/len(binary_seq_1)
