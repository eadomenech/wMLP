# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
from PIL import Image
from scipy import misc


class DAT:  # Discret Arnold Transforms

    def dat2(self, inputImage):
        s = inputImage.size
        input_array = misc.fromimage(inputImage)
        outputImage = Image.new('L', s)
        output_array = misc.fromimage(outputImage)
        for x in range(s[1]):
            for y in range(s[0]):
                output_array[(x + y) % s[1], (2*x + y) % s[0]] = input_array[x, y]
        return misc.toimage(output_array)
