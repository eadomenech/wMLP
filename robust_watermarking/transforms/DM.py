# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
from PIL import Image
from scipy import misc
import random


class DM:  # Dither modulations

    def get(self, im, window=2):
        new = im.copy()
        pix = im.load()
        newpix = new.load()
        width, height = im.size

        area = window*window
        for i in range(width//window):     # loop over pixels
            for j in range(height//window):  # loop over pixels
                avg = 0
                area_pix = []
                for k in range(window):
                    for l in range(window):
                        area_pix.append((k, l))  # make a list of coordinates within the tile
                        try:
                            avg += pix[window*i+k, window*j+l]
                            newpix[window*i+k, window*j+l] = 0  # set everything to black
                        except IndexError:
                            avg += 255/2    # just an arbitrary mean value (when were outside of the image)
                                            # this is just a dirty trick for coping with images that have
                                            # sides that are not multiples of window
                avg = avg/area
                # val = v is the number of pixels within the tile that will be turned white
                val = abs(round(avg/255.0 * (area+0.99) - 0.5))  # 0.99 due to rounding errors
                assert val <= area, 'something went wrong with the val'
                random.shuffle(area_pix)  # randomize pixel coordinates
                for m in range(int(val)):
                    rel_coords = area_pix.pop()  # find random pixel within tile and turn it white
                    newpix[window*i+rel_coords[0], window*j+rel_coords[1]] = 255

        return new
