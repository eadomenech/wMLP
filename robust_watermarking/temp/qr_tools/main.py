from MyQR62 import MyQR62
import numpy as np
from scipy import misc
from PIL import Image


myqr = MyQR62()

Image.fromarray(myqr.get_qr()).show()