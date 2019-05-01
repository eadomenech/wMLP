#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from scipy import misc
import pyopencl as cl
import numpy.linalg as la
import datetime
from time import time
from scipy import misc
from PIL import Image
import math
import numpy as np


class DqKT_GPU:
    """
    Transformada Discreta de Krawtchouk
    """

    def __init__(self):
        coef0 = np.array([0.0669182, 0.181935, 0.333578, 0.46961, 0.527532, 0.472859, 0.325442, 0.151051])
        coef1 = np.array([0.191235, 0.388597, 0.471368,  0.323351, -0.0202087, -0.363246, -0.48874, -0.338313])
        coef2 = np.array([0.357657, 0.473562, 0.222594, -0.215771, -0.407057, -0.123127, 0.353475, 0.495899])
        coef3 = np.array([0.49842, 0.296162, -0.236366, -0.369437, 0.0838858, 0.413053, 0.0225738, -0.541622])
        coef4 = np.array([0.537856, -0.0851108, -0.404159, 0.107868, 0.38164, -0.179119, -0.37539, 0.458079])
        coef5 = np.array([0.449447, -0.419763, -0.0585517, 0.412412, -0.200908, -0.285171, 0.486411, -0.300004])
        coef6 = np.array([0.279847, -0.485158, 0.412485, -0.0429201, -0.351539, 0.49078, -0.35839, 0.1464])
        coef7 = np.array([0.114035, -0.291713, 0.473503, -0.555255, 0.488852, -0.323131, 0.154305, -0.0467557])

        self.coef = np.array([coef0, coef1, coef2, coef3, coef4, coef5, coef6, coef7])

    def dqkt2(self, array):
        """
        2D DKT
        """
        imageBuffer = array.astype(np.float32)

        width = len(imageBuffer[0])
        height = len(imageBuffer)

        image_format_float = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)

        kernelBuffer = np.array([
            0.0669182, 0.181935, 0.333578, 0.46961, 0.527532, 0.472859, 0.325442, 0.151051,
            0.191235, 0.388597, 0.471368,  0.323351, -0.0202087, -0.363246, -0.48874, -0.338313,
            0.357657, 0.473562, 0.222594, -0.215771, -0.407057, -0.123127, 0.353475, 0.495899,
            0.49842, 0.296162, -0.236366, -0.369437, 0.0838858, 0.413053, 0.0225738, -0.541622,
            0.537856, -0.0851108, -0.404159, 0.107868, 0.38164, -0.179119, -0.37539, 0.458079,
            0.449447, -0.419763, -0.0585517, 0.412412, -0.200908, -0.285171, 0.486411, -0.300004,
            0.279847, -0.485158, 0.412485, -0.0429201, -0.351539, 0.49078, -0.35839, 0.1464,
            0.114035, -0.291713, 0.473503, -0.555255, 0.488852, -0.323131, 0.154305, -0.0467557
        ]).astype(np.float32)

        global_size=(width, height)
        #global_size = (1024, 768)
        local_size = (8, 8)

        c_result = np.empty_like(imageBuffer)

        # Speed in normal CPU usage
        #time1 = time()
        #c_temp = (a+b) # adds each element in a to its corresponding element in b
        #c_temp = c_temp * c_temp # element-wise multiplication
        #c_result = c_temp * (a/2.0) # element-wise half a and multiply
        #time2 = time()

        #print("Execution time of test without OpenCL: ", time2 - time1, "s")
        device = cl.get_platforms()[0].get_devices()[0]

        # Simnple speed test
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


        mf = cl.mem_flags
        #img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imageBuffer)
        kernel_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernelBuffer)
        #dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, imageBuffer.nbytes)
        image_orig = cl.image_from_array(ctx, imageBuffer, num_channels=1)
        image_dest = cl.image_from_array(ctx, c_result, num_channels=1)
        #image_dest = cl.Image(ctx, mf.WRITE_ONLY, image_format_float, shape=imageBuffer.shape)

        prg = cl.Program(ctx, """
                #pragma OPENCL EXTENSION cl_khr_fp64 : enable

                __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

                __kernel void transformada(read_only image2d_t img_orig,
                __global const float *k, write_only image2d_t img_dest)
                {
                            int x = get_global_id(0);
                            int y = get_global_id(1);
                            int z = get_local_id(0);
                            int r = get_local_id(1);
                            int2 coord;
                            float4 pixel;
                            double c_temp = 0.0;
                            coord = (int2)(x, y);
                            pixel = read_imagef(img_orig, sampler, coord);
                            for(int a = 0; a < 8; a++){
                                for(int b = 0; b < 8; b++){
                                    float4 p = read_imagef(img_orig, sampler, (int2)(a, b));
                                    c_temp += (k[a * 8 + z] * k[b * 8 + r] * p.x);
                                }
                            }
                            pixel.x = (float)(c_temp); // Suma del valor del pixel procesado
                            write_imagef(img_dest, coord, pixel); // store result in global memory
                    }
                """).build()

        exec_evt = prg.transformada(queue, global_size, local_size, image_orig, kernel_buf, image_dest)

        cl.enqueue_copy(queue, c_result, image_dest, origin=(0, 0), region=global_size)
        print ("Resultados: %s" % c_result)
        return c_result

    def idqkt2(self, array):
        """
        2D IDKT
        """
        idkt = np.ndarray(shape=(8, 8))
        for x in range(8):
            for y in range(8):
                idkt[x, y] = self.IM(array, x, y)
        return idkt

    def M(self, A, m, n):
        M = 0
        for x in range(8):
            for y in range(8):
                M += ((self.coef[x, m]*self.coef[y, n]))*float(A[x, y])
        return M

    def IM(self, A, m, n):
        IM = 0
        for x in range(8):
            for y in range(8):
                IM += float(A[x, y])*self.coef[m, x]*self.coef[n, y]
        return IM
