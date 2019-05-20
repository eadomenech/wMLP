# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def run_main():
    shivani2017 = [
        30.447635,31.219799,31.554887,30.220838,30.164094,32.554847,30.367436,30.041741,30.469106,30.276339,30.356790,31.412881,31.065932,30.302877,30.075065,30.077857,30.498336,30.407601,30.529168,30.423076,30.250703,31.440316,32.011709,31.630896,31.008705,30.489371,29.815700,30.464952,30.175556,29.977137,30.406774,30.737615,31.909841,30.695238,32.095520,30.676564,30.696616,31.450633,30.602407,30.789171,30.222629,29.968478,31.430082,30.480782,30.825104,30.329248,33.263400
    ]
    
    liu2018 = [
        29.144582,29.676223,30.086566,29.057510,28.930893,29.832005,29.097898,29.091994,29.084150,28.998277,29.055450,29.244746,29.275899,29.106724,28.805820,28.935742,28.968122,28.964580,28.830925,29.205075,28.978987,29.269055,29.559399,29.269376,29.115721,29.150580,28.709157,29.093292,28.909379,28.855491,29.132349,29.287283,29.431034,29.273097,29.517002,29.301754,29.190732,29.625035,29.206161,29.462055,28.960510,28.901056,29.335619,29.118898,29.157676,29.099515,29.636072
    ]

    avila2018 = [
        42.960054,43.224420,43.433406,43.005464,42.947479,43.251250,43.031539,42.987245,42.964493,43.008792,43.037624,42.984712,43.088420,42.950827,43.059718,42.965242,42.918636,42.903525,42.893641,43.214623,42.902675,43.072284,43.262157,43.014802,43.058923,43.017645,42.959674,43.060839,42.999239,42.941270,42.963574,43.074415,43.080713,43.050928,43.094788,43.010264,42.984320,43.144119,43.096670,43.174227,42.815593,42.952935,43.055874,43.022928,43.113339,42.980672,43.176219
    ]

    proposed = [
        43.423582,43.564286,43.169471,43.169680,43.281467,43.302932,43.351793,43.290989,43.034992,43.151810,43.169041,43.252290,43.380958,43.180336,43.241857,43.182945,43.178221,43.123956,43.271075,43.156314,43.155034,43.117412,43.006219,43.169170,43.004329,43.308344,43.027126,43.243733,43.025601,43.133267,43.229232,43.387909,43.196611,43.310183,43.318963,43.589430,43.427713,43.739811,43.152013,43.384393,43.088064,43.239997,43.388794,43.125670,43.360060,43.267297,43.254968
    ]

    data = [shivani2017, liu2018, avila2018, proposed]

    fig, ax1 = plt.subplots(figsize=(6, 6))
    fig.canvas.set_window_title('PSNR')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], marker='+', color='blue')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(
        True, linestyle='-', which='major', color='lightgrey',
        alpha=0.5)
    
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Parzival database (47 images)')
    ax1.set_xlabel('Schemes')
    ax1.set_ylabel('PSNR')

    # Now fill the boxes with desired colors
    numDists = 4
    boxColors = ['darkkhaki', 'green', 'blue', 'red']
    for i in range(numDists):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = np.column_stack([boxX, boxY])
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[i])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        # Finally, overplot the sample averages, with horizontal alignment in the center of each box
        ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
                color='w', marker='*', markeredgecolor='k')

    ax1.set_xticklabels(
        [
            '[13]', '[9]', '[2] + FW', 'Proposed'
        ],
        rotation=0, fontsize=14)
    plt.show()


if __name__ == "__main__":
    run_main()