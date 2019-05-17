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
        44.265707,44.676721,45.411788,44.531670,44.694593,44.252413,45.035256,44.493173,45.007116,44.376873,45.185017,44.943083,44.801391,44.854296,44.900670,44.581012,44.560070,45.088821,44.661495,44.690536,44.719412,44.512057,45.017194,44.697061,44.582661,44.674387,45.198149,44.845690,45.022010,45.046487,44.577858,45.003475,44.808285,44.826332,44.977428,44.758299,44.846555,45.587070,44.693239,44.678502,45.128010,44.642840,44.997605,45.267579,44.823783,44.874219,44.804761,44.596934,43.971378,44.644823,44.543947,44.587669,44.738352,44.536479,44.649925,45.459403,44.884012,44.586786,44.753116,44.195675
    ]

    proposed = [
        44.865565,45.486391,46.385161,45.294026,45.506651,45.018650,45.968294,45.300377,45.915015,45.162344,46.146698,45.967969,45.731443,45.765632,45.775734,45.339104,45.228895,46.185381,45.455595,45.562604,45.538270,45.323617,45.944508,45.571139,45.359744,45.591798,46.173250,45.752019,45.895927,46.018509,45.354887,45.971744,45.631633,45.696410,45.840000,45.677589,45.717885,46.689241,45.419025,45.590658,46.015287,45.480218,45.904398,46.347249,45.710122,45.762854,45.665281,45.417849,44.635474,45.422417,45.410899,45.284791,45.633728,45.340174,45.521461,46.470800,45.750450,45.336114,45.710160,44.906104
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
            '[13]', '[9]', '[2]', 'Proposed'
        ],
        rotation=0, fontsize=14)
    plt.show()


if __name__ == "__main__":
    run_main()