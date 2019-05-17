# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def run_main():
    shivani2017 = [
        0.463998,0.466447,0.465299,0.448563,0.453495,0.462315,0.461770,0.458997,0.459774,0.461734,0.452843,0.452799,0.461913,0.458443,0.451774,0.451819,0.442856,0.454081,0.467944,0.447857,0.448230,0.452971,0.454691,0.458396,0.450970,0.458527,0.440801,0.457339,0.442592,0.454924,0.459845,0.466122,0.464720,0.457410,0.462051,0.463235,0.464540,0.467256,0.458411,0.465031,0.456931,0.455697,0.459295,0.462596,0.467379,0.460323,0.461989
    ]
    
    liu2018 = [
        0.331686,0.334807,0.331686,0.333507,0.332206,0.342092,0.335848,0.334027,0.328044,0.334027,0.329605,0.334547,0.334807,0.334807,0.334027,0.333767,0.330905,0.338970,0.329084,0.339750,0.333767,0.334807,0.335068,0.334287,0.332726,0.330645,0.338970,0.328044,0.334027,0.328824,0.334807,0.329865,0.326223,0.328564,0.332986,0.329605,0.335068,0.328044,0.337669,0.330645,0.334547,0.333507,0.331426,0.338970,0.331165,0.338189,0.329605
    ]

    avila2018 = [
        0.048387,0.053330,0.048647,0.053850,0.051509,0.061134,0.048387,0.056452,0.047867,0.054891,0.050468,0.054891,0.057492,0.059053,0.050989,0.066077,0.048907,0.049168,0.055151,0.057752,0.045525,0.054891,0.054891,0.053330,0.065036,0.056191,0.054631,0.063215,0.052549,0.051769,0.059573,0.056972,0.062695,0.055411,0.057752,0.055411,0.062175,0.048127,0.062435,0.060614,0.057492,0.056972,0.047086,0.057232,0.050989,0.058012,0.052289,0.057752,0.059573,0.055671,0.046826,0.049168,0.061134,0.063736,0.053590,0.049948,0.056712,0.063476,0.062695,0.058012,
    ]

    proposed = [
        0.048387,0.053330,0.048647,0.053850,0.051509,0.061134,0.048387,0.056452,0.047867,0.054891,0.050468,0.054891,0.057492,0.059053,0.050989,0.066077,0.048907,0.049168,0.055151,0.057752,0.045525,0.054891,0.054891,0.053330,0.065036,0.056191,0.054631,0.063215,0.052549,0.051769,0.059573,0.056972,0.062695,0.055411,0.057752,0.055411,0.062175,0.048127,0.062435,0.060614,0.057492,0.056972,0.047086,0.057232,0.050989,0.058012,0.052289,0.057752,0.059573,0.055671,0.046826,0.049168,0.061134,0.063736,0.053590,0.049948,0.056712,0.063476,0.062695,0.058012,
    ]

    data = [shivani2017, liu2018, avila2018, proposed]

    fig, ax1 = plt.subplots(figsize=(6, 6))
    fig.canvas.set_window_title('BER')
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
    ax1.set_ylabel('BER')

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