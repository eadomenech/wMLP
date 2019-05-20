# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def run_main():
    shivani2017 = [
        0.474388,0.475282,0.478210,0.468191,0.469664,0.476357,0.473043,0.471928,0.474296,0.473712,0.470193,0.473189,0.475587,0.472086,0.468916,0.469622,0.467874,0.471231,0.476643,0.468752,0.468757,0.473323,0.474012,0.474573,0.471371,0.473723,0.465503,0.471297,0.466044,0.470477,0.472016,0.475443,0.477057,0.472100,0.476796,0.474161,0.474422,0.476366,0.472324,0.474861,0.471404,0.470387,0.473873,0.473948,0.474899,0.473284,0.479321,
    ]
    
    liu2018 = [
        0.326743,0.324922,0.326743,0.323361,0.328304,0.330645,0.326483,0.330125,0.325702,0.319979,0.326223,0.331686,0.322841,0.327263,0.329084,0.325963,0.331426,0.328824,0.326223,0.330645,0.328564,0.332466,0.325702,0.325963,0.324402,0.325963,0.334287,0.323621,0.331946,0.329865,0.330125,0.324922,0.329344,0.324922,0.326223,0.324922,0.325702,0.325182,0.327523,0.325963,0.318939,0.330385,0.326743,0.325702,0.328304,0.326483,0.324142
    ]

    avila2018 = [
        0.076743,0.081426,0.081946,0.085328,0.083767,0.054370,0.087929,0.089490,0.081165,0.088970,0.092352,0.066077,0.086368,0.085588,0.083767,0.082466,0.068678,0.081165,0.084807,0.086368,0.091831,0.069459,0.068418,0.066857,0.069199,0.086108,0.092352,0.082206,0.084547,0.095213,0.098335,0.079344,0.069719,0.083767,0.060094,0.093392,0.079344,0.075702,0.084027,0.077784,0.082206,0.095734,0.070239,0.083247,0.071800,0.093913,0.035120
    ]

    proposed = [
        0.102237,0.091571,0.088710,0.085848,0.080385,0.069719,0.101457,0.080905,0.086629,0.100156,0.082726,0.073101,0.097034,0.097815,0.090791,0.084287,0.081686,0.069719,0.085588,0.090010,0.080645,0.079865,0.077784,0.087409,0.082466,0.104058,0.086629,0.088710,0.079865,0.079344,0.091311,0.089230,0.084027,0.081426,0.083247,0.087929,0.087929,0.096514,0.078564,0.081165,0.097555,0.092872,0.083247,0.096774,0.102237,0.102497,0.070239
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
            '[13]', '[9]', '[2] + FW', 'Proposed'
        ],
        rotation=0, fontsize=14)
    plt.show()


if __name__ == "__main__":
    run_main()