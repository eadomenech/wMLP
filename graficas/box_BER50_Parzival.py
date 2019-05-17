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
        0.048387,0.053330,0.048647,0.053850,0.051509,0.061134,0.048387,0.056452,0.047867,0.054891,0.050468,0.054891,0.057492,0.059053,0.050989,0.066077,0.048907,0.049168,0.055151,0.057752,0.045525,0.054891,0.054891,0.053330,0.065036,0.056191,0.054631,0.063215,0.052549,0.051769,0.059573,0.056972,0.062695,0.055411,0.057752,0.055411,0.062175,0.048127,0.062435,0.060614,0.057492,0.056972,0.047086,0.057232,0.050989,0.058012,0.052289,0.057752,0.059573,0.055671,0.046826,0.049168,0.061134,0.063736,0.053590,0.049948,0.056712,0.063476,0.062695,0.058012
    ]

    proposed = [
        0.048387,0.053330,0.048647,0.053850,0.051509,0.061134,0.048387,0.056452,0.047867,0.054891,0.050468,0.054891,0.057492,0.059053,0.050989,0.066077,0.048907,0.049168,0.055151,0.057752,0.045525,0.054891,0.054891,0.053330,0.065036,0.056191,0.054631,0.063215,0.052549,0.051769,0.059573,0.056972,0.062695,0.055411,0.057752,0.055411,0.062175,0.048127,0.062435,0.060614,0.057492,0.056972,0.047086,0.057232,0.050989,0.058012,0.052289,0.057752,0.059573,0.055671,0.046826,0.049168,0.061134,0.063736,0.053590,0.049948,0.056712,0.063476,0.062695,0.058012
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