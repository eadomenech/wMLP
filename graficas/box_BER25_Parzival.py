# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def run_main():
    shivani2017 = [
        0.484468,0.486914,0.489414,0.484600,0.483428,0.487239,0.483634,0.486030,0.487094,0.484954,0.486413,0.489019,0.486118,0.484713,0.483205,0.485583,0.487496,0.485766,0.485300,0.484330,0.484927,0.485462,0.486730,0.488170,0.486016,0.484856,0.482454,0.485052,0.484445,0.486887,0.484963,0.485308,0.487741,0.485317,0.488677,0.483822,0.486685,0.482896,0.484165,0.484716,0.484315,0.485322,0.484869,0.485174,0.482479,0.485582,0.486797
    ]
    
    liu2018 = [
        0.328044,0.328044,0.322841,0.326743,0.326743,0.330905,0.329865,0.326223,0.322581,0.321800,0.328044,0.329084,0.322581,0.326223,0.329605,0.323101,0.328564,0.324662,0.326743,0.323881,0.327523,0.325963,0.328824,0.324922,0.321280,0.327523,0.338189,0.325963,0.326483,0.329605,0.325182,0.323361,0.323361,0.325963,0.330385,0.324922,0.328824,0.325182,0.324662,0.325963,0.327263,0.328824,0.332466,0.326223,0.327784,0.327784,0.328304
    ]

    avila2018 = [
        0.078564,0.083507,0.084287,0.089490,0.084547,0.055151,0.091311,0.090531,0.082466,0.091311,0.094433,0.068158,0.089230,0.088970,0.086368,0.086108,0.069979,0.082726,0.088450,0.087929,0.095734,0.069719,0.069199,0.069459,0.069979,0.087149,0.094433,0.085068,0.086108,0.097034,0.099896,0.082466,0.072060,0.084807,0.061134,0.095473,0.082206,0.077003,0.084287,0.080385,0.084547,0.097034,0.071800,0.085588,0.074402,0.095473,0.034860
    ]

    proposed = [
        0.104318,0.098075,0.098595,0.088970,0.081426,0.079865,0.102237,0.092092,0.083247,0.094433,0.085588,0.071800,0.094173,0.096774,0.090010,0.081165,0.080125,0.068678,0.087929,0.085848,0.081946,0.076743,0.075442,0.087149,0.084547,0.099636,0.083507,0.083507,0.079084,0.077263,0.091571,0.093132,0.092092,0.081946,0.085588,0.092872,0.089490,0.103018,0.082206,0.091831,0.091831,0.090531,0.084807,0.094953,0.101197,0.094433,0.061134
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