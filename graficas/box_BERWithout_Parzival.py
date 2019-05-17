# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def run_main():
    shivani2017 = [
        0.328963,0.331113,0.332800,0.326988,0.328321,0.329482,0.329277,0.330123,0.328896,0.329195,0.329847,0.326490,0.328295,0.327473,0.325586,0.327319,0.324525,0.327057,0.335886,0.325018,0.324353,0.325489,0.326397,0.327363,0.326276,0.327278,0.326139,0.327206,0.323977,0.326438,0.330886,0.331413,0.329536,0.328447,0.328266,0.329313,0.330754,0.332751,0.327500,0.330971,0.329291,0.326580,0.328341,0.329267,0.334490,0.326500,0.328765
    ]
    
    liu2018 = [
        0.280697,0.287461,0.299688,0.290062,0.270812,0.283039,0.278616,0.267950,0.258065,0.267690,0.271332,0.268991,0.267950,0.265088,0.273933,0.269251,0.286160,0.266909,0.255723,0.272373,0.277575,0.268470,0.268470,0.268210,0.281998,0.271852,0.290583,0.274454,0.273933,0.262747,0.269511,0.284079,0.264048,0.284860,0.278356,0.274974,0.258845,0.292404,0.260146,0.271592,0.263788,0.263788,0.284599,0.261707,0.280697,0.276015,0.264308
    ]

    avila2018 = [
        0.000000,0.001041,0.001041,0.003122,0.001041,0.000000,0.004162,0.003122,0.000000,0.001041,0.001041,0.001041,0.000000,0.000000,0.000000,0.000000,0.000000,0.001041,0.000000,0.001041,0.003122,0.000000,0.001041,0.002081,0.001041,0.000000,0.000000,0.001041,0.000000,0.001041,0.001041,0.001041,0.000000,0.000000,0.001041,0.001041,0.000000,0.001041,0.001041,0.002081,0.000000,0.002081,0.002081,0.000000,0.000000,0.002081,0.001041,0.001041,0.003122,0.000000,0.002081,0.000000,0.001041,0.003122,0.002081,0.000000,0.002081,0.000000,0.000000,0.002081
    ]

    proposed = [
        0.001041,0.000000,0.001041,0.000000,0.000000,0.000000,0.000000,0.000000,0.001041,0.001041,0.000000,0.000000,0.001041,0.000000,0.000000,0.001041,0.001041,0.000000,0.001041,0.000000,0.001041,0.000000,0.000000,0.001041,0.001041,0.000000,0.002081,0.000000,0.000000,0.000000,0.000000,0.000000,0.001041,0.000000,0.000000,0.000000,0.000000,0.001041,0.001041,0.000000,0.000000,0.000000,0.000000,0.000000,0.002081,0.000000,0.000000,0.000000,0.001041,0.001041,0.000000,0.001041,0.001041,0.000000,0.000000,0.001041,0.000000,0.002081,0.000000,0.001041
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