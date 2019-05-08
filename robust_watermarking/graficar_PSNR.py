# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-


def run_main():
    db_images = [
        'csg562-003', 'csg562-004', 'csg562-005', 'csg562-006',
        'csg562-007', 'csg562-008', 'csg562-009', 'csg562-010',
        'csg562-011', 'csg562-012', 'csg562-013', 'csg562-014',
        'csg562-015', 'csg562-016', 'csg562-017', 'csg562-018',
        'csg562-019', 'csg562-020', 'csg562-021', 'csg562-022',
        'csg562-023', 'csg562-024', 'csg562-025', 'csg562-026',
        'csg562-027', 'csg562-028', 'csg562-029', 'csg562-030',
        'csg562-031', 'csg562-032', 'csg562-033', 'csg562-034',
        'csg562-035', 'csg562-036', 'csg562-037', 'csg562-038',
        'csg562-039', 'csg562-040', 'csg562-041', 'csg562-042',
        'csg562-043', 'csg562-044', 'csg562-045', 'csg562-046',
        'csg562-047', 'csg562-048', 'csg562-049', 'csg562-050',
        'csg562-054', 'csg562-055', 'csg562-056', 'csg562-057',
        'csg562-058', 'csg562-059', 'csg562-060', 'csg562-061',
        'csg562-062', 'csg562-063', 'csg562-064', 'csg562-065'
    ]
    import matplotlib.pyplot as plt
    #plt.title('PSNR')
    images = []
    for i in range(60):
        images.append(i+1)
    plt.plot(
        images,
        [
            44.265707,44.676721,45.411788,44.531670,44.694593,44.252413,45.035256,44.493173,45.007116,44.376873,45.185017,44.943083,44.801391,44.854296,44.900670,44.581012,44.560070,45.088821,44.661495,44.690536,44.719412,44.512057,45.017194,44.697061,44.582661,44.674387,45.198149,44.845690,45.022010,45.046487,44.577858,45.003475,44.808285,44.826332,44.977428,44.758299,44.846555,45.587070,44.693239,44.678502,45.128010,44.642840,44.997605,45.267579,44.823783,44.874219,44.804761,44.596934,43.971378,44.644823,44.543947,44.587669,44.738352,44.536479,44.649925,45.459403,44.884012,44.586786,44.753116,44.195675
        ], '^', label='(Avila-Domenech, 2018)', markersize=16)
    plt.plot(
        images,
        [
            45.392011,46.107066,47.121567,45.719088,46.125177,45.429765,46.623017,45.825137,46.534624,45.600135,46.900828,46.521689,46.377243,46.368646,46.414722,45.829868,45.844900,46.699881,46.020174,46.068334,46.190247,45.743656,46.678985,45.998229,45.865410,46.146931,46.937454,46.327903,46.555601,46.576260,45.912152,46.503355,46.288453,46.229548,46.502112,46.291782,46.392490,47.460034,45.998393,46.090524,46.724017,45.956190,46.553511,47.001680,46.361416,46.256829,46.313121,45.858104,45.101254,45.996842,45.955560,45.892936,46.168550,45.810385,46.077493,47.184328,46.338551,45.974291,46.277087,45.410894
        ], 'ro', label='Proposed', markersize=8)
    plt.ylabel('PSNR')
    plt.legend(loc='upper left', numpoints=1)
    plt.axis([0, 60, 43, 48])
    plt.xticks(images, db_images, size='small', color='k', rotation=-85)
    plt.grid(True)
    font = {
        'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 50}
    plt.rc('font', **font)
    plt.show()


if __name__ == "__main__":
    run_main()