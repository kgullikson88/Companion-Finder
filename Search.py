import sys

import GenericSearch









# Define regions contaminated by telluric residuals or other defects. We will not use those regions in the cross-correlation
badregions = [[0, 475],
              # [460, 475],
              [567.5, 575.5],
              [588.5, 598.5],
              [627, 632],
              [647, 655],
              [686, 706],
              [716, 734],
              [759, 9e9],
              # [655, 657],  # H alpha
              # [485, 487],  #H beta
              # [433, 435],  #H gamma
              # [409, 411],  #H delta
              #[396, 398],  #H epsilon
              #[388, 390],  #H zeta
]
interp_regions = []
trimsize = 10

if "darwin" in sys.platform:
    modeldir = "/Volumes/DATADRIVE/Stellar_Models/Sorted/Stellar/Vband/"
elif "linux" in sys.platform:
    modeldir = "/media/FreeAgent_Drive/SyntheticSpectra/Sorted/Stellar/Vband/"
else:
    modeldir = raw_input("sys.platform not recognized. Please enter model directory below: ")
    if not modeldir.endswith("/"):
        modeldir = modeldir + "/"

if __name__ == '__main__':
    # Parse command line arguments:
    fileList = []
    for arg in sys.argv[1:]:
        if 1:
            fileList.append(arg)

    vsini_list = [None for f in fileList]

    GenericSearch.slow_companion_search(fileList, vsini_list,
                                        hdf5_file='/media/ExtraSpace/PhoenixGrid/TS23_Grid.hdf5',
                                        extensions=True,
                                        resolution=None,
                                        trimsize=trimsize,
                                        modeldir=modeldir,
                                        badregions=badregions,
                                        metal_values=(0.0, ),
                                        vsini_values=(1, 5,),
                                        Tvalues=range(3000, 7000, 200),
                                        observatory='McDonald',
                                        debug=False,
                                        vbary_correct=False,
                                        addmode='simple',
                                        obstype='real',
                                        output_mode='hdf5')

