import os

import matplotlib.pyplot as plt
import numpy as np

with open('ImageSets/Segmentation/train.txt', 'r') as fp:
    files_train = [line.rstrip() for line in fp.readlines()]

with open('ImageSets/Segmentation/val.txt', 'r') as fp:
    files_val = [line.rstrip() for line in fp.readlines()]

files_train = [f for f in files_train if os.path.isfile(os.path.join('SegmentationClassSubset/' + f + '.npy'))]
files_val = [f for f in files_val if os.path.isfile(os.path.join('SegmentationClassSubset/' + f + '.npy'))]
files_all = np.array(sorted(list(set(files_train).union(set(files_val)))))

for i, fname in enumerate(files_all):
    if i % 100 == 0:
        print('File', i, fname)

    infname = os.path.join('SegmentationClassSubset/' + fname + '.npy')
    outfname = os.path.join('PNGSegmentation/' + fname + '.png')

    with open(infname, 'rb') as f:
        image_array = np.load(f)
        # save with grayscale so that we can load with one color channel
        plt.imsave(outfname, image_array[:,:,0], vmin=-1, vmax=6, cmap='gray')
