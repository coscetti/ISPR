# Perform image segmentation on one of the eight image thematic subsets.
# Use the normalized cut algorithm to perform image segmentation.
# You are welcome to confront the result with the kmeans segmentation algorithm.

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
import skimage.data as data
import skimage.segmentation as seg
# import skimage.filters as filters
# import skimage.draw as draw
import skimage.color as color
# from skimage.exposure import histogram
from skimage.future import graph

import os.path

save_path = './output/'

# print("Load images")
# images = io.ImageCollection('./Datasets/MSRC_ObjCategImageDatabase_v1/1_*_s.bmp')
# print(len(images))
# for i in range(len(images)):
#     print(images[i].shape)
for i in range(1, 9):
    images = io.ImageCollection('./Datasets/MSRC_ObjCategImageDatabase_v1/' + \
                                    str(i) + '_*_s.bmp')

    print(len(images))
    n = 1
    save_path = './output/'
    save_path += str(i) + '/'
    for img in images:
        print("slic")
        labels = seg.slic(img, compactness=30, n_segments=400)
        out_slic = color.label2rgb(labels, img, kind='avg')

        print("rag")
        g = graph.rag_mean_color(img, labels, mode='similarity')

        print("cut normalized")
        labels2 = graph.cut_normalized(labels, g, num_cuts=10)
        out = color.label2rgb(labels2, img, kind='avg')
        print("boundaries")
        out_cut = seg.mark_boundaries(out, labels2, (0, 0, 0))

        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                                sharey=True, figsize=(6, 8))

        ax[0].imshow(img)
        ax[1].imshow(out_slic)
        ax[2].imshow(out_cut)

        for a in ax:
            a.axis('off')

            plt.tight_layout()
            # plt.show()

            fig.savefig(save_path + str(n) + '.png', transparent=True)
            # io.imsave(save_path + str(n) + '.png',out)
        n += 1
#
# print("slic")
# labels = seg.slic(img, compactness=30, n_segments=400)
# print("rag")
# g = graph.rag_mean_color(img, labels, mode='similarity')
#
# print("cut normalized")
# labels2 = graph.cut_normalized(labels, g, num_cuts=10)
# out = color.label2rgb(labels2, img, kind='avg')
# print("boundaries")
# out = seg.mark_boundaries(out, labels2, (0, 0, 0))
# io.imsave('out1.png',out)
