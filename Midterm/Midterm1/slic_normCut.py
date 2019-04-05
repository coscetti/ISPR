import numpy as np
import matplotlib.pyplot as plt

from skimage import io
import skimage.data as data
import skimage.segmentation as seg
import skimage.color as color
from skimage.future import graph

import os.path

img = io.imread('./face.jpg')
labels = seg.slic(img, compactness=30, n_segments=400)
out_slic = color.label2rgb(labels, img, kind='avg')
g = graph.rag_mean_color(img, labels, mode='similarity')
labels2 = graph.cut_normalized(labels, g, num_cuts=10)
out = color.label2rgb(labels2, img, kind='avg')
out_cut = seg.mark_boundaries(out, labels2, (0, 0, 0))

fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
                        sharey=True, figsize=(6, 8))

ax[0].imshow(img)
ax[1].imshow(out_slic)
ax[2].imshow(out_cut)

for a in ax:
    a.axis('off')

    plt.tight_layout()
plt.show()

    # fig.savefig(save_path + str(n) + '.png', transparent=True)
