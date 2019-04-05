from skimage import io
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
# import skimage.draw as draw
import skimage.color as color
# from skimage.exposure import histogram
from skimage.future import graph
import numpy as np
import matplotlib.pyplot as plt



img = data.coffee()
labels = seg.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels, mode='similarity')


labels2 = graph.cut_normalized(labels, g)
out = color.label2rgb(labels2, img, kind='avg')
out = seg.mark_boundaries(out, labels2, (0, 0, 0))
io.imsave('out.png',out)
