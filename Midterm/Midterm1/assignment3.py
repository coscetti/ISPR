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
# defining a simple function to plot the images
# def image_show(image, nrows=1, ncols=1, cmap='gray'):
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))
#     ax.imshow(image, cmap='gray')
#     ax.axis('off')
#     plt.show()
#     return fig, ax

# img = data.coffee()
# image_show(coffee)
img = io.imread('./Datasets/MSRC_ObjCategImageDatabase_v1/1_2_s.bmp')
# ####################
# # seg.slic segments image using k-means clustering in Color-(x,y,z) space.
# # n_segments - the approximate number of labels in the segmented output image.
# # compactness -
# labels1 = seg.slic(img, compactness=30, n_segments=400, max_iter=10)
# out1 = color.label2rgb(labels1, img, kind='avg')
#
# labels3 = seg.slic(img, compactness=20, n_segments=400, max_iter=20)
# out3 = color.label2rgb(labels3, img, kind='avg')
#
# # graph.rag_mean_color - compute the Region Adjacency Graph using mean colors.
# # mode = distance, similarity
# g1 = graph.rag_mean_color(img, labels1, mode='distance')
# # graph.cut_normalized - perform Normalized Graph cut on the RAG
# labels2 = graph.cut_normalized(labels1, g1)
# out2 = color.label2rgb(labels2, img, kind='avg')
#
# g2 = graph.rag_mean_color(img, labels1, mode='similarity')
# # graph.cut_normalized - perform Normalized Graph cut on the RAG
# labels4 = graph.cut_normalized(labels1, g2)
# out4 = color.label2rgb(labels4, img, kind='avg')
#
# g3 = graph.rag_mean_color(img, labels3, mode='distance')
# # graph.cut_normalized - perform Normalized Graph cut on the RAG
# labels5 = graph.cut_normalized(labels3, g3)
# out5 = color.label2rgb(labels5, img, kind='avg')
#
# g2 = graph.rag_mean_color(img, labels3, mode='similarity')
# # graph.cut_normalized - perform Normalized Graph cut on the RAG
# labels6 = graph.cut_normalized(labels3, g3)
# out6 = color.label2rgb(labels6, img, kind='avg')

# ####################

# fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(6, 8))
# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(6, 8))
#
# ax[0, 0].imshow(out1)
# ax[1, 0].imshow(out2)
# ax[0, 1].imshow(out3)
# ax[2, 0].imshow(out4)
# ax[1, 1].imshow(out5)
# ax[2, 1].imshow(out6)
#
# # for a in ax:
# #     a.axis('off')
#
# plt.tight_layout()
# plt.show()
#
print("slic")
labels = seg.slic(img, compactness=30, n_segments=400)
print("rag")
g = graph.rag_mean_color(img, labels, mode='similarity')

print("cut normalized")
labels2 = graph.cut_normalized(labels, g, num_cuts=10)
out = color.label2rgb(labels2, img, kind='avg')
print("boundaries")
out = seg.mark_boundaries(out, labels2, (0, 0, 0))
io.imsave('out1.png',out)
# # importing a grayscale image from the skimage library
# image = data.binary_blobs()
# plt.imshow(image, cmap='gray')
# plt.show()

# importing multiple images
# images = io.ImageCollection('.\Datasets\MSRC_ObjCategImageDatabase_v1\*.bmp')
# print('Type:', type(images))
# images.files
# image_show(images[0])

# image = io.imread('./Datasets/MSRC_ObjCategImageDatabase_v1/1_8_s.bmp')
# image_show(image)
# plt.show(image)
# fig, ax = plt.subplots(1, 1)
# ax.hist(image.ravel(), bins=32, range=[0, 256])
# ax.set_xlim(0, 256)
# plt.show()

# plt.imshow(image)
# image_segmented = image > 150
# image_show(image_segmented)
# image_show(image)
# image_gray = color.rgb2gray(image)
# image_show(image_gray);
# #
# def circle_points(resolution, center, radius):
#     """
#     Generate points which define a circle on an image.Centre refers to the centre of the circle
#     """
#     radians = np.linspace(0, 2*np.pi, resolution)
#     c = center[1] + radius*np.cos(radians)#polar co-ordinates
#     r = center[0] + radius*np.sin(radians)
#
#     return np.array([c, r]).T
# # Exclude last point because a closed path should not have duplicate points
# points = circle_points(50, [80, 50], 30)[:-1]
#
# fig, ax = image_show(image)
# ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

# hist, hist_centers = histogram(image_gray)
