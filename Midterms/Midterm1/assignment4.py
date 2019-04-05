import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('./3_2_s.bmp')

edges = cv2.Canny(img, 200, 200)

plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
