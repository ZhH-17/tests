import cv2 as cv
import numpy as np

img = np.zeros((10, 20), dtype=np.uint8)
img[-1, :] = img[-1, :] +100
cv.imwrite("test.png", img)
cv.imwrite("test.bmp", img)
try:
    cv.imwrite("testimg", img)
except:
    print("can not save in file name without right subfix")
