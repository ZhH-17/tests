import cv2 as cv
import numpy as np
import os

path = "./data"
fns = [os.path.join(path, fn) for fn in os.listdir(path)]

imgs= []
for fn in fns:
    img = cv.imread(fn, -1)
    imgs.append(img)

destpath = "test1"
if not os.path.isdir(destpath):
    os.mkdir(destpath)
for i, img in enumerate(imgs):
    cv.imwrite(os.path.join(destpath, "%d.jpg" %i), img)
