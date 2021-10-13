import numpy as np
import cv2 as cv
import pdb

video = "/home/zhangh/Downloads/opencv-3.4.6/samples/data/vtest.avi"
cap = cv.VideoCapture(video)

def extract_bg_diff(cap):
    flag, img = cap.read()
    bg = np.zeros_like(img).astype(int)

    times_bg = np.zeros_like(img)
    num = 0
    while True:
        flag, cur = cap.read()
        print(num)
        num += 1
        if not flag:
            break
        mask1 = (img - cur) < 2
        mask2 = (cur - img) < 2
        mask = mask1 + mask2
        bg[mask] = cur[mask]
        img = cur.copy()
    cv.imwrite("bg.png", bg)

def extract_bg_mean(cap):
    flag, img = cap.read()
    bg = img.copy().astype(float)
    num = 0
    while True:
        flag, cur = cap.read()
        if not flag:
            break
        print(num)
        num += 1
        bg = bg + cur
    cv.imwrite("bg.png", bg/(num+1))

def extract_bg_mog(cap):
    bgfg = cv.createBackgroundSubtractorMOG2()
    num = 0
    while True:
        flag, img = cap.read()
        if not flag:
            break
        print(num)
        num += 1
        mask = bgfg.apply(img)
        print(mask.shape)
        # cv.imshow("mask", mask)
        img_show = img.copy()
        img_show[mask>200] = 0
        cv.imshow('Frame', img_show)

        k = cv.waitKey(-1)
        if k == 27:
            break

    cv.destroyAllWindows()

def extract_bg(cap ,method="mean"):
    if method == "mean":
        extract_bg_mean(cap)
    elif method == "diff":
        extract_bg_diff(cap)
    elif method == "mog":
        extract_bg_mog(cap)

if __name__ == "__main__":
    print("background")
    extract_bg(cap, "mog")

