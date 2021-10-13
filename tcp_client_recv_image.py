import time
import numpy as np
import sys
import platform
import socket
import cv2 as cv

ver = sys.version_info.major
# ip = "192.168.2.145"
# port = 2345
# sk = socket.socket()
# sk.connect((ip, port))
sk = 0
h, w = 1080, 3840
global img
img = np.ones((h, w), dtype=np.uint8)*1
img[:200, :] = 200

def str_bytes(s, encoding="utf8"):
    if ver > 2:
        return s.encode(encoding=encoding)
    else:
        return s

def send_message(sk, data):
    sk.send(str_bytes(data))
    time.sleep(0.01)

def recv_img(sk):
    # received_all = np.array([], dtype=np.uint8)
    # while True:
    #     received = conn.recv(1024*1000)
    #     if len(received) > 0:
    #         received = np.ndarray(buffer=received, shape=(len(received), ), dtype='<u1')
    #         if len(received_all)<1:
    #             received_all = received.copy()
    #         else:
    #             received_all = np.concatenate([received_all, received])
    #         if len(received_all) >= plength:
    #             image = received_all
    #             break
    # return image.reshape(h, w)

    img = np.ones((h, w), dtype=np.uint8)*100
    img[::100, :] = 200
    return img

def set_expose_time(expose_time):
    '''
    set expose time and send message then recv image
    '''
    global img
    # send_message(sk, str_bytes(str(expose_time)))
    # img = recv_img(sk)
    img[::200, :] = 0

if __name__ == "__main__":
    windowname = "image"
    cv.namedWindow(windowname, cv.WINDOW_NORMAL)
    trackname = 'time'
    cv.createTrackbar(trackname, windowname, 0, 1123, set_expose_time)
    cv.createTrackbar("start", windowname, 0, 1, lambda x: print(x))

    while True:
        cv.imshow(windowname, img)
        k = cv.waitKey()
        if k == 27:
            break

