import socket
import numpy as np
import cv2 as cv
import time
import pdb

host = socket.gethostname()
port = 2345
sk = socket.socket()
sk.connect((host, port))

# sk.send("close".encode())

h, w = 1080, 1080
img = np.ones((h, w))
img[10:30] = 200.
img = img.astype(np.uint8)
cv.imwrite("send.jpg", img)

t0 = time.time()
for i in range(100):
    sk.sendall("img".encode())
    rec = sk.recv(1024)
    print(i, rec)

    sk.sendall(img.tobytes())
    rec = sk.recv(1024)
    print(i, rec)
print("time: ", time.time()-t0)
