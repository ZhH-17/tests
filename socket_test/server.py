import socket
import numpy as np
import cv2 as cv
import pdb

h, w = 1080, 1080
h, w = 100, 50
plength = w*h

# host = socket.gethostname()
host = "127.0.0.1"
port = 2345
sk = socket.socket()
sk.bind((host, port))
sk.listen(5)

def recv_img(conn):
    received_all = np.array([], dtype=np.uint8)
    while True:
        received = conn.recv(1024*1000)
        if len(received) > 0:
            received = np.ndarray(buffer=received, shape=(len(received), ), dtype='<u1')
            if len(received_all)<1:
                received_all = received.copy()
            else:
                received_all = np.concatenate([received_all, received])
            if len(received_all) >= plength:
                image = received_all
                break

    return image.reshape(h, w)

def connection(conn, add):
    # for a client connection
    while True:
        received = conn.recv(1024)

        if received:
            cmd = received.decode()
            if cmd == "close":
                msg = "finish"
                conn.sendall(msg.encode())
                break
            elif cmd == 'img':
                msg = "ready"
                conn.sendall(msg.encode())
                img = recv_img(conn)
                cv.imwrite("test.jpg", img)
                msg = "received"
                conn.sendall(msg.encode())
            elif cmd == 'test':
                msg = "test end"
                conn.sendall(msg.encode())


while True:
    # wait for connection
    conn, add = sk.accept()
    print(conn, add)
    connection(conn, add)
