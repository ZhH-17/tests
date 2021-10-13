import numpy as np
import cv2 as cv

w, h = 1920, 1080
lengthToPixle = 8. / 2000

def objPoints():
    nrow = 6
    ncol = 9
    points = np.zeros((nrow, ncol, 3), np.float)
    Z = 20
    points[:, :, 2] = Z
    dx, dy = 1, 1
    for i in range(nrow):
        for j in range(ncol):
            points[i, j, 0] = j * dx
            points[i, j, 1] = i * dy - 3
    points = points.reshape(-1, 3)
    return points

def planeToCylinder(pnts, K):
    # pnts: N * 2 (u, v)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pnts_ = []
    for pt in pnts:
        u, v = pt
        u_ = fx * np.arctan((u - cx)/fx) + cx
        v_ = fy * (v - cy) / np.sqrt(fy**2 + (u - cx)**2) + cy
        pnts_.append([u_, v_])
    return np.array(pnts_, dtype=np.float32)

points = objPoints()
rOpticRotate = 1
opticCenter = np.array([0, 0, rOpticRotate])

K = np.array([
    [2000, 0, 960],
    [0, 2000, 540],
    [0, 0, 1]], dtype=float)
dist = np.zeros(5)

T0 = np.eye(4)
T0[:3, 3] = opticCenter
Ts = [T0]

vec = np.array([0, 1, 0], dtype=float)
rotateDegree =  -np.pi / 100.
vec *= rotateDegree
runit, _ = cv.Rodrigues(vec)
Tunit = np.zeros((4, 4))
Tunit[:3, :3] = runit
Tunit[3, 3] = 1.
degree = 0
for i in range(10):
    degree += rotateDegree
    lastCenter = opticCenter
    opticCenter[0] = rOpticRotate * np.sin(degree)
    opticCenter[2] = rOpticRotate * np.cos(degree)
    T = np.eye(4)
    T[:3, :3] = runit
    T[:3, 3] = opticCenter - lastCenter
    Ts.append(T)


img_pnts = []
img_pnts_cylinder = []
imgs = []
imgs_cylinder = []
for i in range(len(Ts)):
    T = Ts[i]
    r = T[:3, :3]
    t = T[:3, 3]
    pnts_prj, _ = cv.projectPoints(np.expand_dims(points, 0), r, t, K, dist)
    pnts_prj = pnts_prj.reshape(-1, 2).astype(np.float32)
    img_pnts.append(pnts_prj)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for ip, pt in enumerate(pnts_prj):
        cv.circle(img, tuple(pt), 5, [255, 0, 255], 2)
        cv.putText(img, str(ip), tuple(pt), 5, 1, [0, 0, 255])
    cv.imwrite("origin%d.png" %i, img)
    imgs.append(img)

    pnts_cy = planeToCylinder(pnts_prj, K)
    img_pnts_cylinder.append(pnts_cy)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for ip, pt in enumerate(pnts_cy):
        cv.circle(img, tuple(pt), 5, [255, 0, 255], 2)
        cv.putText(img, str(ip), tuple(pt), 5, 1, [0, 0, 255])
    cv.imwrite("origin%d_cy.png" %i, img)
    imgs_cylinder.append(img)

# img_merge = imgs[0].copy()
# for i in range(1, len(imgs)):
#     H, _ = cv.findHomography(img_pnts[i], img_pnts[0])
#     cv.warpPerspective()



# K = np.loadtxt("./camera_para.txt", skiprows=2, max_rows=3)
# rvecs = np.loadtxt("./camera_para.txt", skiprows=8, max_rows=3)
# tvecs = np.loadtxt("./camera_para.txt", skiprows=24, max_rows=3)
# dist = np.loadtxt("./camera_para.txt", skiprows=6, max_rows=1)

# pnts_prj, _ = cv.projectPoints(points, np.zeros((3,1)), np.zeros((3,1)), K, None)
# # pnts_prj, _ = cv.projectPoints(points, rvecs[0].reshape(3,1), tvecs[0].reshape(3,1), K, None)
# pnts_prj = pnts_prj[:, 0].astype(np.int)

# img = np.zeros((h, w, 3), dtype=np.uint8)
# for pt in pnts_prj:
#     print(pt)
#     # cv.drawMarker(img, tuple(pt), [0, 0, 255], 2, 10)
#     cv.circle(img, tuple(pt), 5, [0, 0, 255], 2)
# cv.imwrite("tmp.png", img)
