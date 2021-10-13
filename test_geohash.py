from geohash import neighbors
import geohash
import pandas as pd
import numpy as np
import time
'''
geohash长度	Lat位数	Lng位数	Lat误差	Lng误差	km误差
1	2	3	±23	±23	±2500
2	5	5	± 2.8	±5.6	±630
3	7	8	± 0.70	± 0.7	±78
4	10	10	± 0.087	± 0.18	±20
5	12	13	± 0.022	± 0.022	±2.4
6	15	15	± 0.0027	± 0.0055	±0.61
7	17	18	±0.00068	±0.00068	±0.076
8	20	20	±0.000086	±0.000172	±0.01911
9	22	23	±0.000021	±0.000021	±0.00478
10	25	25	±0.00000268	±0.00000536	±0.0005971
11	27	28	±0.00000067	±0.00000067	±0.0001492
12	30	30	±0.00000008	±0.00000017 ±0.0000186

init summary:
 use geohash is slow than compute directly when data is not huge
(7870000, 3)
find : 1.9293441772460938
dist : 2.482185125350952

(78700, 3)
find : 0.7677609920501709
dist : 0.020456314086914062

'''


def haversine(lon1, lat1, lon2, lat2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000.


def is_really_near(df_gps, inds, lon_c, lat_c, dist_thresh):
    dists = haversine(df_gps.iloc[inds]['lon'].tolist(), df_gps.iloc[inds]['lon'].tolist(),
                      lon_c, lat_c)
    inds_really = inds[np.where(dists < dist_thresh)[0]]
    return dists, inds_really


def find_near(df_gps, lon_c, lat_c, dist_thresh=1000):
    pre_in = 6 # determined by dist_thresh
    center_hash = geohash.encode(lat_c, lon_c, pre_in)

    inds = [i for i in range(len(df_gps)) if df_gps.iloc[i]['hashcode'].startswith(center_hash)]
    neighbors = geohash.neighbors(center_hash)
    inds_nei = []
    for nei in neighbors:
        inds_nei.extend(
            [i for i in range(len(df_gps)) if df_gps.iloc[i]['hashcode'].startswith(nei)])
    return inds, inds_nei


precision = 7
gps_data = np.loadtxt("./tmp_gps.txt")
df = pd.read_csv("./tmp_gps.txt", header=None, delimiter=' ', names=['lon', 'lat'])
df['hashcode'] = df.apply(lambda x: geohash.encode(x['lat'], x['lon'], precision), axis=1)
lon_c, lat_c = df.iloc[0]['lon'], df.iloc[0]['lat']
t0 = time.time()
inds, inds_nei = find_near(df, lon_c, lat_c)
df = pd.concat([df]*100, axis=0)
print(df.shape)
t1 = time.time()
print('find :', t1-t0)

dists = is_really_near(df, np.arange(len(df)), lon_c, lat_c, 1000)
t2 = time.time()
print('dist :', t2-t1)
