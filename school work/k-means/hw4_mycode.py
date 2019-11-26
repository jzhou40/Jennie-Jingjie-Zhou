#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:53:25 2019

@author: chengzhao
"""
from scipy.io import arff
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
data = arff.loadarff('segment.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode("utf-8") 
label = df['class']
df = df.drop(['class'],axis = 1)
df = df.apply(stats.zscore)
df = df.fillna(0)

def distance(a,b):
    return np.linalg.norm(a - b)
def k_means(k,centroid):
    it = 1
    while it <= 50:
        cluster = {}
        for c in range(k):
            cluster[c] = []
        for i in range(len(df)):
            dist = float('inf')
            for j in range(k):
                d_each = distance(df.iloc[i,:],centroid[j])
                if dist > d_each:
                    dist = d_each
                    cl = j         
            cluster[cl].append(i)
        last_centroid = centroid
        centroid = []
        for n in range(k):
            centroid.append(np.mean(df.iloc[cluster[n],:],axis=0))
        TorF = []    
        for n in range(k):
            TorF+=[(last_centroid[n]==centroid[n]).all()]
        if all(TorF) == True:
            sse = 0
            for f in range(k):
                for item in range(len(cluster[f])):
                    sse += distance(df.iloc[cluster[f][item],:],centroid[f])**2
            return (cluster,sse)
        it += 1
    sse = 0
    for f in range(k):
        for item in range(len(cluster[f])):
            sse += distance(df.iloc[cluster[f][item],:],centroid[f])**2    
    return (cluster,sse)

indices = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422,
           88, 105, 261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 
           1283, 1288, 1766, 1168, 566, 1812, 214, 53, 423, 50, 705, 1284, 1356, 
           996, 1084, 1956, 254, 711, 1997, 1378, 827, 1875, 424, 1790, 633, 
           208, 1670, 1517, 1902, 1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 
           672, 483, 65, 92, 400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 
           1134, 1462, 954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, 1635, 
           1078, 640, 833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 
           520, 1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 
           1923, 1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 
           1068, 1493, 859, 233, 1846, 1119, 469, 1869, 609, 385, 1182, 1949, 
           1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339, 826, 530, 1173, 
           802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641, 
           661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 
           741, 583, 1464, 209, 1615, 475, 1903, 555, 1046, 379, 1938, 417, 
           1747, 342, 1148, 1697, 1785, 298, 1485, 945, 1097, 207, 857, 1758, 
           1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427, 1434, 
           953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 
           1825, 249, 240, 524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 
           1904, 561, 1270, 1250, 613, 152, 1440, 473, 1834, 1387, 1656, 1028, 
           1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611, 1776, 123, 979, 
           1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 
           1686, 854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 
           582, 816, 1770, 663, 737, 1810, 523, 1243, 944, 1959, 78, 675, 135, 
           1381, 1472] 
res = {}
K = []
for k in range(1,2):
    sse_list = []
    K.append(k)
    for iteration in range(25):
        init = indices[iteration*k:(iteration+1)*k]
        centroid = []
        for i in init:
            centroid.append(df.iloc[i-1,:])
        (cluster,sse) = k_means(k,centroid)
        sse_list.append(sse)
    res[k] = [np.mean(sse_list),np.std(sse_list)]
print(res)
plot_sse = pd.DataFrame(res,index = ['mu','std']).transpose()
fig, ax = plt.subplots()
plt.errorbar(K, plot_sse.mu, yerr=2*plot_sse['std'])
ax.set_xlabel('K')
ax.set_ylabel('mean SSE')
ax.set_title('95% confidence interval of mean SSE')



     