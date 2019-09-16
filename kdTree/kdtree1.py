#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree


# help(KDTree)





np.random.seed(0)
points = np.random.random((3, 2))
tree = KDTree(points)
point = [0.1,0.1]
# kNN
dists, indices = tree.query([point], k=1)
print(dists, indices)
print(points[indices].flatten(),'最接近点的向量')

print(dists.flatten()[0],'最接近点和视频里面图像的距离')


