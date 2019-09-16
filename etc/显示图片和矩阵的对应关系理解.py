# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from skimage import io,data
img='C:/Users/zhangbo284/Desktop/python_all_project/srcAll/face/facenet-master/data/images/data_set/KA.AN3.41.tiff'
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np





img='C:/Users/zhangbo284/Desktop/python_all_project/srcAll/face/facenet-master/data/images/Anthony_Hopkins_0002.jpg'
#result=img.replace("\","/")
print(img)
#raise
lena = mpimg.imread(img) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#lena.shape #(512, 512, 3)
print(lena.shape)
print(type(lena))
lena=lena[:200,:200,:]
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

'''
下面学习一下，图像跟矩阵的关系。
lena=lena[:200,:200,:] 从这一行就可以看出来,图像跟矩阵的关系,矩阵的左上角的数表示图像的左上角的像素点!!!!!!!!
其他点也都是对应排列即可!
记忆起来就是:图像的矩阵里面的横纵角标都是越小越接近图像的左上角.反之越接近图像的右下角!记住这2个趋势即可.
'''


