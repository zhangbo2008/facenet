
a=type([1,2])
print(a)
print(type(a))
b={}.get(a)
print(b)
import numpy as np
x = np.arange(16).reshape(-1,4)
print(np.where(x>5))
'''
从结果可以看出来,返回的是一个多维tuple,第i个tuple代表第i维.顺序是字典序排列.
对应下面结果就是index是(1,2) (1,3) (2,0).....这些位置的x上的数字是大于5的
'''
#(array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=int64), array([2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
#注意这里是坐标是前面的一维的坐标，后面是二维的坐标
