
a=type([1,2])
print(a)
print(type(a))
b={}.get(a)
print(b)
import numpy as np
x = np.arange(16).reshape(-1,4)
print(np.where(x>5))

#(array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=int64), array([2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
#注意这里是坐标是前面的一维的坐标，后面是二维的坐标
