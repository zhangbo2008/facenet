import numpy as np
y=np.array([1,2,3])
x=np.array([1,2,3])
a=np.vstack([y,x])
a=[121 ,137,92,85,55 ,109,42,60, 152,32,48,29,47,44,23,43,37,32
,32, 81,24,22,63,43,43 ,112,29,97,29,82,24 ,117,22,21,21,35
,35,21 ,199,41,24,36,59,46,35, 129,79,63,96,61,22,34,86,23
,62,24,30]
print(a[38])
print(True or True and False )

'''
对于包含and，not，or的表达式，通过优先级关系，处理起来也是较为简单的。利用短路逻辑规则：表达式从左至右运算，若 or 的左侧逻辑值为 True ，则短路 or 后所有的表达式（不管是 and 还是 or），直接输出 or 左侧表达式 。表达式从左至右运算，若 and 的左侧逻辑值为 False ，则短路其后所有 and 表达式，直到有 or 出现，输出 and 左侧表达式False到 or 的左侧，参与接下来的逻辑运算。若 or 的左侧为 False ，或者 and 的左侧为 True 则不能使用短路逻辑。

这个就是因为and只管到后面and语句,也就是or前面.因为or能修改结果.
而or语句如果前面是true那么也短路到后面的and语句.因为这时候只有and能改结果.
--------------------- 
作者：mmdnxh 
来源：CSDN 
原文：https://blog.csdn.net/qq_28267025/article/details/62044871 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''

'''
A and B or C and D
等价于(A and B) or (C and D)
证明:就是用上面的局部短路规则.
'''



print(1 and 1 or 0 and 0)
a=np.array([1,3])
print(a.shape[2])

'''
numpy.tile(A, reps)[source]
Construct an array by repeating A the number of times given by reps.
'''

