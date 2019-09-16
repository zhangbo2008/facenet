'''
测试一下到底for i in list1快
还是for i in range(len(list1)):快
'''






import time
start=time.time()
list1=[999]*100000
for i in list1:
    tmp=i
end=time.time()
print(start-end) 

import time
start=time.time()
list1=[999]*100000
for i in range(len(list1)):
    tmp=list1[i]
end=time.time()
print(start-end) 

'''
答案:
-0.01604437828063965
-0.03509187698364258  所以还是第一个快.所以尽量不要用索引.
'''
import numpy as np
import time
start=time.time()
list1=np.array([999])
list1=np.repeat(list1, 100000, 0)
# print(list1)
for i in list1:
    tmp=list1
end=time.time()
print(start-end) 
'''
-0.013030767440795898        所以单纯的数组遍历用list就可以,不用numpy
-0.021056413650512695
-0.015040159225463867
'''






import numpy as np
import time
start=time.time()
list1=np.array([999])
list2=np.array([999])
# print(list1)
for i in range(100000):
    list1=np.append(list1, list2, 0)
end=time.time()
print(start-end) 
# print(list1)

import numpy as np
import time
start=time.time()
list1=[999]

# print(list1)
for i in range(100000):
    list1.append( 999)
end=time.time()
print(start-end) 
# print(list1)
'''
结论np.append很慢  不如list 的append
'''
