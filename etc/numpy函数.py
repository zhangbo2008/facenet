import numpy as np
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
print (np.vstack((arr1,arr2)))
 
print (np.hstack((arr1,arr2)))
 
a1=np.array([[1,2],[3,4],[5,6]])
a2=np.array([[7,8],[9,10],[11,12]])
print (a1)
print (a2)
print (np.hstack((a1,a2)))
