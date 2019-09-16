#!/usr/bin/env python
# coding: utf-8

# # LOAD PACKAGES!

# In[1]:


import numpy as np
print ("Loading package(s)")


# # PRINT function usages

# In[3]:


print ("Hello, world")

# THERE ARE THREE POPULAR TYPES
# 1. INTEGER
x = 3;
print ("Integer: %01d, %02d, %03d, %04d, %05d" 
       % (x, x, x, x, x))
# 2. FLOAT
x = 123.456;
print ("Float: %.0f, %.1f, %.2f, %1.2f, %2.2f" 
       % (x, x, x, x, x))
# 3. STRING
x = "Hello, world"
print ("String: [%s], [%3s], [%20s]" 
       % (x, x, x))


# # FOR + IF/ELSE 

# In[12]:


dlmethods = ["ANN", "MLP", "CNN", "RNN", "DAE"]

for alg in dlmethods:
    if alg in ["ANN", "MLP"]:
        print ("We have seen %s" % (alg))


# In[ ]:


dlmethods = ["ANN", "MLP", "CNN", "RNN", "DAE"];
for alg in dlmethods:
    if alg in ["ANN", "MLP", "CNN"]:
        print ("%s is a feed-forward network." % (alg))
    elif alg in ["RNN"]:
        print ("%s is a recurrent network." % (alg))
    else:
        print ("%s is an unsupervised method." % (alg))

# Little more advanced?
print("\nFOR loop with index.")
for alg, i in zip(dlmethods, range(len(dlmethods))):
    if alg in ["ANN", "MLP", "CNN"]:
        print ("[%d/%d] %s is a feed-forward network." 
               % (i, len(dlmethods), alg))
    elif alg in ["RNN"]:
        print ("[%d/%d] %s is a recurrent network." 
               % (i, len(dlmethods), alg))
    else:
        print ("[%d/%d] %s is an unsupervised method." 
               % (i, len(dlmethods), alg))


# ## Note that, index starts with 0 ! 

# # Let's make a function in Python

# In[13]:


# Function definition looks like this
def sum(a, b):
    return a+b
X = 10.
Y = 20.
# Usage 
print ("%.1f + %.1f = %.1f" % (X, Y, sum(X, Y)))


# # String operations

# In[14]:


head = "Deep learning" 
body = "very "
tail = "HARD."
print (head + " is " + body + tail)

# Repeat words
print (head + " is " + body*3 + tail)
print (head + " is " + body*10 + tail)

# It is used in this way
print ("\n" + "="*50)
print (" "*15 + "It is used in this way")
print ("="*50 + "\n")

# Indexing characters in the string
x = "Hello, world" 
for i in range(len(x)):
    print ("Index: [%02d/%02d] Char: %s" 
           % (i, len(x), x[i])) 


# In[15]:


# More indexing 
print ""
idx = -2
print ("(%d)th char is %s" % (idx, x[idx]))
idxfr = 0
idxto = 8
print ("String from %d to %d is [%s]" 
       % (idxfr, idxto, x[idxfr:idxto]))
idxfr = 4
print ("String from %d to END is [%s]" 
       % (idxfr, x[idxfr:]))
x = "20160607Cloudy"
year = x[:4]
day = x[4:8]
weather = x[8:]
print ("[%s] -> [%s] + [%s] + [%s] " 
       % (x, year, day, weather))


# # LIST 

# In[16]:


a = []
b = [1, 2, 3]
c = ["Hello", ",", "world"]
d = [1, 2, 3, "x", "y", "z"]
x = []
print x
x.append('a')
print x
x.append(123)
print x
x.append(["a", "b"])
print x
print ("Length of x is %d " 
       % (len(x)))
for i in range(len(x)):
    print ("[%02d/%02d] %s" 
           % (i, len(x), x[i]))


# In[21]:


z = []
z.append(1)
z.append(2)
z.append(3)
z.append('Hello')
for i in range(len(z)):
    print (z[i])


# # DICTIONARY

# In[22]:


dic = dict()
dic["name"] = "Sungjoon"
dic["age"] = 31
dic["job"] = "Ph.D. Candidate"

print dic


# # Class

# In[23]:


class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print ('HELLO, %s!' 
                   % self.name.upper())
        else:
            print ('Hello, %s' 
                   % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"


# In[24]:


def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))
    print


# # RANK 1 ARRAY

# In[28]:


x = np.array([1, 2, 3]) # rank 1 array
print_np(x)

x[0] = 5
print_np(x)


# # RANK 2 ARRAY

# In[29]:


y = np.array([[1,2,3], [4,5,6]]) 
print_np(y)


# # ZEROS

# In[31]:


a = np.zeros((3, 2))  
print_np(a)


# # ONES

# In[32]:


b = np.ones((1, 2))   
print_np(b)


# # IDENTITY

# In[33]:


c = np.eye(2, 2)   
print_np(c)


# # RANDOM (UNIFORM)

# In[34]:


d = np.random.random((2, 2))    
print_np(d)


# # RANDOM (GAUSSIAN)

# In[35]:


e = np.random.randn(1, 10)    
print_np(e)


# # ARRAY INDEXING

# In[ ]:


# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_np(a)

print
# Use slicing to pull out the subarray consisting 
# of the first 2 rows
# and columns 1 and 2; b is the following array 
# of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print_np(b)


# # GET ROW

# In[ ]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_np(a)

row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a

print_np(row_r1)
print_np(row_r2)
print_np(row_r3)


# In[ ]:


a = np.array([[1,2], [3, 4], [5, 6]])
print_np(a)

# An example of integer array indexing.
# The returned array will have shape (3,) and 
b = a[[0, 1, 2], [0, 1, 0]]
print_np(b)

# The above example of integer array indexing 
# is equivalent to this:
c = np.array([a[0, 0], a[1, 1], a[2, 0]])
print_np(c)


# # DATATYPES

# In[ ]:


x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # particular datatype

print_np(x)
print_np(y)
print_np(z)


# ## Array math

# In[36]:


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print x + y
print np.add(x, y)


# In[37]:


#  Elementwise difference; both produce the array
print x - y
print np.subtract(x, y)


# In[38]:


# Elementwise product; both produce the array
print x * y
print np.multiply(x, y)


# In[39]:


# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print x / y
print np.divide(x, y)


# In[40]:


# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print np.sqrt(x)


# In[ ]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])

print_np(x)
print_np(y)
print_np(v)
print_np(w)

# Inner product of vectors; both produce 219
print v.dot(w)
print np.dot(v, w) # <= v * w'
# Matrix / vector product; both produce the rank 1 array [29 67]
print x.dot(v)
print np.dot(x, v) # <= x * v'
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print x.dot(y)
print np.dot(x, y)


# In[ ]:


x = np.array([[1,2],[3,4]])
print_np(x)
print
print x
print x.T
print np.sum(x)  # Compute sum of all elements
print np.sum(x, axis=0)  # Compute sum of each column
print np.sum(x, axis=1)  # Compute sum of each row


# In[ ]:


print x
print x.T


# In[ ]:


v = np.array([1,2,3])
print v 
print v.T


# In[ ]:


v = np.array([[1,2,3]])
print v 
print v.T


# ## Other useful operations

# In[ ]:


# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)    # Create an empty matrix 
                        # with the same shape as x

print_np(x)
print_np(v)
print_np(y)


# In[ ]:


# Add the vector v to each row of the matrix x 
# with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v
print_np(y)


# In[ ]:


vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print_np(vv)             # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"


# In[ ]:


# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using BROADCASTING
print_np(x)
print_np(v)
print_np(y)


# In[ ]:


# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])

print_np(x)
print_np(v)
print x + v


# In[ ]:


# Add a vector to each column of a matrix
print_np(x)
print_np(w)
print (x.T + w).T

# Another solution is to reshape w 
# to be a row vector of shape (2, 1);
print
print x + np.reshape(w, (2, 1))


# ## Matplotlib

# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)


# In[43]:


y_sin = np.sin(x)
y_cos = np.cos(x)
# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])

# Show the figure.
plt.show()


# In[44]:



# Compute the x and y coordinates for points 
# on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

