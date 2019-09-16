#!/usr/bin/env python
# coding: utf-8

# # BASIC TENSORFLOW

# # LOAD PACKAGES

# In[2]:


import numpy as np
import tensorflow as tf
print ("PACKAGES LOADED")


# # SESSION

# In[3]:


sess = tf.Session()
print ("OPEN SESSION")


# # TF CONSTANT

# In[4]:


def print_tf(x):
    print("TYPE IS\n %s" % (type(x)))
    print("VALUE IS\n %s" % (x))
hello = tf.constant("HELLO. IT'S ME. ")
print_tf(hello)


# # TO MAKE THINKS HAPPEN

# In[4]:


hello_out = sess.run(hello)
print_tf(hello_out)


# # OTHER TYPES OF CONSTANTS

# In[5]:


a = tf.constant(1.5)
b = tf.constant(2.5)
print_tf(a)
print_tf(b)


# In[6]:


a_out = sess.run(a)
b_out = sess.run(b)
print_tf(a_out)
print_tf(b_out)


# # OPERATORS

# In[7]:


a_plus_b = tf.add(a, b)
print_tf(a_plus_b)


# In[8]:


a_plus_b_out = sess.run(a_plus_b)
print_tf(a_plus_b_out)


# In[9]:


a_mul_b = tf.mul(a, b)
a_mul_b_out = sess.run(a_mul_b)
print_tf(a_mul_b_out)


# # VARIABLES

# In[10]:


weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
print_tf(weight)


# In[11]:


weight_out = sess.run(weight)
print_tf(weight_out)


# # WHY DOES THIS ERROR OCCURS?

# In[12]:


init = tf.initialize_all_variables()
sess.run(init)
print ("INITIALIZING ALL VARIALBES")


# # ONCE, WE INITIALIZE VARIABLES

# In[13]:


weight_out = sess.run(weight)
print_tf(weight_out)


# # PLACEHOLDERS

# In[15]:


x = tf.placeholder(tf.float32, [None, 5])
print_tf(x)


# ## OPERATION WITH VARIABLES AND PLACEHOLDERS

# In[18]:


oper = tf.matmul(x, weight)
print_tf(oper)


# In[21]:


data = np.random.rand(1, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)


# In[22]:


data = np.random.rand(2, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)

