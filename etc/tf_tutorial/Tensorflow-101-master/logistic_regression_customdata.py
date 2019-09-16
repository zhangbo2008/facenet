#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION WITH CUSTOM DATA

# In[1]:


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print ("Packages loaded")


# # Load data

# In[2]:


# Load them!
cwd = os.getcwd()
loadpath = cwd + "/data/custom_data.npz"
l = np.load(loadpath)

# See what's in here
print (l.files)

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
use_gray = l['use_gray']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("%d classes" % (nclass))


# # Define network

# In[3]:


tf.set_random_seed(0)
# Parameters of Logistic Regression
learning_rate   = 0.001
training_epochs = 1000
batch_size      = 10
display_step    = 100

# Create Graph for Logistic Regression
x = tf.placeholder("float", [None, dim]) 
y = tf.placeholder("float", [None, nclass]) 
W = tf.Variable(tf.zeros([dim, nclass]), name = 'weights')
b = tf.Variable(tf.zeros([nclass]))


# # Define functions

# In[4]:


WEIGHT_DECAY_FACTOR = 1 #  0.000001
l2_loss = tf.add_n([tf.nn.l2_loss(v) 
            for v in tf.trainable_variables()])
_pred = tf.nn.softmax(tf.matmul(x, W) + b) 
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(_pred)
                                     , reduction_indices=1)) 
cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
optm = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cost) 
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.initialize_all_variables()
print ("Functions ready")


# # Optimize

# In[5]:


# Launch the graph
sess = tf.Session()
sess.run(init)
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(ntrain/batch_size)
    # Loop over all batches
    for i in range(num_batch): 
        randidx = np.random.randint(ntrain, size=batch_size)
        batch_xs = trainimg[randidx, :]
        batch_ys = trainlabel[randidx, :]                
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        # Compute average loss
        avg_cost += sess.run(cost
            , feed_dict={x: batch_xs, y: batch_ys})/num_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % 
               (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
        print (" Test accuracy: %.3f" % (test_acc))
print ("Optimization Finished!")


# # CLOSE SESSION

# In[6]:


sess.close()
print ("Session closed.")

