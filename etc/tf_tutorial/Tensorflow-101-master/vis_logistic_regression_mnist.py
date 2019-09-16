#!/usr/bin/env python
# coding: utf-8

# # Visualizing Logistic Regression

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist      = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels


# # Define the graph

# In[2]:


# Parameters of Logistic Regression
learning_rate   = 0.01
training_epochs = 20
batch_size      = 100
display_step    = 5

# Create Graph for Logistic Regression
x = tf.placeholder("float", [None, 784], name="INPUT_x") 
y = tf.placeholder("float", [None, 10], name="OUTPUT_y")  
W = tf.Variable(tf.zeros([784, 10]), name="WEIGHT_W")
b = tf.Variable(tf.zeros([10]), name="BIAS_b")

# Activation, Cost, and Optimizing functions
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(corr, "float"))
init = tf.initialize_all_variables()


# # Launch the graph

# In[3]:


sess = tf.Session()
sess.run(init)


# # Summary writer

# In[4]:


summary_path = '/tmp/tf_logs/logistic_regression_mnist'
summary_writer = tf.train.SummaryWriter(summary_path, graph=sess.graph)
print ("Summary writer ready")


# # Run

# In[5]:


for epoch in range(training_epochs):
    sum_cost = 0.
    num_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(num_batch): 
        randidx  = np.random.randint(trainimg.shape[0], size=batch_size)
        batch_xs = trainimg[randidx, :]
        batch_ys = trainlabel[randidx, :]                
        # Fit training using batch data
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        # Compute average loss
        sum_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = sum_cost / num_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f" 
               % (epoch, training_epochs, avg_cost, train_acc))
print ("Optimization Finished!")

# Test model
test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
print (("Test Accuracy: %.3f") % (test_acc))


# ### Run the command line
# ##### tensorboard --logdir=/tmp/tf_logs/logistic_regression_mnist
# ### Open http://localhost:6006/ into your web browser

# <img src="images/tsboard/logistic_regression_mnist.png">
