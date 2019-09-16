#!/usr/bin/env python
# coding: utf-8

# ## DEEPER MULTILAYER PERCEPTRON WITH DROPOUT

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().run_line_magic('matplotlib', 'inline')


# # LOAD MNIST

# In[2]:


mnist = input_data.read_data_sets('data/', one_hot=True)


# # DEFINE MODEL

# In[3]:


# NETWORK TOPOLOGIES
n_input    = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_hidden_3 = 256 # 3rd layer num features
n_classes  = 10 # MNIST total classes (0-9 digits)
# INPUT AND OUTPUT
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder("float")
# VARIABLES
stddev = 0.05
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print ("NETWORK READY")


# In[4]:


def multilayer_perceptron(_X, _weights, _biases, _keep_prob):
    x_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1 = x_1
    x_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    layer_2 = x_2
    x_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    layer_3 = tf.nn.dropout(x_3, _keep_prob) 
    return (tf.matmul(layer_3, _weights['out']) + _biases['out'])


# # DEFINE FUNCTIONS

# In[5]:


# PREDICTION
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

# LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.initialize_all_variables()
print ("FUNCTIONS READY")


# # RUN

# In[6]:


# PARAMETERS
training_epochs = 20
batch_size      = 100
display_step    = 4
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # ITERATION
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys, dropout_keep_prob: 0.6}
        sess.run(optm, feed_dict=feeds)
        feeds = {x: batch_xs, y: batch_ys, dropout_keep_prob: 1.0}
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # DISPLAY
    if (epoch+1) % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, dropout_keep_prob: 1.0}
        train_acc = sess.run(accr, feed_dict=feeds)
        print ("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels, dropout_keep_prob: 1.0}
        test_acc = sess.run(accr, feed_dict=feeds)
        print ("TEST ACCURACY: %.3f" % (test_acc))
print ("OPTIMIZATION FINISHED")

