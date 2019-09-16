#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
 Deeper Multi-Layer Pecptron with XAVIER Init
 Xavier init from {Project: https://github.com/aymericdamien/TensorFlow-Examples/}
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mnist = input_data.read_data_sets('data/', one_hot=True)


# In[3]:


# Xavier Init
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


# In[4]:


# Parameters
learning_rate   = 0.001
training_epochs = 50
batch_size      = 100
display_step    = 1

# Network Parameters
n_input    = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_hidden_3 = 256 # 3rd layer num features
n_hidden_4 = 256 # 4th layer num features
n_classes  = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder("float")

# Create model
def multilayer_perceptron(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), _keep_prob)
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])), _keep_prob)
    layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])), _keep_prob) 
    layer_4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4'])), _keep_prob) 
    return (tf.matmul(layer_4, _weights['out']) + _biases['out']) # No need to use softmax??

# Store layers weight & bias
weights = {
    'h1': tf.get_variable("h1", shape=[n_input, n_hidden_1],    initializer=xavier_init(n_input,n_hidden_1)),
    'h2': tf.get_variable("h2", shape=[n_hidden_1, n_hidden_2], initializer=xavier_init(n_hidden_1,n_hidden_2)),
    'h3': tf.get_variable("h3", shape=[n_hidden_2, n_hidden_3], initializer=xavier_init(n_hidden_2,n_hidden_3)),
    'h4': tf.get_variable("h4", shape=[n_hidden_3, n_hidden_4], initializer=xavier_init(n_hidden_3,n_hidden_4)),
    'out': tf.get_variable("out", shape=[n_hidden_4, n_classes], initializer=xavier_init(n_hidden_4,n_classes))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost) # Adam Optimizer

# Accuracy 
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.initialize_all_variables()

print ("Network Ready")


# In[5]:


# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})/total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})
        print ("Training accuracy: %.3f" % (train_acc))

print ("Optimization Finished!")


# In[6]:


test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, dropout_keep_prob:1.})
print ("Training accuracy: %.3f" % (test_acc))

