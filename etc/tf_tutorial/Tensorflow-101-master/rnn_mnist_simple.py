#!/usr/bin/env python
# coding: utf-8

# # Sequence classification with LSTM

# In[1]:


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print ("Packages imported")

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimgs, trainlabels, testimgs, testlabels  = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels 
ntrain, ntest, dim, nclasses  = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print ("MNIST loaded")


# # We will treat the MNIST image $\in \mathcal{R}^{28 \times 28}$ as $28$ sequences of a vector $\mathbf{x} \in \mathcal{R}^{28}$. 
# # Our simple RNN consists of  
# 1. One input layer which converts a $28$ dimensional input to an $128$ dimensional hidden layer, 
# 2. One intermediate recurrent neural network (LSTM) 
# 3. One output layer which converts an $128$ dimensional output of the LSTM to $10$ dimensional output indicating a class label. 

# <img src="images/etc/rnn_input3.jpg" width="700" height="400" >

# # Contruct a Recurrent Neural Network

# In[2]:


diminput  = 28
dimhidden = 128
dimoutput = nclasses
nsteps    = 28
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])), 
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}


# In[3]:


def _RNN(_X, _istate, _W, _b, _nsteps, _name):
    # 1. Permute input from [batchsize, nsteps, diminput] 
    #   => [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput] 
    _X = tf.reshape(_X, [-1, diminput])
    # 3. Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 4. Splite data to 'nsteps' chunks. An i-th chunck indicates i-th batch data 
    _Hsplit = tf.split(0, _nsteps, _H) 
    # 5. Get LSTM's final output (_LSTM_O) and state (_LSTM_S)
    #    Both _LSTM_O and _LSTM_S consist of 'batchsize' elements
    #    Only _LSTM_O will be used to predict the output. 
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, initial_state=_istate)
    # 6. Output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']    
    # Return! 
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O 
    }
print ("Network ready")


# # Out Network looks like this

# <img src="images/etc/rnn_mnist_look.jpg" width="700" height="400" >

# # Define functions

# In[4]:


learning_rate = 0.001
x      = tf.placeholder("float", [None, nsteps, diminput])
istate = tf.placeholder("float", [None, 2*dimhidden]) 
    # state & cell => 2x n_hidden
y      = tf.placeholder("float", [None, dimoutput])
myrnn  = _RNN(x, istate, weights, biases, nsteps, 'basic')
pred   = myrnn['O']
cost   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
optm   = tf.train.AdamOptimizer(learning_rate).minimize(cost) # Adam Optimizer
accr   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))
init   = tf.initialize_all_variables()
print ("Network Ready!")


# # Run!

# In[5]:


training_epochs = 5
batch_size      = 128
display_step    = 1
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph=sess.graph)
print ("Start optimization")
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
        # Fit training using batch data
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
        sess.run(optm, feed_dict=feeds)
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict=feeds)/total_batch
    # Display logs per epoch step
    if epoch % display_step == 0: 
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
        train_acc = sess.run(accr, feed_dict=feeds)
        print (" Training accuracy: %.3f" % (train_acc))
        testimgs = testimgs.reshape((ntest, nsteps, diminput))
        feeds = {x: testimgs, y: testlabels, istate: np.zeros((ntest, 2*dimhidden))}
        test_acc = sess.run(accr, feed_dict=feeds)
        print (" Test accuracy: %.3f" % (test_acc))
print ("Optimization Finished.")


# ## What we have done so far is to feed 28 sequences of vectors $ \mathbf{x} \in \mathcal{R}^{28}$. 
# 
# ## What will happen if we feed first 25 sequences of $\mathbf{x}$?

# In[6]:


# How may sequences will we use?
nsteps2     = 25

# Test with truncated inputs
testimgs = testimgs.reshape((ntest, nsteps, diminput))
testimgs_trucated = np.zeros(testimgs.shape)
testimgs_trucated[:, 28-nsteps2:] = testimgs[:, :nsteps2, :]
feeds = {x: testimgs_trucated, y: testlabels, istate: np.zeros((ntest, 2*dimhidden))}
test_acc = sess.run(accr, feed_dict=feeds)
print (" If we use %d seqs, test accuracy becomes %.3f" % (nsteps2, test_acc))


# # What's going on inside the RNN?

# ## Inputs to the RNN

# In[7]:


batch_size = 5
xtest, _ = mnist.test.next_batch(batch_size)
print ("Shape of 'xtest' is %s" % (xtest.shape,))


# ## Reshaped inputs

# In[8]:


# Reshape (this will go into the network)
xtest1 = xtest.reshape((batch_size, nsteps, diminput))
print ("Shape of 'xtest1' is %s" % (xtest1.shape,))


# ## Feeds: inputs and initial states

# In[9]:


feeds = {x: xtest1, istate: np.zeros((batch_size, 2*dimhidden))}


# ## Each indivisual input to the LSTM

# In[10]:


rnnout_X = sess.run(myrnn['X'], feed_dict=feeds)
print ("Shape of 'rnnout_X' is %s" % (rnnout_X.shape,))


# ## Each indivisual intermediate state 

# In[11]:


rnnout_H = sess.run(myrnn['H'], feed_dict=feeds)
print ("Shape of 'rnnout_H' is %s" % (rnnout_H.shape,))


# ## Actual input to the LSTM (List)

# In[12]:


rnnout_Hsplit = sess.run(myrnn['Hsplit'], feed_dict=feeds)
print ("Type of 'rnnout_Hsplit' is %s" % (type(rnnout_Hsplit)))
print ("Length of 'rnnout_Hsplit' is %s and the shape of each item is %s" 
       % (len(rnnout_Hsplit), rnnout_Hsplit[0].shape))


# ## Output from the LSTM (List)

# In[13]:


rnnout_LSTM_O = sess.run(myrnn['LSTM_O'], feed_dict=feeds)
print ("Type of 'rnnout_LSTM_O' is %s" % (type(rnnout_LSTM_O)))
print ("Length of 'rnnout_LSTM_O' is %s and the shape of each item is %s" 
       % (len(rnnout_LSTM_O), rnnout_LSTM_O[0].shape))


# ## Final prediction

# In[14]:


rnnout_O = sess.run(myrnn['O'], feed_dict=feeds)
print ("Shape of 'rnnout_O' is %s" % (rnnout_O.shape,))

