#!/usr/bin/env python
# coding: utf-8

# # CNN FINETUNING WITH PRE-TRAINED VGG NET

# In[1]:


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc 
import scipy.io
from tensorflow.examples.tutorials.mnist import input_data
#get_ipython().run_line_magic('matplotlib', 'inline')
print ("Packages loaded.")


# # LOAD DATA

# In[2]:


cwd = os.getcwd()
loadpath = cwd + "/data/data4vgg.npz"
l = np.load(loadpath)

# Show Files
print (l.files)#['trainlabel', 'trainimg', 'testimg', 'testlabel']
print(l['trainimg'].shape)
print(l['trainimg'][0].shape)
print(l['trainimg'][0])
print(l['trainimg'][1])
print(l['testimg'].shape)


# # PARSE DATA

# In[3]:


trainimg   = l['trainimg']
trainlabel = l['trainlabel']
testimg    = l['testimg']
testlabel  = l['testlabel']
ntrain     = trainimg.shape[0]
nclass     = trainlabel.shape[1]
dim        = trainimg.shape[1]
ntest      = testimg.shape[0]

print ("%d train images loaded" % (ntrain))
print ("%d test images loaded"  % (ntest))
print ("%d dimensional input"   % (dim))
print ("%d classes"             % (nclass))
print ("shape of 'trainimg' is %s" % (trainimg.shape,))
print ("shape of 'testimg' is %s" % (testimg.shape,))


# # GENERATE TENSORS FOR TRAINING AND TESTING 

# In[5]:


trainimg_tensor = np.ndarray((ntrain, 112, 112, 3))
for i in range(ntrain):
    currimg = trainimg[i, :]
    currimg = np.reshape(currimg, [112, 112, 3])
    trainimg_tensor[i, :, :, :] = currimg 
print ("shape of trainimg_tensor is %s" % (trainimg_tensor.shape,))
    
testimg_tensor = np.ndarray((ntest, 112, 112, 3))
for i in range(ntest):
    currimg = testimg[i, :]
    currimg = np.reshape(currimg, [112, 112, 3])
    testimg_tensor[i, :, :, :] = currimg 
print ("shape of testimg_tensor is %s" % (testimg_tensor.shape,))

#import matplotlib.image as mpimg 
'''
从矩阵画出图像:

    
    
tmp=trainimg_tensor[2]
tmp=tmp.reshape(112,112,3)
plt.imshow(tmp)
plt.show()
'''



# # DEFINE A FUNCTION FOR USING PRETRAINED VGG NETWORK



'''
看l里面的东西
'''



print(l)









# In[6]:
'''
下面net是网络,data_path是pre_trained参数
'''

def net(data_path, input_image):
    '''
    layers是一个tuple因为tuple比list速度快.
    '''
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)#这个是读取.mat文件的方法.
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    '''
    下面for循环进行参数赋值.同事对net进行建立神经网络
    '''
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net, mean_pixel
def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
def preprocess(image, mean_pixel):
    return image - mean_pixel
def unprocess(image, mean_pixel):
    return image + mean_pixel


# # EXTRACT FEATURES FROM THE VGG NETWORK

# In[7]:


VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
with tf.Graph().as_default(), tf.Session() as sess:
    with tf.device("/cpu:0"):
        img_placeholder = tf.placeholder(tf.float32, shape=(None, 112, 112, 3))
        net_val, mean_pixel = net(VGG_PATH, img_placeholder)
        #net_val是神经网络和对应的参数.
        #training_tensor是所有的目前做训练需要的数据集.
        #testing_tensor是所有的目前做训练需要的数据集.
        '''
        为什么是最后一层.eval(feed_dict={img_placeholder: trainimg_tensor})?
        貌似这么写之后就只训练最后一层的参数????????only time will tell
        他不是训练的,这里.eval最后一层是为了读取最后一层的shape是神马样的.
        
        
        为什么fine tune一般只改最后一层.因为最后一层是dp算法第一步,所以是必须做的一步,
        所以只做这一步会收敛速度非常快.
        '''
        train_features = net_val['relu5_4'].eval(feed_dict={img_placeholder: trainimg_tensor})
        test_features = net_val['relu5_4'].eval(feed_dict={img_placeholder: testimg_tensor})
print ("TYPE OF 'train_features' IS %s" % (type(train_features)))
print ("SHAPE OF 'train_features' IS %s" % (train_features.shape,))
print ("TYPE OF 'test_features' IS %s" % (type(test_features)))
print ("SHAPE OF 'test_features' IS %s" % (test_features.shape,))
print("PREPROCESSING DONE")

'''

现在参数已经都赋值给net了,net网络也架好了.所有参数10s左右就赋值完毕了.


'''
# # VECTORIZE CNN FEATURES

# In[8]:


train_vectorized = np.ndarray((ntrain, 7*7*512))

#print(train_vectorized)


test_vectorized  = np.ndarray((ntest, 7*7*512))
for i in range(ntrain):
    curr_feat = train_features[i, :, :, :]
    curr_feat_vec = np.reshape(curr_feat, (1, -1))
    train_vectorized[i, :] = curr_feat_vec
print(train_features)
'''
这里面的train_features就是已经跑完原始的固定系数的网络,需要跑完

train_features = net_val['relu5_4'].eval(feed_dict={img_placeholder: trainimg_tensor})
test_features = net_val['relu5_4'].eval(feed_dict={img_placeholder: testimg_tensor})
就表示结果是跑完relu5_4这一层之后的结果.
'''
for i in range(ntest):
    curr_feat = test_features[i, :, :, :]
    curr_feat_vec = np.reshape(curr_feat, (1, -1))
    test_vectorized[i, :] = curr_feat_vec
    
print ("SHAPE OF 'train_vectorized' IS %s" % (train_vectorized.shape,))
#print(train_vectorized)
print ("SHAPE OF 'test_vectorized' IS %s" % (test_vectorized.shape,))

##'''
#TYPE OF 'train_features' IS <class 'numpy.ndarray'>
#SHAPE OF 'train_features' IS (69, 7, 7, 512)
#TYPE OF 'test_features' IS <class 'numpy.ndarray'>
#SHAPE OF 'test_features' IS (18, 7, 7, 512)
#
#
#SHAPE OF 'train_vectorized' IS (69, 25088)
#SHAPE OF 'test_vectorized' IS (18, 25088)
#'''
# # DEFINE NETWORKS AND FUNCTIONS (ADD 2LAYER MLP)

# In[9]:


# Parameters
learning_rate   = 0.0001
training_epochs = 100
batch_size      = 100
display_step    = 10
'''
下面是改动网络架构.从7*7*512------>1024----->n_output
这2个全连接层.


问题:这两个网络连接上了,但是为什么只训练后一个网络的参数,第一个网络的参数取定不变呢?
注意看159行.用的是tf.constant
注意看下面网络weights里面参数用的是tf.Variable.
所以最后只有最后2层的参数进行了传导.!!!!!!!!!!!!!!!!!!!!
这就是fine tune原理

tf.Variable(trainable=False),就可以把这个变量不放入bp算法中求导数.就是不进入optimizer里面.


'''
# Network
with tf.device("/cpu:0"):
    n_input  = dim
    n_output = nclass
    weights  = {
        'wd1': tf.Variable(tf.random_normal([7*7*512, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
    }
    biases   = {
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
    '''
    下面conv_basic表示的是最后建立的2层fc
    '''
    def conv_basic(_input, _w, _b, _keepratio):
        # Input
        _input_r = _input
        # Vectorize
        _dense1 = tf.reshape(_input_r, [-1, _w['wd1'].get_shape().as_list()[0]])
        # Fc1
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # Fc2
        _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        # Return everything
        out = {'input_r': _input_r, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out }
        return out

# tf Graph input
x = tf.placeholder(tf.float32, [None, 7*7*512])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 


'''
构建最重要的部分,学习我们新加入的层
'''
with tf.device("/cpu:0"):
    _pred = conv_basic(x, weights, biases, keepratio)['out']
    '''
    tf.nn.softmax_cross_entropy_with_logits
    这个函数集成了softmax和log交叉熵.所以直接给y和yhat即可.
    其中logits=logits, labels=y_  其中logits放入yhat, labels放入y即可. yhat表示模型预测出来的,y表示真实标签.
    '''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels= y))
    optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    _corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
    init = tf.initialize_all_variables()

print ("Network Ready to Go!")


# # CNN FINETUNING 

# In[10]:


sess = tf.Session()
sess.run(init)
'''
训练
'''
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0. #每一次都用0.而不用0 估计是为了节约类型转化的时间.直接定义成float
    num_batch = int(ntrain/batch_size)+1
    # Loop over all batches
    for i in range(num_batch): 
        randidx  = np.random.randint(ntrain, size=batch_size)
        batch_xs = train_vectorized[randidx, :]
        batch_ys = trainlabel[randidx, :]                
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/num_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: test_vectorized, y: testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished!")

