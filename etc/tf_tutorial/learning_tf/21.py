""" Generative Adversarial Networks (GAN).
Using generative adversarial networks (GAN) to generate digit images from a
noise distribution.
References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

# Training Params
num_steps = 1000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}


# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    '''
    得到的向量是[None,image_dim]  image_dim:784 表示28*28的flat后的结果
    '''
    return out_layer


# Discriminator
def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Build Networks
# Network Inputs
'''
生成假的gen_input
'''
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')

# Build Generator Network
gen_sample = generator(gen_input)
'''
gen_sample得到的是得到的向量是[None,image_dim]  image_dim:784 表示28*28的flat后的结果
'''

'''
生成真的disc_input
'''
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')



# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)
'''
disc_real和disc_fake是真,假图片提取后的一个在0,1之间的实数!
'''
# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
'''
下面我们要学习的是discriminator这个分类器:
    这个分类器好就表示:
        这个loss函数为什么用log来计算呢?因为discrimator函数里面有sigmoid函数,所以
        极大似然就是log函数.
        
        从函数单调性分析知道:disfake->0:  gen_loss->无穷
                            disreal->1:  disc_loss->0
        那么为什么需要使用2个loss函数呢?
        以为如果只用第一个显然没法评估disc_real的值
        如果只用第二个那么disc_real,disc_fake没法区别到底哪个值趋近1哪个值趋近0.
        所以也不行.所以必须要用2个loss函数来刻画.
'''
# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables

'''
下面几行初始化变量
'''
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
            biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
#        print(batch_x.shape) 看出是(128.784)类型的.说明图像dimension是1*784类型的.
        
        # Generate noise to feed to the generator
        '''
        高斯噪音
        '''
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        '''
        sess里面run 最后的optimizer即可.
        '''
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
    print('33333333333333333')
    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    print('11111111111111')
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        print('222222222')
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        print('4444444444')
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)
    print('5555555555')

    f.show()
    print('55555555551')
    plt.draw()
    print('55555555552')
    plt.show()
#    plt.waitforbuttonpress()这个代码在spyder里面跑不了,用cmd跑.这个命令主要是在ion画图模式生成动画用.