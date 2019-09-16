#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import glob
import os
import time
import numpy as np

class StopWatch():
    startTime = time.time()
    def __init__(self):
        self.startTime = time.time()
    def StartTime(self):
        self.startTime = time.time()
    def CheckTime(self):
        return time.time() - self.startTime

    def PrintCheckTime(self, msg):
        elapsedTime = self.CheckTime()
        print msg + ' : ' + str(elapsedTime) + ' sec, ' + str(elapsedTime/60) + ' min'

class TrainingPlot():

    listXcoord = []
    listTrainLoss = []
    listTrainAcc = []
    listTestLoss = []
    listTestAcc = []
    iterSampleCount = 0
    iter = 0
    sumTestAcc = 0
    batchSize = 0
    sampleCount = 0
    totalIter = 0
    avgTestAcc = 0
    watchTotal = StopWatch()
    watchSingle = StopWatch()

    def __init__(self):

        plt.ion()
        plt.show()
        self.watchTotal.StartTime()
        self.watchSingle.StartTime()

    def SetConfig(self, batchSize, sampleCount, totalIter):
        self.batchSize = batchSize
        self.sampleCount = sampleCount
        self.totalIter = totalIter


    def AddTest(self, trainAcc, testAcc):
        #self.sampleCount += self.batchSize
        self.iter += 1
        self.listXcoord.append(self.iter)
        self.listTrainAcc.append(trainAcc)
        self.listTestAcc.append(testAcc)

        self.sumTestAcc += testAcc
        self.avgTestAcc = self.sumTestAcc/self.iter
        print ('#Iter %d / %d :TrainAcc %f, TestAcc %f, AvgAcc %f' %
               (self.iter, self.totalIter, trainAcc,testAcc,self.avgTestAcc))

    def Add(self, iter, trainLoss, testLoss, trainAcc, testAcc):
        self.sampleCount += self.batchSize
        self.iter = iter
        self.listXcoord.append(self.iter)
        self.listTrainLoss.append(trainLoss)
        self.listTrainAcc.append(trainAcc)
        self.listTestLoss.append(testLoss)
        self.listTestAcc.append(testAcc)
        totalElapsedTime = self.watchTotal.CheckTime()
        singleEapsedTime = self.watchSingle.CheckTime()
        self.watchSingle.StartTime()
        #print self.watch.startTime,elapsedTime
        totalEstimatedTime = (float(self.totalIter) / (float(self.iter) + 0.0001)) * totalElapsedTime
        print ('#Iter %d / %d : Train %f, Test %f, TrainAcc %f, TestAcc %f (1 iter %.02f sec, total %.02f min, %.02f hour remained)' %
               (self.iter, self.totalIter,trainLoss, testLoss, trainAcc,testAcc,
                singleEapsedTime, totalElapsedTime/60, (totalEstimatedTime - totalElapsedTime)/3600))


    def Show(self):
        plt.clf()
        plt.subplot(1,2,1)
        plt.title('Loss : %.03f' % (self.listTestLoss[-1]))
        if (len(self.listTrainLoss) > 0):
            plt.plot(self.listXcoord,self.listTrainLoss, color= '#%02x%02x%02x' % (58,146,204) )
        if (len(self.listTestLoss) > 0):
            plt.plot(self.listXcoord,self.listTestLoss, color= '#%02x%02x%02x' % (220,98,45))
        plt.legend(['train', 'val'])
        plt.subplot(1,2,2)
        plt.title('Acc : %.03f' % (self.listTestAcc[-1]))
        if (len(self.listTrainAcc) > 0):
            plt.plot(self.listXcoord,self.listTrainAcc, color= '#%02x%02x%02x' % (58,146,204) )
        if (len(self.listTestAcc) > 0):
            plt.plot(self.listXcoord,self.listTestAcc, color='#%02x%02x%02x' % (34,177,76))
        plt.ylim([0,1])

        plt.draw()
        plt.pause(0.001)
        plt.show()


# In[2]:


"""
Convolutional Encoder Decoder Net

Usage :
1. Download CamVid dataset ()
2. Run createDB once (Set following condition to 1)
# Create DB (run once)
if (0):

3. Reset condition to 0 and run training

"""

import tensorflow as tf
# from TrainingPlot import *
from PIL import Image
import cPickle as pkl
import time
width = 128 # 320
height = 128 # 224
classes = 22
kernelSize = 7
featureSize = 64
resumeTraining = True

weights = {
    'ce1': tf.get_variable("ce1",shape= [kernelSize,kernelSize, 3, featureSize]),
    'ce2': tf.get_variable("ce2",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'ce3': tf.get_variable("ce3",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'ce4': tf.get_variable("ce4",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd4': tf.get_variable("cd4",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd3': tf.get_variable("cd3",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd2': tf.get_variable("cd2",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd1': tf.get_variable("cd1",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'dense_inner_prod': tf.get_variable("dense_inner_prod",shape= [1, 1, featureSize,classes])
}

def CreateDB(categoryName):
    pathLoad1 = categoryName
    pathLoad2 = categoryName + 'annot'
    # curPath = os.path.dirname(os.path.abspath(__file__))
    curPath = os.getcwd()
    print curPath
    fileList1 = glob.glob(curPath + '/' + pathLoad1 + '/*.png')

    #fileList2 = glob.glob(pathLoad2 + '/*.png')
    #plt.ion()
    trainFile = open(categoryName + '.txt','wt')
    count = 0
    occupancyList = []
    for filename in fileList1:
        img1 = Image.open(filename)
        #filename2 = filename.replace('_IPMImg', '_IPMLabel')
        filename2 = curPath + '/' + pathLoad2 + '/' + os.path.basename(filename)
        img2 = Image.open(filename2)

        print >> trainFile, filename
        print >> trainFile, filename2

        #cropimg.save(pathSave + '/' + filename)
        #plt.title(pathCity)
        #plt.imshow(cropimg)
        #plt.show()
        count += 1

    print ('%d data list created' % count)
    trainFile.close()
    return occupancyList

# input : [m x h x w x c]
def Unpooling(inputOrg, size, mask=None):
    # m, c, h, w order
    # print 'start unpooling'
    # size = tf.shape(inputOrg)
    m = size[0]
    h = size[1]
    w = size[2]
    c = size[3]
    input = tf.transpose(inputOrg, [0, 3, 1, 2])
    # print input.get_shape()
    x = tf.reshape(input, [-1, 1])
    k = np.float32(np.array([1.0, 1.0]).reshape([1,-1]))
    # k = tf.Variable([1.0, 1.0],name="weights")
    # k = tf.reshape(k,[1,-1])
    # k = np.array(k).reshape([1, -1])
    output = tf.matmul(x, k)

    output = tf.reshape(output,[-1, c, h, w * 2])
    # m, c, w, h
    xx = tf.transpose(output, [0, 1, 3, 2])
    xx = tf.reshape(xx,[-1, 1])
    # print xx.shape

    output = tf.matmul(xx, k)
    # m, c, w, h
    output = tf.reshape(output, [-1, c, w * 2, h * 2])
    output = tf.transpose(output, [0, 3, 2, 1])
    # print mask
    outshape = tf.pack([m, h * 2, w * 2, c])

    if mask != None:
        dense_mask = tf.sparse_to_dense(mask, outshape, output, 0)
        # print dense_mask
        # print 'output',output
        # print 'mask',mask
        # print dense_mask
            # output = tf.mul(output, mask)

        return output, dense_mask
    else:
        return output

# max pool + stride 2 transpose conv
def Model(X, W):

    # Encoder
    encoder1 = tf.nn.conv2d(X, W['ce1'], strides=[1, 1, 1, 1], padding='SAME')
    encoder1 = tf.nn.batch_normalization(encoder1,0.001,1.0,0,1,0.0001)
    encoder1 = tf.nn.relu(encoder1)
    encoder1 = tf.nn.max_pool(encoder1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # encoder1 = tf.nn.dropout(encoder1, 0.5)

    encoder2 = tf.nn.conv2d(encoder1, W['ce2'], strides=[1, 1, 1, 1], padding='SAME')
    encoder2 = tf.nn.batch_normalization(encoder2, 0.001, 1.0, 0, 1, 0.0001)
    encoder2 = tf.nn.relu(encoder2)
    encoder2 = tf.nn.max_pool(encoder2, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    # encoder2 = tf.nn.dropout(encoder2, 0.5)

    encoder3 = tf.nn.conv2d(encoder2, W['ce3'], strides=[1, 1, 1, 1], padding='SAME')
    encoder3 = tf.nn.batch_normalization(encoder3, 0.001, 1.0, 0, 1, 0.0001)
    encoder3 = tf.nn.relu(encoder3)
    encoder3 = tf.nn.max_pool(encoder3, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    # encoder3 = tf.nn.dropout(encoder3, 0.5)

    encoder4 = tf.nn.conv2d(encoder3, W['ce4'], strides=[1, 1, 1, 1], padding='SAME')
    encoder4 = tf.nn.batch_normalization(encoder4, 0.001, 1.0, 0, 1, 0.0001)
    encoder4 = tf.nn.relu(encoder4)
    encoder4 = tf.nn.max_pool(encoder4, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    # encoder4 = tf.nn.dropout(encoder4, 0.5)

    # Decoder
    decoder4 = Unpooling(encoder4, [tf.shape(X)[0], height / 16, width / 16, featureSize])
    decoder4 = tf.nn.conv2d(decoder4, W['cd4'], strides=[1, 1, 1, 1], padding='SAME')
    decoder4 = tf.nn.batch_normalization(decoder4, 0.001, 1.0, 0, 1, 0.0001)
    decoder4 = tf.nn.relu(decoder4)
    # decoder4 = tf.nn.dropout(decoder4, 0.5)

    decoder3 = Unpooling(encoder3, [tf.shape(X)[0], height/8, width/8, featureSize])
    decoder3 = tf.nn.conv2d(decoder3, W['cd3'], strides=[1, 1, 1, 1], padding='SAME')
    decoder3 = tf.nn.batch_normalization(decoder3, 0.001, 1.0, 0, 1, 0.0001)
    decoder3 = tf.nn.relu(decoder3)
    # decoder3 = tf.nn.dropout(decoder3, 0.5)

    decoder2 = Unpooling(decoder3, [tf.shape(X)[0], height/4, width/4, featureSize])
    decoder2 = tf.nn.conv2d(decoder2, W['cd2'], strides=[1, 1, 1, 1], padding='SAME')
    decoder2 = tf.nn.batch_normalization(decoder2, 0.001, 1.0, 0, 1, 0.0001)
    decoder2 = tf.nn.relu(decoder2)
    # decoder2 = tf.nn.dropout(decoder2, 0.5)

    decoder1 = Unpooling(decoder2, [tf.shape(X)[0], height / 2, width / 2, featureSize])
    decoder1 = tf.nn.conv2d(decoder1, W['cd1'], strides=[1, 1, 1, 1], padding='SAME')
    decoder1 = tf.nn.batch_normalization(decoder1, 0.001, 1.0, 0, 1.0, 0.0001)
    decoder1 = tf.nn.relu(decoder1)
    # decoder1 = tf.nn.dropout(decoder1, 0.5)

    output = tf.nn.conv2d(decoder1, W['dense_inner_prod'], strides=[1, 1, 1, 1], padding='SAME')

    # return output, mask1, mask2, mask3
    return output


def DenseToOneHot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def LoadTrainingData(filename,sampleCount=None):
    class DataSets(object):
        pass

    datalistFile = open(filename, "rt")
    fileList = datalistFile.readlines()
    # print len(fileList)
    data = None
    label = None
    if sampleCount == None:
        sampleCount = len(fileList)

    for i in range(0,sampleCount,2):
    # for i in range(0,50,2):
        file = fileList[i].replace('\n','')
        # print ('%d / %d' % (i, len(fileList)))
        img = Image.open(file)
        img = img.resize((width, height))
        rgb = np.array(img).reshape(1, height, width,3)

        # pixels = np.concatenate((np.array(rgb[0]).flatten(),np.array(rgb[1]).flatten(),np.array(rgb[2]).flatten()),axis=0)
        # pixels = pixels.reshape(pixels.shape[0], 1)
        if i == 0:
            data = rgb
        else:
            data = np.concatenate((data, rgb),axis=0)

        # file = fileList[i * 2 + 1].replace('\n', '')
        # label = Image.open(file)

        file = fileList[i+1].replace('\n', '')
        # print i,file
        img = Image.open(file)
        img = img.resize((width, height), Image.NEAREST)
        labelImage = np.array(img).reshape(1, height, width,1)

        if i == 0:
            label = labelImage
        else:
            # print data.shape
            label = np.concatenate((label, labelImage), axis=0)
    labelOneHot = np.zeros((label.shape[0],label.shape[1], label.shape[2], classes))
    for row in range(height):
        for col in range(width):
            single = label[:, row, col, 0]
            # print single.shape
            # exit(0)
            # print index
            oneHot = DenseToOneHot(single, classes)
            labelOneHot[:, row, col, :] = oneHot
    # for i in range(22):
    #     plt.subplot(1,2,1)
    #     plt.imshow(data[0,:,:,:].reshape(height,width,3))
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(labelOneHot[0,:,:,i].reshape(height,width))
    #     plt.show()
    return [data.astype(np.float32)/255, label, labelOneHot.astype(np.float32)]

def ShowDebuggingPlot():
    global batchData, batchLabel
    index = np.random.randint(trainData.shape[0])
    batchData = trainData[index:index+1]
    batchLabel = trainLabelOneHot[index:index+1]
    predMaxOut = sess.run(predMax, feed_dict={x: batchData, y: batchLabel})
    yMaxOut = sess.run(yMax, feed_dict={x: batchData, y: batchLabel})
    # for i in range(22):
    # show predicted image
    plt.figure(2)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.title('Input')
    img = trainData[index, :, :, :].reshape(height, width, 3)
    # plt.imshow(img * 255)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.title('Ground truth')
    img = yMaxOut[0, :, :].reshape(height, width)
    plt.imshow(img)
    plt.subplot(2, 2, 3)
    plt.title('[Training] Prediction')
    plt.imshow(predMaxOut[0, :, :].reshape(height, width))
    plt.subplot(2, 2, 4)
    plt.title('Error')
    plt.imshow(img - predMaxOut[0, :, :].reshape(height, width))
    plt.show()
    
def ShowValPlot():
    global batchData, batchLabel
    index = np.random.randint(valData.shape[0])
    batchData = valData[index:index+1]
    batchLabel = valLabelOneHot[index:index+1]
    predMaxOut = sess.run(predMax, feed_dict={x: batchData, y: batchLabel})
    yMaxOut = sess.run(yMax, feed_dict={x: batchData, y: batchLabel})
    # for i in range(22):
    # show predicted image
    plt.figure(3)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.title('Input')
    img = valData[index, :, :, :].reshape(height, width, 3)
    # plt.imshow(img * 255)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.title('Ground truth')
    img = yMaxOut[0, :, :].reshape(height, width)
    plt.imshow(img)
    plt.subplot(2, 2, 3)
    plt.title('[Validation] Prediction')
    plt.imshow(predMaxOut[0, :, :].reshape(height, width))
    plt.subplot(2, 2, 4)
    plt.title('Error')
    plt.imshow(img - predMaxOut[0, :, :].reshape(height, width))
    plt.show()


# In[3]:


# main body

# Create DB (run once)
camvidpath = 'data/seg/SegNet-Tutorial-master/CamVid/'
if (1):
    CreateDB(camvidpath + 'train')
    CreateDB(camvidpath + 'val')
    
startTime = time.time()
print('Start data loading')
sampleCount = 20
trainData, trainLabel, trainLabelOneHot = LoadTrainingData(camvidpath + 'train.txt',sampleCount)
valData, valLabel, valLabelOneHot = LoadTrainingData(camvidpath + 'val.txt',sampleCount/2)

print('Finished in %d sec' % (time.time() - startTime))

# Define functions
x = tf.placeholder(tf.float32, [None, height,width,3])
y = tf.placeholder(tf.float32, [None, height,width,classes])

# input = tf.reshape(x, shape=[-1, 224, 320, 3])
# output : m x height x width x classes

# pred, mask1, mask2, mask3 = Model(x, weights)
pred = Model(x, weights)

linearizePred = tf.reshape(pred,shape=[-1,classes])
linearizeY = tf.reshape(y,shape=[-1,classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(linearizePred, linearizeY))


# accuracy
# print yNumber, tf.argmax(pred, 3)
predMax = tf.argmax(pred, 3)
yMax = tf.argmax(y, 3)
correct_pred = tf.equal(tf.argmax(y,3), tf.argmax(pred, 3)) # Count correct predictions


acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average

learning_rate = 0.0001

optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Fit all training data
batch_size = 2
n_epochs = 10000

print("Strart training..")

trainingPlot = TrainingPlot()
trainingPlot.SetConfig(batch_size, 500, n_epochs)


# In[7]:


print (trainData.shape)
print (trainLabel.shape)
print (trainLabelOneHot.shape)


# In[ ]:



# you need to initialize all variables
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    # if (resumeTraining):
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("nets/deconv")
    if resumeTraining == False:
        print "Start from scratch"
    elif  checkpoint:
        print "Restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "Couldn't find checkpoint to restore from. Starting over."


    for epoch_i in range(n_epochs):
        trainLoss = []
        trainAcc = []
        for start, end in zip(range(0, len(trainData), batch_size), range(batch_size, len(trainData), batch_size)):

            batchData = trainData[start:end]
            batchLabel = trainLabelOneHot[start:end]

            sess.run(optm, feed_dict={x: batchData, y: batchLabel})
            trainLoss.append(sess.run(cost, feed_dict={x: batchData, y: batchLabel}))
            trainAcc.append(sess.run(acc_op, feed_dict={x: batchData, y: batchLabel}))

        trainLoss = np.mean(trainLoss)
        trainAcc = np.mean(trainAcc)

        # run validation
        valLoss = sess.run(cost, feed_dict={x: valData, y: valLabelOneHot})
        valAcc = sess.run(acc_op, feed_dict={x: valData, y: valLabelOneHot})

        # trainingPlot.Add(epoch_i, trainLoss, valLoss, trainAcc, valAcc)
        # plt.figure(1)
        # trainingPlot.Show()

        # save snapshot
        if resumeTraining and epoch_i % 1000 == 0:
            print "training on image #%d" % epoch_i
            saver.save(sess, 'nets/deconv/progress', global_step=epoch_i)

        # show debugging image
        if epoch_i % 500 == 0:
            trainingPlot.Add(epoch_i, trainLoss, valLoss, trainAcc, valAcc)
            plt.figure(1)
            trainingPlot.Show()
            # Debug plot
            ShowDebuggingPlot()
            # Plot on validation set
            ShowValPlot()

        # print(epoch_i, "/", n_epochs, loss)

print("Training done. ")


# In[ ]:




