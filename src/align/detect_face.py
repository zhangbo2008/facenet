""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems

import numpy as np
import tensorflow as tf
#from math import floor
import cv2
import os

def layer(op):
    """Decorator for composable network layers."""
    '''
    装饰器.op表示原始的层,对于每一层网络结构都进行了封装.从而函数写的更简洁,高级.可读性强.
    就是看一看做一个函数共性提取到这个装饰器里面了.
    
    这个装饰器是类的函数的装饰器,所以也需要写一个self,即使这个装饰器写在了类外面.
    '''
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        '''
        如果kwargs里面有'name'就返回kwargs['name']
        否则就:返回self.get_unique_name(op.__name__)
        
        args:是数组参数:<class 'tuple'>: (3, 3, 10, 1, 1)
        kwargs:是字典参数 :dict: {'padding': 'VALID', 'relu': False, 'name': 'conv1'}
        '''
        
#        print(args)
#        print(kwargs)
#        print(type(kwargs))
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
#        print(kwargs)
#        print(type(kwargs))
        '''
        当这个layer已经有'name'那么name就是字典里面name的值.否则就是加下划线的name
        
        print(name)
        print(name)
        print('****************************')
        print(op.__name__)  #看出来函数. __name__ 表示的是函数的名字.
        print('-------------------------------------')
        '''
#        print(name)
        # Figure out the layer inputs.   self.terminals表示当前层的入参
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:#如果terminal:也就是上一层的输出是一个张量就给layer_input参数.
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.#下一行进入op运算,也就是整个网络的核心代码!!!!!!!!!!!!!!!!!!!!!!!!!
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output #结果保存到layers[name]里面,这样只要这个步骤跑过了,就用layers[name]就能访问这个中间变量了.
        # This output is now the input for the next layer. 
        self.feed(layer_output) #继续扔给feed,给一下一次使用.这样下一层就不用feed了.
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):#这个是所有网络的父类

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
#         反反复复
        self.inputs = inputs#这里面inputs是一个字典
        # The current list of terminal nodes 
        self.terminals = []
        # Mapping from layer names to layers
        '''
        
        dict函数的构造方法.
        temp = dict(name='xiaoming', age=18)
        
        
        
        
        '''
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()#利用子类重写的这个setup方法把self.layers传进去

    def setup(self):#这是一个类似java接口的概念,子类直接重写即可.
        """Construct the network. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """#np.load读完是一个字典.
        data_dict = np.load(data_path, encoding='latin1').item() #pylint: disable=no-member
        #数据是一个字典,其中prelu里面的参数是alpha.也就是这一层也引入了一个变量进行学习.用np来恢复参数确实麻烦.需要挨个赋值
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):#这个是赋值的核心代码:reuse=True因为之前建立网络时候已经初始化这些变量了.
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise
                            
    def feed(self, *args):#返回的是一个付完参数的神经网络.
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        
#        print('args:',args)
        assert len(args) != 0
        '''
        每一次运行feed都更改这个网络对象的属性terminals=[]
        '''
        self.terminals = []
        for fed_layer in args:
            '''
            isinstance(fed_layer, string_types)
            判断fed_layer是不是一个字符串.
            '''

#            print(type(fed_layer))
            if isinstance(fed_layer, string_types):
                try:
#                    print('fed_layer:',fed_layer)
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)#terminals表示当前的输出层
        return self 

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''
        suffixed:后缀的
        '''
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        
        ident:识别
        
        字典里面的key如果这个字符串的开始是prefix那么就返回true  t.startswith(prefix)
        
        最终这个get_unique_name函数返回的是layers里面拥有这个prefix变量开头的key 的数量和+1记做indent.
        拼成的prefix_indent
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        """Verifies that the padding is one of the supported ones."""
        assert padding in ('SAME', 'VALID')

    @layer   #.conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
    def conv(self,
             inp,
             k_h,        #1
             k_w,        #1
             c_o,        #4
             s_h,        #1    表示每一次横向移动卷积核几个单位
             s_w,        #1    表示每一次纵向移动卷积核几个单位
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)#确认padding传参的正确性.
        # Get the number of channels in the input,inp就是入参是一个placeholder对象
        c_i = int(inp.get_shape()[-1])#读取channel参数.看来必须3色图.inp:(NOne,None,None,3)
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):#有名字就用这个with语句写
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))#make_var:通过名字获取张量
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output   #上面的output是改进的relu,可以让数据取到负数时候仍然有一个小的梯度可以传播,alpha要取一个很小的数.

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc


    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax
    
class PNet(Network):
    def setup(self):#self.feed('data')返回一个网络对象,并且已经把data这个placeholder张量放进去了
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='PReLU1')#.conv这个运行conv函数,这种带有装饰器的函数,就是等价于把函数代入装饰器运行,
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             .softmax(3,name='prob1'))
#下面这一行是feed('PReLu3'),表示从PRelu3层开始接下面的conv4-2层.第一个prob1已经可以输出特征了
        (self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
        
class RNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')
             .softmax(1,name='prob1'))

        (self.feed('prelu4') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv5-2'))

class ONet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))

def create_mtcnn(sess, model_path):
    if not model_path:
        model_path,_ = os.path.split(os.path.realpath(__file__))
    '''
    model_path:str: C:\\Users\\zhangbo284\\Desktop\\python_all_project\\srcAll\\face\\facenet-master\\src\\align
    _:str: detect_face.py
    
    

    '''

    with tf.variable_scope('pnet'):#下行如果不写name就placeholder函数就默认用op_type_name赋值
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = PNet({'data':data})#搭建PNet网络,PNet继承NetWork所以要先跑NetWork的构造函数.    因为子类没写构造函数所以需要拿父类的构造函数来当构造函数.参数也是直接传给了父类的__init__
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)#最后还有关闭这个with_scope的close代码
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet = RNet({'data':data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet = ONet({'data':data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    #总结这3种网络P,R,O之间都不一样.对于kernal的参数,层数.是3个浅网络,下行进行feed后读取预测值,run里面第一个tuple里面装的是需要输出的2个值的name,feed_dict放入input数据也就是data这个变量的name.所以name是一个非常重要的量
    pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
    rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})#img指的不是原始图片,只是一个形参.
    onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
    return pnet_fun, rnet_fun, onet_fun

def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold  表示的是3个网络的阈值.
    factor: the factor used to create a scaling pyramid(为什么是一个金字塔状的?) of face sizes to detect in the image.
    """#这个就是检测人脸框的核心代码.
    factor_count=0
    total_boxes=np.empty((0,9))
    points=np.empty(0)
    h=img.shape[0]
    w=img.shape[1]
    minl=np.amin([h, w])
    m=12.0/minsize
    minl=minl*m
    # create scale pyramid
    scales=[]#并且factor这个变量也可以修改,越小,运行速度越快.
    while minl>=12:#在循环里面每一次都放缩一次scales,把框minl给变小.   #如果minisize变大,那么m变小,minl变小所以scales变少,所以变快.
        scales += [m*np.power(factor, factor_count)]#所以minisize表示的是识别框的最小长宽.
        minl = minl*factor
        factor_count += 1
    #上面的循环可以看出来,scales的比例表示的是整个图片的搜索框大小相除的比例.一直除到minl<12停止.所以最小搜索框就是12像素.
    # first stage,对于每一个比例参数进行遍历找框
    for scale in scales:
        hs=int(np.ceil(h*scale))
        ws=int(np.ceil(w*scale))#向上取整.是必须向上取整,因为如果0.1就可以避免取到0这个bug值.
        im_data = imresample(img, (hs, ws))#利用cv2进行图像放缩
        im_data = (im_data-127.5)*0.0078125#做一个线性变换,也就是基本是归一化操作.放缩到-1,1区间内.
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0,2,1,3))#这都是什么操作???先转职,再跑网络再转职回去!??http://www.cnblogs.com/huipengbo/p/9774747.html这里面有图片处理后的效果
        out = pnet(img_y)#但是问题是为什么需要这么3个操作,迷的不行.注意pnet,onet,rnet的结果都是一个tuple,里面一个是中间结果一个是最终结果.
        out0 = np.transpose(out[0], (0,2,1,3))#为什么转职??答案:这个地方经过看代码发现imresample函数底层用opencv来实现它的代码是宽和高拧过来了!所以这里面需要transpose再变回去才行.
        out1 = np.transpose(out[1], (0,2,1,3))
        
        boxes, _ = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])
        ##boxes    shape: (21, 9)[
        # inter-scale nms？到底什么意思
        pick = nms(boxes.copy(), 0.5, 'Union')#pick是那些优良的框对应的index,也就是下面要处理的框都从pick中来
        if boxes.size>0 and pick.size>0:#nms是一个成熟的算法:https://blog.csdn.net/shuzfan/article/details/52711706
            boxes = boxes[pick,:]
            total_boxes = np.append(total_boxes, boxes, axis=0)
            #下面的total_boxes变量就表示所有的比例大小的框全体.
    numbox = total_boxes.shape[0]
    if numbox>0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick,:]#再按照0.7滤过一遍,这次跟上次0.5不同,因为0.5是每一个scale内部赛选,0.7是整体赛选.这个情况复杂所以设置0.7条件宽松一点容错率高.
        regw = total_boxes[:,2]-total_boxes[:,0]
        regh = total_boxes[:,3]-total_boxes[:,1]#regw,regh宽和高
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh  #total_boxes[:,5] 从5到8代表什么含义?为什么用乘法?
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)#就是向下取整
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox>0:
        # second stage
        tempimg = np.zeros((24,24,3,numbox))#这个24是怎么定的?是根据352行的data输入shape定下来的
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))#当前边框的像素点矩阵
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (24, 24))#记住公式:A and B or C and D  #等价于(A and B) or (C and D)                                                  
            else:#记住imresample有一个行列扭的运算
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = rnet(tempimg1)#第二个stage就是第一个pnet处理完,选完框之后的所有预选框都给rnet做data参数传进网络.
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]
        ipass = np.where(score>threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]        #mv是特征
        if total_boxes.shape[0]>0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))#也跟第一次first stage相似
            total_boxes = rerec(total_boxes.copy())#跟第一次类似

    numbox = total_boxes.shape[0]
    if numbox>0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48,48,3,numbox))#这个是根据上层网络的输入定下的.跟上面24是类似的.
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])  #最后输出的点.
        out2 = np.transpose(out[2])
        score = out2[1,:]
        points = out1  #这个是第三层独有的变量points,但是代表什么含义呢?
        ipass = np.where(score>threshold[2])
        points = points[:,ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]

        w = total_boxes[:,2]-total_boxes[:,0]+1  #表示每一个框的宽
        h = total_boxes[:,3]-total_boxes[:,1]+1
        points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1#np.tile(a,(5,1))表示把a张量横复制5次,纵复制1次.
        points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
        if total_boxes.shape[0]>0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick,:]
            points = points[:,pick] #points什么意思
                
    return total_boxes, points


def bulk_detect_face(images, detection_window_size_ratio, pnet, rnet, onet, threshold, factor):
    """Detects faces in a list of images
    images: list containing input images
    detection_window_size_ratio: ratio of minimum face size to smallest image dimension
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold [0-1]
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    all_scales = [None] * len(images)
    images_with_boxes = [None] * len(images)

    for i in range(len(images)):
        images_with_boxes[i] = {'total_boxes': np.empty((0, 9))}

    # create scale pyramid
    for index, img in enumerate(images):
        all_scales[index] = []
        h = img.shape[0]
        w = img.shape[1]
        minsize = int(detection_window_size_ratio * np.minimum(w, h))
        factor_count = 0
        minl = np.amin([h, w])
        if minsize <= 12:
            minsize = 12

        m = 12.0 / minsize
        minl = minl * m
        while minl >= 12:
            all_scales[index].append(m * np.power(factor, factor_count))
            minl = minl * factor
            factor_count += 1

    # # # # # # # # # # # # #
    # first stage - fast proposal network (pnet) to obtain face candidates
    # # # # # # # # # # # # #

    images_obj_per_resolution = {}                  

#     TODO: use some type of rounding to number module 8 to increase probability that pyramid images will have the same resolution across input images

    for index, scales in enumerate(all_scales):
        h = images[index].shape[0]
        w = images[index].shape[1]

        for scale in scales:
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))

            if (ws, hs) not in images_obj_per_resolution:
                images_obj_per_resolution[(ws, hs)] = []

            im_data = imresample(images[index], (hs, ws))
            im_data = (im_data - 127.5) * 0.0078125
            img_y = np.transpose(im_data, (1, 0, 2))  # caffe uses different dimensions ordering
            images_obj_per_resolution[(ws, hs)].append({'scale': scale, 'image': img_y, 'index': index})

    for resolution in images_obj_per_resolution:
        images_per_resolution = [i['image'] for i in images_obj_per_resolution[resolution]]
        outs = pnet(images_per_resolution)

        for index in range(len(outs[0])):
            scale = images_obj_per_resolution[resolution][index]['scale']
            image_index = images_obj_per_resolution[resolution][index]['index']
            out0 = np.transpose(outs[0][index], (1, 0, 2))
            out1 = np.transpose(outs[1][index], (1, 0, 2))

            boxes, _ = generateBoundingBox(out1[:, :, 1].copy(), out0[:, :, :].copy(), scale, threshold[0])

            # inter-scale nms
            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                images_with_boxes[image_index]['total_boxes'] = np.append(images_with_boxes[image_index]['total_boxes'],
                                                                          boxes,
                                                                          axis=0)

    for index, image_obj in enumerate(images_with_boxes):
        numbox = image_obj['total_boxes'].shape[0]
        if numbox > 0:
            h = images[index].shape[0]
            w = images[index].shape[1]
            pick = nms(image_obj['total_boxes'].copy(), 0.7, 'Union')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            regw = image_obj['total_boxes'][:, 2] - image_obj['total_boxes'][:, 0]
            regh = image_obj['total_boxes'][:, 3] - image_obj['total_boxes'][:, 1]
            qq1 = image_obj['total_boxes'][:, 0] + image_obj['total_boxes'][:, 5] * regw
            qq2 = image_obj['total_boxes'][:, 1] + image_obj['total_boxes'][:, 6] * regh
            qq3 = image_obj['total_boxes'][:, 2] + image_obj['total_boxes'][:, 7] * regw
            qq4 = image_obj['total_boxes'][:, 3] + image_obj['total_boxes'][:, 8] * regh
            image_obj['total_boxes'] = np.transpose(np.vstack([qq1, qq2, qq3, qq4, image_obj['total_boxes'][:, 4]]))
            image_obj['total_boxes'] = rerec(image_obj['total_boxes'].copy())
            image_obj['total_boxes'][:, 0:4] = np.fix(image_obj['total_boxes'][:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(image_obj['total_boxes'].copy(), w, h)

            numbox = image_obj['total_boxes'].shape[0]
            tempimg = np.zeros((24, 24, 3, numbox))

            if numbox > 0:
                for k in range(0, numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                    tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = images[index][y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                        tempimg[:, :, :, k] = imresample(tmp, (24, 24))
                    else:
                        return np.empty()

                tempimg = (tempimg - 127.5) * 0.0078125
                image_obj['rnet_input'] = np.transpose(tempimg, (3, 1, 0, 2))

    # # # # # # # # # # # # #
    # second stage - refinement of face candidates with rnet
    # # # # # # # # # # # # #

    bulk_rnet_input = np.empty((0, 24, 24, 3))
    for index, image_obj in enumerate(images_with_boxes):
        if 'rnet_input' in image_obj:
            bulk_rnet_input = np.append(bulk_rnet_input, image_obj['rnet_input'], axis=0)

    out = rnet(bulk_rnet_input)
    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    score = out1[1, :]

    i = 0
    for index, image_obj in enumerate(images_with_boxes):
        if 'rnet_input' not in image_obj:
            continue

        rnet_input_count = image_obj['rnet_input'].shape[0]
        score_per_image = score[i:i + rnet_input_count]
        out0_per_image = out0[:, i:i + rnet_input_count]

        ipass = np.where(score_per_image > threshold[1])
        image_obj['total_boxes'] = np.hstack([image_obj['total_boxes'][ipass[0], 0:4].copy(),
                                              np.expand_dims(score_per_image[ipass].copy(), 1)])

        mv = out0_per_image[:, ipass[0]]

        if image_obj['total_boxes'].shape[0] > 0:
            h = images[index].shape[0]
            w = images[index].shape[1]
            pick = nms(image_obj['total_boxes'], 0.7, 'Union')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            image_obj['total_boxes'] = bbreg(image_obj['total_boxes'].copy(), np.transpose(mv[:, pick]))
            image_obj['total_boxes'] = rerec(image_obj['total_boxes'].copy())

            numbox = image_obj['total_boxes'].shape[0]

            if numbox > 0:
                tempimg = np.zeros((48, 48, 3, numbox))
                image_obj['total_boxes'] = np.fix(image_obj['total_boxes']).astype(np.int32)
                dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(image_obj['total_boxes'].copy(), w, h)

                for k in range(0, numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                    tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = images[index][y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                        tempimg[:, :, :, k] = imresample(tmp, (48, 48))
                    else:
                        return np.empty()
                tempimg = (tempimg - 127.5) * 0.0078125
                image_obj['onet_input'] = np.transpose(tempimg, (3, 1, 0, 2))

        i += rnet_input_count

    # # # # # # # # # # # # #
    # third stage - further refinement and facial landmarks positions with onet
    # # # # # # # # # # # # #

    bulk_onet_input = np.empty((0, 48, 48, 3))
    for index, image_obj in enumerate(images_with_boxes):
        if 'onet_input' in image_obj:
            bulk_onet_input = np.append(bulk_onet_input, image_obj['onet_input'], axis=0)

    out = onet(bulk_onet_input)

    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    out2 = np.transpose(out[2])
    score = out2[1, :]
    points = out1

    i = 0
    ret = []
    for index, image_obj in enumerate(images_with_boxes):
        if 'onet_input' not in image_obj:
            ret.append(None)
            continue

        onet_input_count = image_obj['onet_input'].shape[0]

        out0_per_image = out0[:, i:i + onet_input_count]
        score_per_image = score[i:i + onet_input_count]
        points_per_image = points[:, i:i + onet_input_count]

        ipass = np.where(score_per_image > threshold[2])
        points_per_image = points_per_image[:, ipass[0]]

        image_obj['total_boxes'] = np.hstack([image_obj['total_boxes'][ipass[0], 0:4].copy(),
                                              np.expand_dims(score_per_image[ipass].copy(), 1)])
        mv = out0_per_image[:, ipass[0]]

        w = image_obj['total_boxes'][:, 2] - image_obj['total_boxes'][:, 0] + 1
        h = image_obj['total_boxes'][:, 3] - image_obj['total_boxes'][:, 1] + 1
        points_per_image[0:5, :] = np.tile(w, (5, 1)) * points_per_image[0:5, :] + np.tile(
            image_obj['total_boxes'][:, 0], (5, 1)) - 1
        points_per_image[5:10, :] = np.tile(h, (5, 1)) * points_per_image[5:10, :] + np.tile(
            image_obj['total_boxes'][:, 1], (5, 1)) - 1

        if image_obj['total_boxes'].shape[0] > 0:
            image_obj['total_boxes'] = bbreg(image_obj['total_boxes'].copy(), np.transpose(mv))
            pick = nms(image_obj['total_boxes'].copy(), 0.7, 'Min')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            points_per_image = points_per_image[:, pick]

            ret.append((image_obj['total_boxes'], points_per_image))
        else:
            ret.append(None)

        i += onet_input_count

    return ret


# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox,reg):
    """Calibrate bounding boxes"""#reg是修正参数?表示每一个像素需要修正多少
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox
 
def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""#猜测这个imap就是热力图,他上面点的值越大越表示像人脸.
    stride=2          #imap reg是2个特征张量  imap:2维  reg:3维
    cellsize=12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:,:,0])
    dy1 = np.transpose(reg[:,:,1])
    dx2 = np.transpose(reg[:,:,2])
    dy2 = np.transpose(reg[:,:,3])#这4个正好是边框的特征.也就是说reg保存了边框的特征
    y, x = np.where(imap >= t)       #为什么要找值大于t的点?热力图          
    if y.shape[0]==1:            #imap里面存的是分数
        dx1 = np.flipud(dx1)  
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y,x)]
    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
    if reg.size==0:
        reg = np.empty((0,3))
    bb = np.transpose(np.vstack([y,x]))      #bb是所有的有效index,表示框的左上角的坐标
    q1 = np.fix((stride*bb+1)/scale)  #这个scale在这里面使用.为什么这么使用呢???因为bb里面的坐标已经是被scale之后的小图片了
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)#表示边框缩放后大小是cellsize,为什么乘以stride?????????????????reg表示什么??
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
    return boundingbox, reg  #boundingbox表示
#boundingbox   dict: {'T': array([[ 4.50000000e+01,  4.80000000e+01,  1.05000000e+02,\n         1.05000000e+02,  1.08000000e+02,  1.15000000e+02,\n         1.18000000e+02,  1.18000000e+02,  1.41000000e+02,\n         1.41000000e+02,  1.68000000e+02,  1.68000000e+02,\n         1.71000000e+02,  1.71000000e+02,  1.71000000e+02,\n         2.21000000e+02,  2.25000000e+02,  2.25000000e+02,\n         2.25000000e+02,  2.28000000e+02,  2.31000000e+02],\n       [ 1.91000000e+02,  1.91000000e+02,  2.18000000e+02,\n         2.21000000e+02,  2.21000000e+02,  1.21000000e+02,\n         1.21000000e+02,  1.28000000e+02,  1.08000000e+02,\n         1.21000000e+02,  1.78000000e+02,  1.81000000e+02,\n         1.61000000e+02,  1.71000000e+02,  1.75000000e+02,\n         2.21000000e+02,  2.15000000e+02,  2.18000000e+02,\n         2.21000000e+02,  2.01000000e+02,  2.01000000e+02],\n       [ 6.30000000e+01,  6.60000000e+01,  1.23000000e+02,\n         1.23000000e+02,  1.26000000e+02,  1.33000000e+02,\n         1.36000000e+02,  1.36000000e+02, ...
# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]   #boxes:np.hstack([q1, q2, np.expand_dims(score,1), reg])
    y1 = boxes[:,1]   #其中hstack表示horizon排列,vstack表示vertical排列
    x2 = boxes[:,2]                  
    y2 = boxes[:,3]  #上面4行表示从q1,q2中切出box坐标.
    s = boxes[:,4]   #对应的分数
    area = (x2-x1+1) * (y2-y1+1) #表示boxes里面每一框的像素个数.
    I = np.argsort(s)     #I里面是按照s的大小升序排列
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:#当I.size==1时候,counter=1,然后下面切pick[0:1] pick长度1
        i = I[-1] #最大对应的index，I： [22 10  4  8  5  0  2 12 11 24 16  6 13  7 17 20 23 21  1 19  3 18  9 15 14] 然后变成： [22 10  4  8  5  0  2 12 11 24 16  6  7 17 20 23 21  1 19  3 18  9] 这时看出来每一次的while循环就是把其中的部分指标扔了。所以每一次i都上当前最大score对应index
        pick[counter] = i
        counter += 1
        idx = I[0:-1] #除了I的最后一个
        xx1 = np.maximum(x1[i], x1[idx])  #把数组x1[idx]中的每一个位置的数都跟x1[i]取max
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])   #注意这个地方是min，而上面是取max
        yy2 = np.minimum(y2[i], y2[idx])#得到的xx1,yy1,xx2,yy2是就是表示找到与best_score区域的交集记做inter区域
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1) #w,h表示上面这个inter区域的横纵像素个数.
        inter = w * h#看看那些矩形组成是有效的,多用numpy速度快
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])#除以所有面积里面最小的
        else:
            o = inter / (area[i] + area[idx] - inter) #分母表示的就是并的面积,分子是交的面积.
        I = I[np.where(o<=threshold)]#最核心的代码就是这个赛选框啦!这个<=0.5表示交/并不能超过百分之50.超过了就表示跟bes_score部分重叠太多,没有讨论必要.因为best_index=i已经用pick[counter] = i放入pick数组里面了!这样就把这部分代码基本说清楚了!
    pick = pick[0:counter]     
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):#这个函数处理画的框超出边界的情况,把超出的框给缩回去.
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:,0].copy().astype(np.int32)
    y = total_boxes[:,1].copy().astype(np.int32)
    ex = total_boxes[:,2].copy().astype(np.int32)#ex表示end_x
    ey = total_boxes[:,3].copy().astype(np.int32)

    tmp = np.where(ex>w)#tmp:超出w边界的坐标
    edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
    ex[tmp] = w
    
    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5#通过这2行把长方形的框缩小成方形的.
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA

def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #@UndefinedVariable
    return im_data

    # This method is kept for debugging purpose
#     h=img.shape[0]
#     w=img.shape[1]
#     hs, ws = sz
#     dx = float(w) / ws
#     dy = float(h) / hs
#     im_data = np.zeros((hs,ws,3))
#     for a1 in range(0,hs):
#         for a2 in range(0,ws):
#             for a3 in range(0,3):
#                 im_data[a1,a2,a3] = img[int(floor(a1*dy)),int(floor(a2*dx)),a3]
#     return im_data

