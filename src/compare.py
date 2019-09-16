"""Performs face alignment and calculates L2 distance between the embeddings of images."""

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

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face

def main(args):
    '''
    传进来的args是一个对象,所以下面对象.属性可以访问他的属性值.
    '''
    #next line,we crop the picture's face and resize it to 160*160
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)#在tf中,运行完语句会自动进行析构操作.析构掉session,虽然这行没写session但是底层用了tf就会跑完这行之后自动析构里面的session
    with tf.Graph().as_default():#底层用threading建立图,with里面最后一句跑完会关闭然后析构
        
        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            '''
             tf.get_default_graph().get_tensor_by_name在图中按照名称读取数据,
             因为tf的特点就是把变量都放到一个图里面.
            '''#这时变量都放到会话和图里面了.当前的会话是facenet网络,不是load_and_align_data这个函数里面的图像align网络了.
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")   
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#上面3行获取图中的变量给他们定义3个变量,一个输入input,一个输出embedding一个标签:phase_train(这里面是使用所以下面给标签赋值False)
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)#得到emb就是需要的嵌入后的向量.
            '''
            emb:得到的嵌入后的向量.
            '''
            '''
            nrof_images:需要判定的图片的数量.
            '''
            nrof_images = len(args.image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, args.image_files[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            '''
            输出一个字典output
            '''
            output={}
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                    output[(i,j)]=dist
                print('')
                print(output)
            return output

            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    '''
    读取图片然后矫正图片,我理解就是把图片里面的人脸部分扣出来
    '''
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) 
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    #上面得到的pnet,r,o 是3个函数.下面用这3个函数放入detect_face里面进行裁剪图片.2019-01-03,23点01明天继续从这里搞
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')#读取图片到一个np.array,这个变量可以在变量监视器里面看到里面的shape等属性值.
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:#bounding_boxes 返回5个数,前4个表示坐标最后一个表示得分.
          image_paths.remove(image)
          print("can't detect face, remove ", image)#去掉没有脸的图片
          continue
        det = np.squeeze(bounding_boxes[0,0:4])#det得到的框的坐标
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)   #det跟边框进行裁剪运算,margin表示的是:det跟最后需要的框之间的距离*2
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')#misc.imresize就是解析度的变化,把像素变成(image_size, image_size)
        prewhitened = facenet.prewhiten(aligned)#进行白化.
        img_list.append(prewhitened)
    images = np.stack(img_list)#把img_list按照0轴摞起来.具体就是把2个160*160*3的变成一个2*160*160*3的array对象.
    '''
    返回一个array,内容是每一个图片.
    '''
    return images


def parse_arguments(argv):
    print(argv)
    print(type(argv))
    print(type(argv[0]))
    print(type(argv[1]))

    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
