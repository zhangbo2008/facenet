
'''
出入参数一个是图片路径,一个是文件夹路径,输出一个图片路径表示文件夹路径中跟图片路径最像的那个图片是什么
然后显示这2个图片.用肉眼对比.
'''

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
from  compare import * #利用这个方法来覆盖compare里面的main方法,同时引入compare里面的其他函数.
def main(args):
    '''
    传进来的args是一个对象,所以下面对象.属性可以访问他的属性值.
    '''
    #next line,we crop the picture's face and resize it to 160*160
    '''
    把all_pic_in_lab里面的文件都拼成一个list
    
    '''
    tmp2=args.all_pic_in_lab[0]
#    print(tmp)
    tmp=os.listdir(tmp2)
#    print(tmp)    
#    raise
    for i in range(len(tmp)):
        tmp[i]=tmp2+'\\'+tmp[i]
#    print(tmp)
    
    '''
    下面的load_and_align_data这个函数,如果文件夹里面图片没有脸,那么图片就自动去除在集合里面,也就是不输出结果.
    '''
    
    all_pic_in_lab = load_and_align_data(tmp, args.image_size, args.margin, args.gpu_memory_fraction)
#    print(all_pic_in_lab.shape)
    
    
    pic_to_compare = load_and_align_data(args.pic_to_compare, args.image_size, args.margin, args.gpu_memory_fraction)
    all_pic=np.concatenate(( pic_to_compare,all_pic_in_lab), axis=0)
#    print(all_pic.shape)
#    print(all_pic_in_lab.shape)
#    raise
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            '''
             tf.get_default_graph().get_tensor_by_name在图中按照名称读取数据,
             因为tf的特点就是把变量都放到一个图里面.
            '''
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")   
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: all_pic, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            '''
            emb:得到的嵌入后的向量.
            '''

            '''
            输出一个字典output
            '''
            output={}
            for i in range(1,len(all_pic)):

                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[i,:]))))
#                    print(dist)
#                    print('  %1.4f  ' % dist, end='')
                    output[i]=dist
#            print(output)
            best=min(output, key=output.get)#返回value最小的key值
            return output,tmp[best-1]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
import argparse
import os
import sys

#c=os.path.abspath(os.path.join(os.getcwd(), '..','data','images','Anthony_Hopkins_0002.jpg'))
#print(a)              
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    #the given name for arguments must be fronted by --or-
    parser.add_argument('-model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('-all_pic_in_lab', type=str, nargs='+', help='all_pic_in_lab')
    parser.add_argument('-pic_to_compare', type=str, nargs='+', help='pic_to_compare')
    parser.add_argument('-image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('-margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('-gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)
#print(a)
#print(b)
#把use_facenet_compare_pics.py作为一个函数封装起来.要运行就main()即可.
def main_true():
    return  main(
        (
                parse_arguments(#arguments below
                        
            ['-model','models','-all_pic_in_lab',a,'-pic_to_compare',b,'-image_size','160']
            
            )
         )
    )
'''
这个.py文件用main_true来运行即可.main是底层函数不用管.
a表示图片库
b表示需要判别的图片
返回一个字典.
'''

a=os.path.abspath(os.path.join(os.getcwd(), '..','data','images','data_set'))
b=os.path.abspath(os.path.join(os.getcwd(), '..','data','images','222.png'))
'''
下面一行的表示打印a这个文件夹内最接近b这个图片的图片的文件绝对路径.
'''
result=main_true()[1]

print(result)

'''
下面显示开始和匹配的图片,用肉眼看看像不像!!!!!!!!!!!!!!!!!
'''

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from skimage import io,data
img=b
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread(img) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#lena.shape #(512, 512, 3)

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()



img=result
#result=img.replace("\","/")
#print(img)
#raise
lena = mpimg.imread(img) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#lena.shape #(512, 512, 3)

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

'''
2018-12-30,18点57看出来效果可以.
'''
'''
2019-01-03,14点31
红哥说的不用遍历目标图片和文件夹中图片挨个距离这种O(N)算法.
而使用聚类算法,把文件夹中的图片聚类比如聚10类,找聚类之后的10个中点聚类目标图片最近的作为新的赛选集来做,这样就是O(logN)的算法.
这个算法显然是错误的.并且数据越大错误率会很高.
证明:比如你聚类聚2类,其中第一类中点距离目标很近,但是第二类有一个聚类目标最近的,但是第二类有一堆拖后腿的导致第二类中点距离目标很远.这样答案就错了
证毕.
'''

