'''
这个文件调用compare.py
来实现,输入图片,调用别人配置号的参数来计算几个图片之间的距离.如果距离越小说明越接近.
'''
import compare
'''


#下面3行是该路径的写法
base_img_path=os.path.abspath(os.path.join(os.getcwd(), ".."))
base_img_path=os.path.abspath(os.path.join(base_img_path, ".."))
base_img_path=os.path.abspath(os.path.join(base_img_path, "data"))

def parse_arguments(argv):
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
'''

import argparse
import os
import sys



a=os.path.abspath(os.path.join(os.getcwd(), '..','data','images','Anthony_Hopkins_0001.jpg'))
b=os.path.abspath(os.path.join(os.getcwd(), '..','data','images','Anthony_Hopkins_0002.jpg'))
c=os.path.abspath(os.path.join(os.getcwd(), '..','data','images','Anthony_Hopkins_0002.jpg'))
print(a)              
def parse_arguments(argv):
    parser = argparse.ArgumentParser()              
    #the given name for arguments must be fronted by --or-
    parser.add_argument('-model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('-image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('-image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('-margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('-gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)
print(a)
print(b)
#把use_facenet_compare_pics.py作为一个函数封装起来.要运行就main()即可.
def main():
    return  compare.main(
        (
                parse_arguments(#arguments below
                        
            ['-model','models','-image_files',a,b,'-image_size','160']
            
            )
         )
    )
#a=main()
#print()
#print(a)
#print(type(a))
#print(type(main()))    
print(main())       #返回得到的是所有的距离矩阵.


