'''
这个文件是用来调用align_dataset_mtcnn的.
'''



'''
This problem is solved !
The error was due to the version of tensorflow
So the model was build using a different version than the one I was using
I had tensorflow 1.5
So i install tensorflow 1.3 and it worked !
'''

import tensorflow as tf
print(tf.__version__)


import validate_on_lfw


# import sys
# sys.path.append(r'../')#利用系统变量中加入上级目录,下面就能引入上级目录的库包了.
# import src.align.test111 as test111
# print('引入上级库包成功')
# test111.main()
# import os
# print(os.path.abspath('../'))
# print(os.path.abspath('~'))
# print(os.path.abspath('.'))      #.表示当前所在文件夹的路径

parameter=validate_on_lfw.parse_arguments([
'../datasets/lfw/lfw_mtcnnpy_160' ,
'20180402-114759' ,
'--distance_metric', '1',
'--use_flipped_images' ,
'--subtract_mean' ,
'--use_fixed_image_standardization'
    
    ])
def main():
    return  validate_on_lfw.main(
        (
               parameter
            
            )
         )
    

print(main())














