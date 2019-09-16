'''
这个文件是用来调用align_dataset_mtcnn的.
'''
import align.align_dataset_mtcnn 
print(34)
print(34)
print(34)
print(34)

import sys
sys.path.append(r'../')#利用系统变量中加入上级目录,下面就能引入上级目录的库包了.
import src.align.test111 as test111
print('引入上级库包成功')
test111.main()
import os
print(os.path.abspath('../'))
# print(os.path.abspath('~'))
# print(os.path.abspath('.'))      #.表示当前所在文件夹的路径

parameter=align.align_dataset_mtcnn.parse_arguments([
   r'../datasets/lfw/raw',
r'../datasets/lfw/lfw_mtcnnpy_160',
'--image_size','160'   ,        
'--margin', '32',
'--random_order', 
'--gpu_memory_fraction', '0.25' 
    
    ])
def main():
    return  align.align_dataset_mtcnn.main(
        (
               parameter
            
            )
         )
    

print(main())














