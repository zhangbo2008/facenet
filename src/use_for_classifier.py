'''
这个文件是用来调用align_dataset_mtcnn的.
'''
import classifier 
print(34)
print(34)
print(34)
print(34)

import sys
sys.path.append(r'../')#利用系统变量中加入上级目录,下面就能引入上级目录的库包了.
# import src.align.test111 as test111
print('引入上级库包成功')
# test111.main()
import os
print(os.path.abspath('../'))
# print(os.path.abspath('~'))
# print(os.path.abspath('.'))      #.表示当前所在文件夹的路径

parameter=classifier.parse_arguments([
'TRAIN',
' /home/david/datasets/lfw/lfw_mtcnnalign_160 ',
' /home/david/models/model-20170216-091149.pb ',
' ~/models/lfw_classifier.pkl ',
' --batch_size', '1000 ',
 '--min_nrof_images_per_class', '40',
'  --nrof_train_images_per_class', '35 ',
'--use_split_dataset',
    
    ])
def main():
    return  classifier.main(
        (
               parameter
            
            )
         )
    

print(main())














