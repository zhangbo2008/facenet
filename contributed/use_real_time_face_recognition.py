import numpy as np
import os
import face
from scipy import misc
import pickle


#下面这个函数跑完就会把图片矩阵都放进data.pkl文件中
def pretreatment_saving_embedding():
    path_for_all_pic2=os.path.dirname(__file__) + "/../data/images/data_set/"
    path_for_all_pic=os.listdir(path_for_all_pic2)
    print(path_for_all_pic)
    tool=face.Recognition()  
    a=[]
    for i in  path_for_all_pic:
        img = misc.imread(path_for_all_pic2+i, mode='RGB')
        tmp=tool.identify2(img)#这个tool.identify里面进行了detect,align和reshape到160*160
        a.append(tmp[0].embedding)
    tmp=a
    tmp2=path_for_all_pic
    
    #pickle 进去
    
    
    
    
    
    output = open('data.pkl', 'wb')
    
    # Pickle dictionary using protocol 0.
    pickle.dump(tmp, output)
    
    # Pickle the list using the highest protocol available.
    pickle.dump(tmp2, output)
    
    output.close()
    
    pkl_file = open('data.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    
    
    data2 = pickle.load(pkl_file)
    print(data1)
    print(data2)
    pkl_file.close()
    
# pretreatment_saving_embedding()
# raise
#     
# pretreatment_saving_embedding() 
import real_time_face_recognition
real_time_face_recognition.main([])

