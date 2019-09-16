#!/usr/bin/env python
# coding: utf-8

# # IMPORT PACKAGES

# In[1]:
# -*- coding: utf-8 -*-

'''
项目地址:https://github.com/sjchoi86/Tensorflow-101
'''


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#get_ipython().run_line_magic('matplotlib', 'inline')
print ("Packages loaded.")


# # FACE DETECTOR

# #### https://raw.githubusercontent.com/shantnu/Webcam-Face-Detect/master/haarcascade_frontalface_default.xml

# In[3]:


# Load Face Detector
cwd = os.getcwd()
print(cwd)
clsf_path = cwd + "/data/haarcascade_frontalface_default.xml"
print(clsf_path)
face_cascade = cv2.CascadeClassifier(clsf_path)
print ("face_cascade is %s" % (face_cascade))


# # LOAD IMAGE WITH FACES

# In[5]:
#'''
#C:\Users\zhangbo284\Desktop\python项目全体\srcAll\Tensorflow-101-master\images\celebs.jpg
#C:\Users\zhangbo284\Desktop\python项目全体\srcAll\Tensorflow-101-master\images
#'''

# THIS IS BGR 
imgpath = cwd + "\images\celebs.jpg"
# imgpath = cwd + "/../../img_dataset/celebs/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0001.jpg"
img_bgr = cv2.imread(imgpath)
#img_bgr = cv2.imread('C:/Users/zhangbo284/Desktop/celebs.jpg')
#print(cwd,2222222222)
#print(imgpath,33333333333)
#print(img_bgr,44444444)

# CONVERT TO RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


# # DETECT FACE

# In[4]:

#cv2.imshow('sdfasd',img_bgr)
#cv2.waitKey(1)
#raise
faces = face_cascade.detectMultiScale(img_gray)
print ("%d faces deteced. " % (len(faces)))


# # PLOT DETECTED FACES

# In[5]:


# PLOT
plt.figure(0)
plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
ca = plt.gca()
for face in faces:
    ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
                           , fill=None, alpha=1, edgecolor='red'))
plt.title("Face detection with Viola-Jones")
plt.draw()


# # DETECT FACES IN THE IMAGES IN A FOLDER

# In[10]:







