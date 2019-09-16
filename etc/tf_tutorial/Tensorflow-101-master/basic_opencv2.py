#!/usr/bin/env python
# coding: utf-8

# # IMPORT PACKAGES

# In[1]:


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# get_ipython().run_line_magic('matplotlib', 'inline')
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


# THIS IS BGR 
imgpath = cwd + "/images/celebs.jpg"
print(imgpath,9999999999)
# imgpath = cwd + "/../../img_dataset/celebs/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0001.jpg"
img_bgr = cv2.imread(imgpath)
print(img_bgr)
# CONVERT TO RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


# # DETECT FACE

# In[4]:


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


path = cwd + "/../../img_dataset/celebs/Arnold_Schwarzenegger"
flist = os.listdir(path)
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
for f in flist:
    if os.path.splitext(f)[1].lower() not in valid_exts:
        continue
    fullpath = os.path.join(path, f)
    img_bgr = cv2.imread(fullpath)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img_gray)
    # PLOT
    plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
    ca = plt.gca()
    for face in faces:
        ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
                               , fill=None, alpha=1, edgecolor='red'))
    plt.title("Face detection with Viola-Jones")
    plt.show()


# In[ ]:




