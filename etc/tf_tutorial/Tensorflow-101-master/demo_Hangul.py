#!/usr/bin/env python
# coding: utf-8

# # "data/nine_dreams/ninedreams.txt" IS REQUIRED

# # SPECIFY FILE ENCODING TYOE IN PYTHON

# In[1]:


# -*- coding: utf-8 -*-
print ("UTF-8 ENCODING")


# # LOAD PACKAGES

# In[3]:


import chardet # https://github.com/chardet/chardet
import glob
import codecs
import sys
import os
from TextLoader import *
from Hangulpy3 import *
print ("PACKAGES LOADED") 


# # CONVERT UTF8-ENCODED TXT FILE

# In[4]:


def conv_file(fromfile, tofile):
    with open(fromfile, "rb") as f:
        sample_text=f.read(10240)
    pred = chardet.detect(sample_text)
    if not pred['encoding'] in ('EUC-KR', 'UTF-8', 'CP949', 'UTF-16LE'):
        print ("WARNING! Unknown encoding! : %s = %s") % (fromfile, pred['encoding'])
        pred['encoding'] = "CP949" # 못찾으면 기본이 CP949
        formfile = fromfile + ".unknown"
    elif pred['confidence'] < 0.9:
        print ("WARNING! Unsured encofing! : %s = %s / %s")
        get_ipython().run_line_magic('(fromfile,', "pred['confidence'], pred['encoding'])")
        formfile = fromfile + ".notsure"
    with codecs.open(fromfile, "r", encoding=pred['encoding'], errors="ignore") as f:
        with codecs.open(tofile, "w+", encoding="utf8") as t:
            all_text = f.read()
            t.write(all_text)


# # "data/nine_dreams/ninedreams_utf8.txt" IS GENERATED

# In[5]:


# SOURCE TXT FILE
fromfile = "data/nine_dreams/ninedreams.txt"
# TARGET TXT FILE
tofile   = "data/nine_dreams/ninedreams_utf8.txt"
conv_file(fromfile, tofile)
print ("UTF8-CONVERTING DONE")
print (" [%s] IS GENERATED" % (tofile))


# # DECOMPOSE HANGUL (THIS PART IS IMPORTANT!)

# In[6]:


def dump_file(filename):
    result=u"" # <= UNICODE STRING 
    with codecs.open(filename,"r", encoding="UTF8") as f:
        for line in f.readlines():
            line = tuple(line)
            result = result + decompose_text(line)
    return result
print ("FUNCTION READY")


# # PYTHON 2 AND 3 COMPATIBILITY

# In[7]:


if sys.version_info.major == 2:
    parsed_txt = dump_file(tofile).encode("utf8") 
else:
    parsed_txt = dump_file(tofile) 

print ("Parsing %s done" % (tofile))
# PRINT FIRST 100 CHARACTERS
print (parsed_txt[:100])


# # "data/nine_dreams/input.txt" IS GENERATED

# In[8]:


with open("data/nine_dreams/input.txt", "w") as text_file:
    text_file.write(parsed_txt)
print ("Saved to a txt file")
print (text_file)


# # COMPOSE HANGUL CHARACTER FROM PHONEME 

# In[9]:


data=[u'\u3147', u'\u3157', u'\u1d25', u'\u3134', u'\u3161', u'\u3139', u'\u1d25'
      , u' ', u'\u314f', u'\u3147', u'\u3145', u'\u314f', u'\u1d25', u'\u1d25'
      , u'\u3163', u'\u1d25', u' ', u'\u3147', u'\u1d25', u'\u3155', u'\u1d25'
      , u'\u3134', u'\u314f', u'\u1d25', u'\u3155', u'\u3147', u'\u1d25'
      , u'\u315b', u'\u3131', u'\u1d25', u'\u3147', u'\u3139', u'\u3146'
      , u'\u1d25', u'\u3137', u'\u314f', u'\u314e', u'\u3139', u'\u1d25'
      , u'\u3134', u'\u1d25', u'\u3145', u'\u3163', u'\u1d25', u'\u1d25'
      , u'\u314f', u'\u1d25', u'\u314e', u'\u314f', u'\u3147', u'\u3131'
      , u'\u3157', u'\u3134', u'\u1d25', u'\u1d25', u'\u315b', u'\u1d25'
      , u'\u3148', u'\u3153', u'\u3136', u'\u1d25', u' ', u'\u3145', u'\u3150'
      , u'\u3141', u'\u3136', u'\u3161', u'\u3134', u'\u3163', u'\u1d25', u'.'
      , u'\u3148', u'\u3153', u'\u3134', u'\u314e', u'\u3153', u'\u1d25', u'\u1d25'
      , u'\u3147', u'\u314f', u'\u3134', u'\u3148', u'\u314f', u'\u3139', u'\u315d'
      , u'\u314c', u'\u1d25', u'\u3161', u'\u3134', u'\u3148', u'\u3163', u'\u313a'
      , u'\u1d25', u' ', u'\u3147', u'\u3161', u'\u3146', u'\u1d25', u'?', u'\u3134'
      , u'\u1d25', u'\u314e', u'\u3163', u'\u1d25', u'\u3147', u'\u3148', u'\u314f'
      ]
print automata("".join(data))


# # GENERATE "vocab.pkl" and "data.npy" in "data/nine_dreams/" FROM "data/nine_dreams/input.txt" 

# In[10]:


data_dir    = "data/nine_dreams"
batch_size  = 50
seq_length  = 50
data_loader = TextLoader(data_dir, batch_size, seq_length)


# # DATA_LOADER IS:

# In[11]:


print ( "type of 'data_loader' is %s, length is %d" 
       % (type(data_loader.vocab), len(data_loader.vocab)) )


# # DATA_LOADER.VOCAB IS:

# In[12]:


print ("data_loader.vocab looks like \n%s " % (data_loader.vocab,))


# # DATA_LOADER.CHARS IS:

# In[13]:


print ( "type of 'data_loader.chars' is %s, length is %d" 
       % (type(data_loader.chars), len(data_loader.chars)) )


# # CHARS CONVERTS INDEX -> CHAR

# In[14]:


print ("data_loader.chars looks like \n%s " % (data_loader.chars,))


# In[27]:


for i, char in enumerate(data_loader.chars):
    # GET INDEX OF THE CHARACTER
    idx = data_loader.vocab[char]
    print ("[%02d] %03s (%02d)" 
           % (i, automata("".join(char)), idx))


# In[ ]:




