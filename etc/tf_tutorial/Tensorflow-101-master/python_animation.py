#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pylab as pl
from IPython import display
for i in range(100):
    pl.plot(pl.randn(100))
    display.clear_output(wait=True)
    display.display(pl.gcf())
    # time.sleep(0.1)
    pl.clf()


# In[ ]:




