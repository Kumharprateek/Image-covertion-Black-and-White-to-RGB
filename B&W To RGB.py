#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install opencv-python


# In[3]:


get_ipython().system('pip install IPython')


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import Image
cv2.__version__
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


image=cv2.imread ('thor.jpg',0)
image


# In[10]:


print(image)


# In[5]:


pip install opencv-python


# In[13]:


img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[15]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray


# In[16]:


negative = 255-img
plt.imshow(negative)


# In[17]:


blured=cv2.GaussianBlur(negative,(21,21),0)
blured


# In[19]:


def dodge_img(x,y):
    return cv2.divide(x,255-y,scale=250)
dodged_img = dodge_img(img,blured)
plt.imshow(dodged_img)


# In[21]:


def burn_img(image,mask):
    return 255-(cv2.divide(255-image,255-mask,scale=200))
final_image=burn_img(dodged_img,blured)
plt.imshow(final_image)


# In[22]:


cv2.imwrite('thor-b&w.jpg',final_image)


# 
