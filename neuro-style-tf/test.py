# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:03:19 2017

@author: chuang
"""

## test scripts 

#import numpy as np
from src import vgg19,util
#import cv2

#%%
x = util.read_image('styles/kandinsky.jpg') ## return 4 d, demeaned image
vgg = vgg19.build_model(x)

#%%