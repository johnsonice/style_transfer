# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:43:33 2017

@author: chuang
"""

from style_net import evaluate
import matplotlib.pyplot as plt
#import numpy as np
#%matplotlib inline

options = {'checkpoint': 'style_net/training_model',
         'device': '/gpu:0'
         }
#%%
img = 'style_net/examples/content/chicago.jpg'
net = evaluate.net(options)
#%%
result = net.predict(img)
plt.imshow(result)
plt.imsave('test.jpg',result)
