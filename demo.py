# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:43:33 2017

@author: chuang
"""

from style_net import inference
import matplotlib.pyplot as plt
#import numpy as np
#%matplotlib inline


################
### inference ##
################

options = {'checkpoint': 'style_net/model/seated-nude.ckpt',
         'device': '/gpu:0'
         }
net = inference.net(options)
#%%
img = 'style_net/examples/content/content6.jpg'
result = net.predict(img)
plt.imshow(result)
net.save('test.jpg',result)

