# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:43:33 2017

@author: chuang
"""

from style_net import inference
#import matplotlib.pyplot as plt
import os 
#import numpy as np
#%matplotlib inline


################
### inference ##
################

options = {'checkpoint': 'style_net/training_model/western_dream.ckpt',
         'device': '/cpu:0'
         }
net = inference.net(options)
#%%
imgs_path = 'style_net/examples/content'
out_path = 'style_net/examples/results'
imgs = [f for f in os.listdir(imgs_path)]

#img = 'style_net/examples/content/content6.jpg'
for f in imgs:
    result = net.predict(os.path.join(imgs_path,f))
    #plt.imshow(result)
    net.save(os.path.join(out_path,f),result)

