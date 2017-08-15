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

options = {'checkpoint': 'style_net/training_model',
         'device': '/gpu:0'
         }
#%%
img = 'style_net/examples/content/chicago.jpg'
net = inference.net(options)
#%%
result = net.predict(img)
plt.imshow(result)
net.save('test.jpg',result)

################
## training ####
################
#%%
from style_net import style

# set up hyper premeters 
class args(object):
    """
    turn dictionary into objects 
    """
    def __init__(self,dictionary):
        for key,val in dictionary.items():
            setattr(self,key,val)

options_dict = {
            'style': 'style_net/examples/style/udnie.jpg',
            'checkpoint_dir': 'style_net/training_model',
            'model_name': 'udnie.ckpt',
            'test': None,         # 'examples/test/stata.jpg', 
            'test_dir': None,     # 'examples/test',       
            'content_weight': 7.0e0,  # default is 7.5
            'style_weight': 1e2,
            'checkpoint_iterations' : 10,          #2000,
            'batch_size': 4,                    ## after these are all default options
            'train_path': 'style_net/data/train_test',
            'slow': False,
            'epochs':2,
            'vgg_path': 'style_net/data/imagenet-vgg-verydeep-19.mat',
            'tv_weight':2e2,
            'learning_rate':1e-3,
            'DEVICE': '/gpu:0'
        }

options = args(options_dict)

#%%
styles = ['kandinsky','la_muse','rain_princess','seated-nude']

for s in styles:
    options.style = 'style_net/examples/style/' + s + '.jpg'
    options.model_name = s + '.ckpt'
    
    style.train(options)