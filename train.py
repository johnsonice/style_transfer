#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:28:49 2017

@author: chengyu
"""

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
            'content_weight': 7.5e0,  # default is 7.5
            'style_weight': 1e2,
            'checkpoint_iterations' : 2000,
            'batch_size': 4,                    ## after these are all default options
            'train_path': 'style_net/data/train2014',
            'slow': False,
            'epochs':2,
            'vgg_path': 'style_net/data/imagenet-vgg-verydeep-19.mat',
            'tv_weight':2e2,
            'learning_rate':1e-3,
            'DEVICE': '/gpu:0'
        }

options = args(options_dict)

#%%
styles = ['hayao_miyazaki_totoro','transverse_line','robotech','super_saiyan','western_dream',
'sketch','woman-with-hat-matisse']

for s in styles:
    options.style = 'style_net/examples/style/' + s + '.jpg'
    options.model_name = s + '.ckpt'
    
    style.train(options)
