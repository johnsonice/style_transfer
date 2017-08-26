#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:46:58 2017
@author: huang
"""

from style_net_single import neural_style as sns
#%%
options= {
        'max_iterations':20,
        'device':'/cpu:0',
        'model_weights':'./style_net_single/imagenet-vgg-verydeep-19.mat',
        'style_imgs_weights':[1.0],
        'style_imgs_dir':'./style_net_single/styles',
        'style_imgs':['starry-night.jpg'],
        'content_img_dir':'./style_net_single/image_input',
        'content_img':'face.jpg',
        'init_img_type':'content',          ## default  ['random', 'content', 'style']
        'img_output_dir':'./style_net_single/image_output',
        'img_name':'testing.jpg',
        'print_iterations': 5               ## print every 5 iteration for test 
          }
stylenet = sns.net(options)
#%%

stylenet.render_single_image()