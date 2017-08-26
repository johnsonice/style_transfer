#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:46:58 2017
@author: huang
"""

from neuro_style_tf import style_net_single as sns
#%%
options= {
        'max_iterations':20,
        'device':'/cpu:0',
        'model_weights':'./neuro_style_tf/imagenet-vgg-verydeep-19.mat',
        'style_imgs_weights':[1.0],
        'style_imgs_dir':'./neuro_style_tf/styles',
        'style_imgs':['starry-night.jpg'],
        'content_img_dir':'./neuro_style_tf/image_input',
        'content_img':'face.jpg',
        'init_img_type':'content',          ## default  ['random', 'content', 'style']
        'img_output_dir':'./neuro_style_tf/image_output',
        'img_name':'testing.jpg',
        'print_iterations': 5               ## print every 5 iteration for test 
          }
stylenet = sns.net(options)
#%%

stylenet.render_single_image()
