#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:45:53 2017

@author: huang
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from style_net import inference
import tensorflow as tf 
import numpy as np
import os 
from utils import get_img,down_size_img
import scipy
import matplotlib.pyplot as plt
%matplotlib inline

#%%
### export to pb ##
def export_to_pb(ckpt,pb_path):
    
    options = {
            'checkpoint': '../style_net/model/starry-night.ckpt',
            'device': '/cpu:0',
            'gpu_memory': 0.0,
            'max_size': 1080
             }
    net = inference.net(options)
    
    output_node = ['img_placeholder','add_37']
    graph_def = tf.graph_util.convert_variables_to_constants(net.sess,net.sess.graph_def,output_node)
    tf.train.write_graph(graph_def,'.','test.pb',as_text=False)

#%%
### laod pb file ### 
def load_pb():
    path = 'test.pb'
    tf.reset_default_graph()
    with open(path,mode='rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')
    
    sess = tf.Session()
    input_node = sess.graph.get_tensor_by_name('img_placeholder:0')
    output_node = sess.graph.get_tensor_by_name('add_37:0')
        #sess.run(out_node, feed_dict={in_node:img})
    
    return input_node,output_node,sess
#%%
### use pb file to predict ####
def predict(img,mx,sess,input_node,output_node):
    ## read image into nd array
    if type(img) != np.ndarray: img = get_img(img)
    ## if image it too big, rescale it 
    h,w,d = img.shape
    scale = 1.0
    if h > w and h > mx:
        scale = mx/h
        img = down_size_img(img,scale)
    if w > mx:
        scale = mx/w
        img = down_size_img(img,scale)
    
    img = np.expand_dims(img,0)
    _preds = sess.run(output_node, feed_dict={input_node:img})
    _preds = np.clip(_preds.squeeze(),0,255).astype(np.uint8)
    
    ## scale the image back to original size if needed
    if scale != 1.0:
        _preds = scipy.misc.imresize(_preds,(h,w,d))

    return _preds

#%%
ckpt='../style_net/model/starry-night.ckpt'
pb_path = 'test.pb'
export_to_pb(ckpt,pb_path)
input_node,output_node,sess = load_pb()

#%%
img_path = '1.jpg'
img = get_img(img_path)
result = predict(img,1080,sess,input_node,output_node)

#%%
plt.imshow(result)

