import sys
import os
src = os.path.join(os.path.dirname(os.path.realpath(__file__)),'src')
sys.path.insert(0, src)
import transform, numpy as np
import scipy.misc

import tensorflow as tf
from utils import get_img, exists,down_size_img #, list_files ,save_img

class net(object):
    def __init__(self,options):
        
        self.model = options['checkpoint']
        self.device = options['device']
        self.gpu_memory = options['gpu_memory']
        
        print('loading style net')
        self.sess,self.in_node,self.out_node = self.load(self.model,self.device)

    def load(self,model,device_t):
        # build the graph 
        tf.reset_default_graph()
        g = tf.Graph()
        # initiate a session 
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        if self.gpu_memory > 0.0:
            soft_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory
        else:
            #soft_config.gpu_options.allow_growth = True
            pass
        sess = tf.Session(graph = g, config=soft_config)
        
        with g.as_default(), g.device(device_t):
            #tf.reset_default_graph()
            img_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,3],
                                             name='img_placeholder')
            preds = transform.net(img_placeholder)
    
            saver = tf.train.Saver()
            if os.path.isdir(model):
                ckpt = tf.train.get_checkpoint_state(model)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
            else:
                if check_file(model+'.index'):
                    saver.restore(sess, model)
                else:
                    raise Exception("No checkpoint found...")
                    
        in_node = sess.graph.get_tensor_by_name('img_placeholder:0')
        out_node = sess.graph.get_tensor_by_name('add_37:0')
        return sess,in_node,out_node

    def predict(self,img,mx):
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
        _preds = self.sess.run(self.out_node, feed_dict={self.in_node:img})
        _preds = np.clip(_preds.squeeze(),0,255).astype(np.uint8)
        
        ## scale the image back to original size if needed
        if scale != 1.0:
            _preds = scipy.misc.imresize(_preds,(h,w,d))
    
        return _preds
    
    def save(self,out_path,img):
        scipy.misc.imsave(out_path, img)
        return None

def check_file(file_path):
    return os.path.isfile(file_path)
