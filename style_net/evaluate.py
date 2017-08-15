import sys
import os
src = os.path.join(os.path.dirname(os.path.realpath(__file__)),'src')
sys.path.insert(0, src)
import transform, numpy as np
import scipy.misc
import tensorflow as tf
from utils import get_img, exists #, list_files ,save_img

def default_options():
    options_dict = {
                    'checkpoint_dir':'net/training_model',
                    'in_path': 'net/examples/content/chicago.jpg',
                    'device': '/gpu:0',
            }
    
    return options_dict

def load(options_dict):
    checkpoint_dir = options_dict['checkpoint_dir']
    device_t = options_dict['device']
    # build the graph 
    tf.reset_default_graph()
    g = tf.Graph()
    # initiate a session 
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    sess = tf.Session(graph = g, config=soft_config)
    
    with g.as_default(), g.device(device_t):
        img_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,3],
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)
    
    in_node = sess.graph.get_tensor_by_name('img_placeholder:0')
    out_node = sess.graph.get_tensor_by_name('add_37:0')
    return sess,in_node,out_node

def predict(sess,in_node,out_node,options_dict):
    img = get_img(options_dict['in_path'])
    img = np.expand_dims(img,0)
    _preds = sess.run(out_node, feed_dict={in_node:img})
    
    return _preds.squeeze()
#%%
def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
