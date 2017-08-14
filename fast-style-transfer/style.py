from __future__ import print_function
import sys, os
sys.path.insert(0, './src')
#import numpy as np, scipy.misc 
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

## turn dictionary into an object 
class args(object):
    """
    turn dictionary into objects 
    """
    def __init__(self,dictionary):
        for key,val in dictionary.items():
            setattr(self,key,val)
    
#%%
##%%
#parser = build_parser()
#options_dict = vars(parser.parse_args())


options_dict = {
            'style': 'examples/style/udnie.jpg',
            'checkpoint_dir': 'training_model',
			'model_name': 'udnie.ckpt',
            'test': None,         # 'examples/test/stata.jpg', 
            'test_dir': None,     # 'examples/test',       
            'content_weight': 7.0e0,  # default is 7.5
            'style_weight': 1e2,
            'checkpoint_iterations' : 2000,
            'batch_size': 4,                    ## after these are all default options
            'train_path': 'data/train2014',
            'slow': False,
            'epochs':2,
            'vgg_path': 'data/imagenet-vgg-verydeep-19.mat',
            'tv_weight':2e2,
            'learning_rate':1e-3,
            'DEVICE': '/gpu:0'
        }

options = args(options_dict)


#%%
    
def main():
#    parser = build_parser()
#    options = parser.parse_args()
#    check_opts(options)

    style_target = get_img(options.style)
    content_targets = _get_files(options.train_path)


    kwargs = {
        "slow":options.slow,
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size":options.batch_size,
        "save_path":os.path.join(options.checkpoint_dir,options.model_name),
        "learning_rate":options.learning_rate
    }

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)

    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)

if __name__ == '__main__':
    main()
