import sys, os
src = os.path.join(os.path.dirname(os.path.realpath(__file__)),'src')
sys.path.insert(0, src)
#import numpy as np, scipy.misc 
from optimize import optimize
from utils import save_img, get_img, exists, list_files

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

    
def train(options):
    
    style_target = get_img(options.style)
    content_targets = _get_files(options.train_path)
    if len(content_targets) ==0: raise ValueError('Can not load training image, double check training data dir!') 

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
