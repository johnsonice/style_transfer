import tensorflow as tf
import numpy as np
import utils
import vgg19
import collections
import argparse

#%%
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Image Style Transfer Using Convolutional Neural Networks"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_path', type=str, default='pre_trained_model', help='The directory where the pre-trained model was saved')
    parser.add_argument('--content', type=str, default='images/tubingen.jpg', help='File path of content image (notation in the paper : p)', required = True)
    parser.add_argument('--style', type=str, default='images/starry-night.jpg', help='File path of style image (notation in the paper : a)', required = True)
    parser.add_argument('--output', type=str, default='result.jpg', help='File path of output image', required = True)
	
    parser.add_argument('--loss_ratio', type=float, default=1e-3, help='Weight of content-loss relative to style-loss')

    parser.add_argument('--content_layers', nargs='+', type=str, default=['conv4_2'], help='VGG19 layers used for content loss')
    parser.add_argument('--style_layers', nargs='+', type=str, default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                        help='VGG19 layers used for style loss')

    parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[1.0], help='Content loss for each content is multiplied by corresponding weight')
    parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[.2,.2,.2,.2,.2],
                        help='Style loss for each content is multiplied by corresponding weight')

    parser.add_argument('--initial_type', type=str, default='content', choices=['random','content','style'], help='The initial image for optimization (notation in the paper : x)')
    parser.add_argument('--max_size', type=int, default=512, help='The maximum width or height of input images')
    parser.add_argument('--content_loss_norm_type', type=int, default=3, choices=[1,2,3], help='Different types of normalization for content loss')
    parser.add_argument('--num_iter', type=int, default=1000, help='The number of iterations to run')

    return parser.parse_args()

#%%
## update args information
#args = vars(parse_args())
#if args is None:
#    exit()

args = {
        'content': 'images/tubingen.jpg',
        'style': 'images/starry-night.jpg',
        'output': 'result.jpg',
        'model_path': 'pre_trained_model',
        'max_size':None,
        'initial_type':'content',
        'content_layers':['conv4_2'],
        'style_layers':['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        'content_layer_weights':[1.0],
        'style_layer_weights':[.2,.2,.2,.2,.2],
        'num_iter':300,    
        'loss_ratio':1e-3,              ## content loss relative to style loss, in the paper, it suggets (1e-4,1e-1)
        'content_loss_norm_type': 3     ## Different types of normalization for content loss, choice [1,2,3]
        }

#%%
# initiate VGG19 model
model_file_path = args['model_path'] + '/' + vgg19.MODEL_FILE_NAME
vgg_net = vgg19.VGG19(model_file_path)
#%%
# load content image and style image
content_image = utils.load_image(args['content'], max_size=args['max_size'])
# resize style image the same size as content image 
style_image = utils.load_image(args['style'], shape=(content_image.shape[0],content_image.shape[1])) ## dimentions are reversed for PIL.Image,resize

#%%
# initial guess for output
# the picture where you want to add gredient on 
if args['initial_type'] == 'content':
    init_image = content_image
elif args['initial_type'] == 'style':
    init_image = style_image
elif args['initial_type'] == 'random':
    init_image = np.random.normal(size=content_image.shape, scale=np.std(content_image))
#%%
# check input images for style-transfer
# utils.plot_images(content_image,style_image, init_image)

# create a map for content layers info
CONTENT_LAYERS = {}
for layer, weight in zip(args['content_layers'],args['content_layer_weights']):
    CONTENT_LAYERS[layer] = weight
CONTENT_LAYERS = collections.OrderedDict(sorted(CONTENT_LAYERS.items())) 
# create a map for style layers info
STYLE_LAYERS = {}
for layer, weight in zip(args['style_layers'], args['style_layer_weights']):
    STYLE_LAYERS[layer] = weight
STYLE_LAYERS = collections.OrderedDict(sorted(STYLE_LAYERS.items())) 
#%%

## proprocess data 
p0 = np.float32(vgg_net.preprocess(content_image))
a0 = np.float32(vgg_net.preprocess(style_image))
x0 = np.float32(vgg_net.preprocess(init_image)) ## trainable initial image
tf.reset_default_graph()

def _build_graph():
    
    ## trainable initial image 
    x = tf.Variable(x0,trainable=True,dtype=tf.float32)   ## passed x0 
    x = tf.expand_dims(x,0)
    
    ## graph input 
    p_in = tf.placeholder(tf.float32,shape=p0.shape,name='contnet')
    p = tf.expand_dims(p_in,0)  ## make it 4 dimension
    a_in = tf.placeholder(tf.float32,shape=a0.shape,name='style')
    a = tf.expand_dims(a_in,0)  ## make it 4 dimension

    ## get content feature for content loss 
    content_layers = vgg_net.feed_forward(p,scope = 'content')
    Ps = {}
    for id in CONTENT_LAYERS:
        Ps[id] = content_layers[id]  ## extract nodes for all content layers
    
    ## get style layer for style loss 
    style_layers = vgg_net.feed_forward(a,scope='style')
    As = {}
    for id in STYLE_LAYERS:
        As[id]=_gram_matrix(style_layers[id])
    
    Fs = vgg_net.feed_forward(x,scope='mixed')
    
    #### comput loss 
    L_content = 0 
    L_style = 0 
    
    for id in Fs:
        if id in CONTENT_LAYERS:
            ##content loss##
            F = Fs[id]                  ## content feature of x 
            P = Ps[id]                  ## contnet feature of P 
            _, h, w, d = F.get_shape()  ## first return value is batch size
            N = h.value*w.value         ## product of width and height 
            M = d.value 
            w = CONTENT_LAYERS[id]      ## the relative weights for content layers, if there is only one layer, it is 1
            
            # You may choose different normalization constant
            if args['content_loss_norm_type'] == 1:
                L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / 2     ## this is one half of sum squared error
                                                                         ## cost used in original paper 
            elif args['content_loss_norm_type'] == 2:
                L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / (N*M) #artistic style transfer for videos
            elif args['content_loss_norm_type'] == 3:   # this is from https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
                L_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F - P), 2))
    
        elif id in STYLE_LAYERS:
            F = Fs[id]
            _, h, w, d = F.get_shape()  ## first return value is batch size
            N = h.value*w.value         ## product of width and height 
            M = d.value
            w = STYLE_LAYERS[id]        ## relative weights for styles layers 
            
            G = _gram_matrix(F)         ## style feature of x 
            A = As[id]                  ## style feature of a 
            
            L_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-A), 2))
    ## end of for id in Fs loop 
    
    # fix beta as 1 
    alpha = args['loss_ratio']
    beta = 1  # this can be changed, depends on how much weights you put on style 
    L_total = alpha*L_content + beta*L_style


    """ define optimizer L-BFGS """
    # this call back function is called every after loss is updated
    global _iter
    _iter = 0
    def callback(tl, cl, sl):
        global _iter
        print('iter : %4d/%4d' % (_iter,args['num_iter']), 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
        _iter += 1
    
    ## optimizer 
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(L_total, method='L-BFGS-B', options={'maxiter': args['num_iter']})
    
    ## initialize variable 
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    optimizer.minimize(sess,feed_dict={a_in:a0,p_in:p0},
                           fetches=[L_total, L_content, L_style], loss_callback=callback)
    
    ## get the return output image 
    final_image = sess.run(x)   ## so x is the only veriable here that is being updated 
    
    # ensure the image has valid pixel-values between 0 and 255
    final_image = np.clip(vgg_net.undo_preprocess(final_image), 0.0, 255.0)
    final_image = np.squeeze(final_image)
    
    return final_image
    
def _gram_matrix(tensor):
    
    shape = tensor.get_shape()
    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])
    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram
#%%
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
return_image = _build_graph()
sess.close()

#%%
utils.save_image(return_image,args['output'])
utils.plot_images(content_image,style_image, return_image)


