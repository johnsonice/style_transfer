import tensorflow as tf
import numpy as np
import utils
import vgg19
import style_transfer
import os

import argparse

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

"""add one dim for batch"""
# VGG19 requires input dimension to be (batch, height, width, channel)
def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)

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
        'style_layer_weights':[.2,.2,.2,.2,.2]
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

# create a map for style layers info
STYLE_LAYERS = {}
for layer, weight in zip(args.style_layers, args.style_layer_weights):
    STYLE_LAYERS[layer] = weight

#%%
# open session
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

## build the graph
#st = style_transfer.StyleTransfer(session = sess,
#                                  content_layer_ids = CONTENT_LAYERS,
#                                  style_layer_ids = STYLE_LAYERS,
#                                  init_image = add_one_dim(init_image),
#                                  content_image = add_one_dim(content_image),
#                                  style_image = add_one_dim(style_image),
#                                  net = vgg_net,
#                                  num_iter = args.num_iter,
#                                  loss_ratio = args.loss_ratio,
#                                  content_loss_norm_type = args.content_loss_norm_type,
#                                  )
## launch the graph in a session
#result_image = st.update()
#
## close session
#sess.close()
#
## remove batch dimension
#shape = result_image.shape
#result_image = np.reshape(result_image,shape[1:])
#
## save result
#utils.save_image(result_image,args.output)
#
## utils.plot_images(content_image,style_image, result_image)

