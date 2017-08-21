import tensorflow as tf
import numpy as np 
import time                       
import cv2
import os

from src.util import maybe_make_directory,write_image,read_image,check_image,preprocess,postprocess,read_flow_file,read_weights_file,normalize
from src.vgg19 import build_model

'''
  'a neural algorithm for artistic style' loss functions
'''
def content_layer_loss(p, x):
  _, h, w, d = p.get_shape()
  M = h.value * w.value
  N = d.value
  if args.content_loss_function   == 1:
    K = 1. / (2. * N**0.5 * M**0.5)
  elif args.content_loss_function == 2:
    K = 1. / (N * M)
  elif args.content_loss_function == 3:  
    K = 1. / 2.
  loss = K * tf.reduce_sum(tf.pow((x - p), 2))
  return loss

def style_layer_loss(a, x):
  _, h, w, d = a.get_shape()
  M = h.value * w.value
  N = d.value
  A = gram_matrix(a, M, N)
  G = gram_matrix(x, M, N)
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
  return loss

def gram_matrix(x, area, depth):
  F = tf.reshape(x, (area, depth))
  G = tf.matmul(tf.transpose(F), F)
  return G

def mask_style_layer(a, x, mask_img):
  _, h, w, d = a.get_shape()
  mask = get_mask_image(mask_img, w.value, h.value)
  mask = tf.convert_to_tensor(mask)
  tensors = []
  for _ in range(d.value): 
    tensors.append(mask)
  mask = tf.stack(tensors, axis=2)
  mask = tf.stack(mask, axis=0)
  mask = tf.expand_dims(mask, 0)
  a = tf.multiply(a, mask)
  x = tf.multiply(x, mask)
  return a, x

def sum_masked_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  masks = args.style_mask_imgs
  for img, img_weight, img_mask in zip(style_imgs, weights, masks):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      a, x = mask_style_layer(a, x, img_mask)
      style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  for img, img_weight in zip(style_imgs, weights):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_content_losses(sess, net, content_img):
  sess.run(net['input'].assign(content_img))
  content_loss = 0.
  for layer, weight in zip(args.content_layers, args.content_layer_weights):
    p = sess.run(net[layer])
    x = net[layer]
    p = tf.convert_to_tensor(p)
    content_loss += content_layer_loss(p, x) * weight
  content_loss /= float(len(args.content_layers))
  return content_loss

#############################################################################################
'''
  rendering -- where the magic happens
'''
#############################################################################################
def stylize(content_img, style_imgs, init_img, frame=None):
  with tf.device(args.device), tf.Session() as sess:
    # setup network
    net = build_model(content_img)
    
    # style loss
    if args.style_mask:
      L_style = sum_masked_style_losses(sess, net, style_imgs)
    else:
      L_style = sum_style_losses(sess, net, style_imgs)
    
    # content loss
    L_content = sum_content_losses(sess, net, content_img)
    
    # denoising loss
    L_tv = tf.image.total_variation(net['input'])
    
    # loss weights
    alpha = args.content_weight
    beta  = args.style_weight
    theta = args.tv_weight
    
    # total loss
    L_total  = alpha * L_content
    L_total += beta  * L_style
    L_total += theta * L_tv

    # optimization algorithm
    optimizer = get_optimizer(L_total)

    if args.optimizer == 'adam':
      minimize_with_adam(sess, net, optimizer, init_img, L_total)
    elif args.optimizer == 'lbfgs':
      minimize_with_lbfgs(sess, net, optimizer, init_img)
    
    output_img = sess.run(net['input'])
    
    if args.original_colors:
      output_img = convert_to_original_colors(np.copy(content_img), output_img)

    if args.video:
      write_video_output(frame, output_img)
    else:
      write_image_output(output_img, content_img, style_imgs, init_img)

def minimize_with_lbfgs(sess, net, optimizer, init_img):
  if args.verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, loss):
  if args.verbose: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
  train_op = optimizer.minimize(loss)
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  iterations = 0
  while (iterations < args.max_iterations):
    sess.run(train_op)
    if iterations % args.print_iterations == 0 and args.verbose:
      curr_loss = loss.eval()
      print("At iterate {}\tf=  {:.5E}".format(iterations, curr_loss))
    iterations += 1

def get_optimizer(loss):
  print_iterations = args.print_iterations if args.verbose else 0
  if args.optimizer == 'lbfgs':
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B',
      options={'maxiter': args.max_iterations,
                  'disp': print_iterations})
  elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
  return optimizer

def write_video_output(frame, output_img):
  fn = args.content_frame_frmt.format(str(frame).zfill(4))
  path = os.path.join(args.video_output_dir, fn)
  write_image(path, output_img)
  
def write_image_output(output_img, content_img, style_imgs, init_img):
  out_dir = os.path.join(args.img_output_dir, args.img_name)
  maybe_make_directory(out_dir)
  img_path = os.path.join(out_dir, args.img_name+'.png')
  content_path = os.path.join(out_dir, 'content.png')
  init_path = os.path.join(out_dir, 'init.png')

  write_image(img_path, output_img)
  write_image(content_path, content_img)
  write_image(init_path, init_img)
  index = 0
  for style_img in style_imgs:
    path = os.path.join(out_dir, 'style_'+str(index)+'.png')
    write_image(path, style_img)
    index += 1
    
  # save the configuration settings
  out_file = os.path.join(out_dir, 'meta_data.txt')
  f = open(out_file, 'w')
  f.write('image_name: {}\n'.format(args.img_name))
  f.write('content: {}\n'.format(args.content_img))
  index = 0
  for style_img, weight in zip(args.style_imgs, args.style_imgs_weights):
    f.write('styles['+str(index)+']: {} * {}\n'.format(weight, style_img))
    index += 1
  index = 0
  if args.style_mask_imgs is not None:
    for mask in args.style_mask_imgs:
      f.write('style_masks['+str(index)+']: {}\n'.format(mask))
      index += 1
  f.write('init_type: {}\n'.format(args.init_img_type))
  f.write('content_weight: {}\n'.format(args.content_weight))
  f.write('style_weight: {}\n'.format(args.style_weight))
  f.write('tv_weight: {}\n'.format(args.tv_weight))
  f.write('content_layers: {}\n'.format(args.content_layers))
  f.write('style_layers: {}\n'.format(args.style_layers))
  f.write('optimizer_type: {}\n'.format(args.optimizer))
  f.write('max_iterations: {}\n'.format(args.max_iterations))
  f.write('max_image_size: {}\n'.format(args.max_size))
  f.close()

'''
  image loading and processing
'''
def get_init_image(init_type, content_img, style_imgs, frame=None):
  if init_type == 'content':
    return content_img
  elif init_type == 'style':
    return style_imgs[0]
  elif init_type == 'random':
    init_img = get_noise_image(args.noise_ratio, content_img)
    return init_img
  # only for video frames
  elif init_type == 'prev':
    init_img = get_prev_frame(frame)
    return init_img
  elif init_type == 'prev_warped':
    init_img = get_prev_warped_frame(frame)
    return init_img

def get_content_frame(frame):
  fn = args.content_frame_frmt.format(str(frame).zfill(4))
  path = os.path.join(args.video_input_dir, fn)
  img = read_image(path)
  return img

def get_content_image(content_img):
  path = os.path.join(args.content_img_dir, content_img)
   # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  h, w, d = img.shape
  mx = args.max_size
  # resize if > max size
  if h > w and h > mx:
    w = (float(mx) / float(h)) * w
    img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
  if w > mx:
    h = (float(mx) / float(w)) * h
    img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
  img = preprocess(img) ## de - mean the image 
  
  return img

def get_style_images(content_img):
  _, ch, cw, cd = content_img.shape
  style_imgs = []
  for style_fn in args.style_imgs:
    path = os.path.join(args.style_imgs_dir, style_fn)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    style_imgs.append(img)
  return style_imgs

def get_noise_image(noise_ratio, content_img):
  np.random.seed(args.seed)
  noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
  img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
  return img

def get_mask_image(mask_img, width, height):
  path = os.path.join(args.content_img_dir, mask_img)
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  check_image(img, path)
  img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
  img = img.astype(np.float32)
  mx = np.amax(img)
  img /= mx
  return img

def get_prev_frame(frame):
  # previously stylized frame
  prev_frame = frame - 1
  fn = args.content_frame_frmt.format(str(prev_frame).zfill(4))
  path = os.path.join(args.video_output_dir, fn)
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  return img

def get_prev_warped_frame(frame):
  prev_img = get_prev_frame(frame)
  prev_frame = frame - 1
  # backwards flow: current frame -> previous frame
  fn = args.backward_optical_flow_frmt.format(str(frame), str(prev_frame))
  path = os.path.join(args.video_input_dir, fn)
  flow = read_flow_file(path)
  warped_img = warp_image(prev_img, flow).astype(np.float32)
  img = preprocess(warped_img)
  return img

def get_content_weights(frame, prev_frame):
  forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
  backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
  forward_path = os.path.join(args.video_input_dir, forward_fn)
  backward_path = os.path.join(args.video_input_dir, backward_fn)
  forward_weights = read_weights_file(forward_path)
  backward_weights = read_weights_file(backward_path)
  return forward_weights #, backward_weights

def warp_image(src, flow):
  _, h, w = flow.shape
  flow_map = np.zeros(flow.shape, dtype=np.float32)
  for y in range(h):
    flow_map[1,y,:] = float(y) + flow[1,y,:]
  for x in range(w):
    flow_map[0,:,x] = float(x) + flow[0,:,x]
  # remap pixels to optical flow
  dst = cv2.remap(
    src, flow_map[0], flow_map[1], 
    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
  return dst

def convert_to_original_colors(content_img, stylized_img):
  content_img  = postprocess(content_img)
  stylized_img = postprocess(stylized_img)
  if args.color_convert_type == 'yuv':
    cvt_type = cv2.COLOR_BGR2YUV
    inv_cvt_type = cv2.COLOR_YUV2BGR
  elif args.color_convert_type == 'ycrcb':
    cvt_type = cv2.COLOR_BGR2YCR_CB
    inv_cvt_type = cv2.COLOR_YCR_CB2BGR
  elif args.color_convert_type == 'luv':
    cvt_type = cv2.COLOR_BGR2LUV
    inv_cvt_type = cv2.COLOR_LUV2BGR
  elif args.color_convert_type == 'lab':
    cvt_type = cv2.COLOR_BGR2LAB
    inv_cvt_type = cv2.COLOR_LAB2BGR
  content_cvt = cv2.cvtColor(content_img, cvt_type)
  stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
  c1, _, _ = cv2.split(stylized_cvt)
  _, c2, c3 = cv2.split(content_cvt)
  merged = cv2.merge((c1, c2, c3))
  dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
  dst = preprocess(dst)
  return dst

def render_single_image():
  content_img = get_content_image(args.content_img)
  style_imgs = get_style_images(content_img)
  with tf.Graph().as_default():
    print('\n---- RENDERING SINGLE IMAGE ----\n')
    init_img = get_init_image(args.init_img_type, content_img, style_imgs)
    tick = time.time()
    stylize(content_img, style_imgs, init_img)
    tock = time.time()
    print('Single image elapsed time: {}'.format(tock - tick))


class args_obj(object):
    """
    turn dictionary into objects 
    """
    def __init__(self,dictionary):
        for key,val in dictionary.items():
            setattr(self,key,val)
#%%
def default_args():
    args={
            'verbose':True,
            'style_mask':False,
            'model_weights':'imagenet-vgg-verydeep-19.mat',
            'image_name':'testing.jpg',
            'style_imgs':['starry-night.jpg'],
            'img_output_dir':'./image_output',
            'style_imgs_weights':[1.0],
            'content_img':'face.jpg',
            'style_imgs_dir':'./styles',
            'content_img_dir':'./image_input',  ## default
            'init_img_type':'content',          ## default  ['random', 'content', 'style']
            'max_size':512,                   ## default is 512 
            'content_weight':5e0,               ## default
            'style_weight':1e4,                 ## default 
            'tv_weight':1e-3,                   ## default
            'content_loss_function':1,          ## choice 1, 2, 3
            'content_layers':['conv4_2'],       ## default
            'style_layers': ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],   ## default
            'content_layer_weights':[1.0],       ## default
            'style_layer_weights':[0.2,0.2,0.2,0.2,0.2],        ## default 
            'original_colors':False,            ## True or False, if keep original color
            'color_convert_tyle': 'yuv',        ## choice ['yuv', 'ycrcb', 'luv', 'lab'] Color space for conversion to original colors
            'color_convert_time':'after',       ## before or after
            'noise_ratio':1.0,                  ## default: "Interpolation value between the content image and noise image if the network is initialized with 'random'.")
            'seed':0,                           ## default 
            'pooling_tyle':'avg',               ## max pooling or avg pooling 
            'device':'/cpu:0',                  ## or '/cpu:0'
            'optimizer': 'lbfgs',               ## or adam 
            'learning_rate':1e0,                ## default learning rate for adam 
            'max_iterations':1000,              ## default 
            'print_iterations': 1,
            'videoo' :False,   
            'video_output_dir':'./video_output',
            'content_frame_frmt':'frame_{}.ppm'
            }
    arg = args_obj(args)
    
    arg.style_layer_weights   = normalize(arg.style_layer_weights)
    arg.content_layer_weights = normalize(arg.content_layer_weights)
    arg.style_imgs_weights    = normalize(arg.style_imgs_weights)
    
    return arg 
#%%
#def main():
global args
args = default_args()
#render_single_image()
content_img = get_content_image(args.content_img)  ## one 4 d image 
style_imgs = get_style_images(content_img)  ## a list of style images 
with tf.Graph().as_default():
    print('\n---- RENDERING SINGLE IMAGE ----\n')
    init_img = get_init_image(args.init_img_type, content_img, style_imgs)
    tick = time.time()
    stylize(content_img, style_imgs, init_img)
    tock = time.time()
    print('Single image elapsed time: {}'.format(tock - tick))  
  
#if __name__ == '__main__':
#  main()
  
#%%


