# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import struct
import errno

'''
  utilities and i/o
'''
def read_image(path):
  # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  img = preprocess(img)
  return img

def write_image(path, img):
  img = postprocess(img)
  cv2.imwrite(path, img)

def preprocess(img):
  # bgr to rgb
  img = img[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
  img = img[np.newaxis,:,:,:]
  img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  return img

def postprocess(img):
  img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  img = img[0]
  img = np.clip(img, 0, 255).astype('uint8')
  # rgb to bgr
  img = img[...,::-1]
  return img

def read_flow_file(path):
  with open(path, 'rb') as f:
    # 4 bytes header
    header = struct.unpack('4s', f.read(4))[0]
    # 4 bytes width, height    
    w = struct.unpack('i', f.read(4))[0]
    h = struct.unpack('i', f.read(4))[0]   
    flow = np.ndarray((2, h, w), dtype=np.float32)
    for y in range(h):
      for x in range(w):
        flow[0,y,x] = struct.unpack('f', f.read(4))[0]
        flow[1,y,x] = struct.unpack('f', f.read(4))[0]
  return flow

def read_weights_file(path):
  lines = open(path).readlines()
  header = list(map(int, lines[0].split(' ')))
  w = header[0]
  h = header[1]
  vals = np.zeros((h, w), dtype=np.float32)
  for i in range(1, len(lines)):
    line = lines[i].rstrip().split(' ')
    vals[i-1] = np.array(list(map(np.float32, line)))
    vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
  # expand to 3 channels
  weights = np.dstack([vals.astype(np.float32)] * 3)
  return weights

def normalize(weights):
  denom = sum(weights)
  if denom > 0.:
    return [float(i) / denom for i in weights]
  else: return [0.] * len(weights)

def maybe_make_directory(dir_path):
  if not os.path.exists(dir_path):  
    os.makedirs(dir_path)

def check_image(img, path):
  if img is None:
    raise OSError(errno.ENOENT, "No such file", path)


