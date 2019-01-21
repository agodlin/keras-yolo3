"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import random

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx, dy = ((w-nw)//2, (h-nh)//2)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    new_image = Image.fromarray(np.zeros((w, h), dtype=np.float32))
    new_image.paste(Image.fromarray(image), (dx, dy))
    image = np.asarray(new_image)
    image_data = image / 256.
    image_data[image_data > 1] = 1
    image_data[image_data < 0] = 0
    image_data = np.expand_dims(image_data, axis=2)

    return image_data

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def gamma_correction(image, correction=1.0, is_normalized=True, scale=255):
    if not is_normalized:
        image /= scale
    # image **= correction
    image = np.sign(image) * np.abs(image) ** correction
    if not is_normalized:
        image *= scale
    return image
                                            

def get_random_data(annotation_line, input_shape, randomize=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''randomize preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = cv2.imread(line[0], cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ih, iw = image.shape[:2]

    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not randomize:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    
    
    new_image = Image.fromarray(np.zeros((w, h), dtype=np.float32))
    new_image.paste(Image.fromarray(image), (dx, dy))
    image = np.asarray(new_image)

    # flip image or not
    flip = rand()<.5
    if flip:
        image = cv2.flip(image, flipCode=1)

    if False:
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(nn/p.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # distort image
    image = gamma_correction(np.array(image), random.uniform(0.75, 1.25), is_normalized=False, scale=65535)
    image_data = image / 256.
    image_data[image_data > 1] = 1
    image_data[image_data < 0] = 0
    image_data = np.expand_dims(image_data, axis=2)
        
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
