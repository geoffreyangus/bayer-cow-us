import numbers
import random

import cv2
import numpy as np
import PIL
import scipy
import torch
import torchvision
from sacred import Ingredient


training_ingredient = Ingredient('preprocessing')

@training_ingredient.config
def training_config():
    preprocess_fns = [
        {
            'fn': 'resize_clip',
            'args': {
                'size': (224, 224)
            }
        },
        {
            'fn': 'normalize',
            'args': {}
        }
    ]

builder_ingredient = Ingredient('preprocessing')

@builder_ingredient.config
def builder_config():
    preprocess_fns = [
        {
            'fn': 'resize_clip',
            'args': {
                'size': (210, 280)
            }
        },
        {
            'fn': 'crop_clip_horizontally_by_proportion',
            'args': {
                'ratio': (0.25, 0.75)
            }
        },
        {
            'fn': 'rgb_to_grayscale',
            'args': {}
        }
    ]


def rgb_to_grayscale(clip):
    d, h, w, c = clip.shape
    gray = np.zeros((d, h, w))
    for i, frame in enumerate(clip):
        gray[i, :, :] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray


def crop_clip_horizontally_by_proportion(clip, ratio):
    d, h, w, c = clip.shape
    min_h = 0
    max_h = h
    min_w = int(w * ratio[0])
    max_w = int(w * ratio[1]) - min_w
    return crop_clip(clip, min_h, min_w, max_h, max_w)


def crop_clip_vertically_by_proportion(clip, ratio):
    d, h, w, c = clip.shape
    min_h = int(h * ratio[0])
    max_h = int(h * ratio[1]) - min_h
    min_w = 0
    max_w = w
    return crop_clip(clip, min_h, min_w, max_h, max_w)



def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = np.array([img[min_h:min_h + h, min_w:min_w + w, :] for img in clip])

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = np.array([
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ])
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow
