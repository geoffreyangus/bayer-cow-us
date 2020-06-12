"""
Functional implementation of several basic data-augmentation transforms
for pytorch video inputs.

Sources:
- https://github.com/hassony2/torch_videovision
- https://github.com/YU-Zhiyang/opencv_transforms_torchvision
"""

import numbers
import random

import cv2
import numpy as np
import PIL
import scipy
import torch
import torchvision
from sacred import Ingredient


training_ingredient = Ingredient('transforms')


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
        },
        {
            'fn': 'extract_instance',
            "args": {
                "instance_only": True,
                "p_add_diff_class": 0.0,
                "p_add_same_class": 0.0,
                "splits": [
                    "train",
                    "valid"
                ]
            },
        }
    ]

    augmentation_fns = [
        # {
        #     'fn': 'shuffle',
        #     'args': {}
        # },
#         {
#             'fn': 'extract_instance',
#             'args': {
#                 'p_add_same_class': 0.0,
#                 'p_add_diff_class': 0.0,
#                 'instance_only': True,
#                 'splits': ['train', 'valid']
#             }
#         },
#         {
#             'fn': 'random_flip',
#             'args': {
#                 'axis': 0
#             }
#         },
#         {
#             'fn': 'random_flip',
#             'args': {
#                 'axis': 1
#             }
#         },
#         {
#             'fn': 'random_flip',
#             'args': {
#                 'axis': 2
#             }
#         },
#         {
#             'fn': 'jitter',
#             'args': {
#                 'brightness': [0.5]
#             }
#         }
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


INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}

# PREPROCESSING


def normalize(clip):
    return (clip - np.mean(clip)) / np.std(clip)


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
        scaled = np.expand_dims(scaled, axis=3)
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


# AUGMENTATION


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def random_offset(clip, offset=3, offset_range=[0,1]):
    """
    """
    assert offset >= max(offset_range), \
        f'offset {offset} must be greater than or equal to max(offset_range)'
    return clip[random.randint(*offset_range)::offset]


def random_flip(clip, axis=0):
    """
    """
    if random.random() < 0.5:
        return np.flip(clip, axis)
    return clip


def jitter(clip, brightness=[], contrast=[], saturation=[], hue=[]):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    clip (np.ndarray) (T, H, W, C) matrix
    dims (list) the indices of the channels to change
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """
    if isinstance(clip[0], np.ndarray):
        num_channels = clip.shape[3] # iterate through channels
        if len(brightness) == 0:
            brightness = [0] * num_channels
        if len(contrast) == 0:
            contrast = [0] * num_channels
        if len(saturation) == 0:
            saturation = [0] * num_channels
        if len(hue) == 0:
            hue = [0] * num_channels

        for i in range(num_channels):
            brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                get_jitter_params(
                    brightness[i], contrast[i], saturation[i], hue[i])
            if brightness_factor != None:
                clip[:,:,:,i] = clip[:,:,:,i] * brightness_factor
            if contrast_factor != None:
                mean = round(clip[:,:,:,i].mean())
                clip[:,:,:,i] = (1 - contrast_factor) * mean + contrast_factor * clip[:,:,:,i]
            if saturation_factor != None:
                raise NotImplementedError('Saturation augmentation ' +
                                          'not yet implemented')
            if hue_factor != None:
                raise NotImplementedError('Hue augmentation not ' +
                                          'yet implemented.')

    else:
        raise TypeError('Expected numpy.ndarray ' +
                        'but got list of {0}'.format(type(clip[0])))
    return clip


def get_jitter_params(brightness, contrast, saturation, hue):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(
            max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor