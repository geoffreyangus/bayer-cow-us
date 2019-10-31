"""
"""
import os

import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
import skvideo.io

def play(video):
    """
    """
    fig = plt.figure(figsize=(12, 8))
    im = plt.imshow(video[0,:,:])
    plt.close() # this is required to not display the generated image

    def init():
        im.set_data(video[0,:,:])

    def animate(i):
        im.set_data(video[i,:,:])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                   interval=50)
    return anim

def process_loop(filepath, size=(120, 160), skip=1):
    """
    """
    video = skvideo.io.vread(filepath)
    video = resize_clip(video, size)
    D, H, W, C = video.shape
    video = video[::skip, :, W//4:(3*W)//4, 0]
    return video

def process_loops(filedir, size=(120, 160), skip=1):
    """
    """
    concat = []
    for filename in tqdm(os.listdir(filedir)):
        concat.append(process_loop(os.path.join(filedir, filename), size, skip))
    video = np.concatenate(concat, axis=0)
    np.prod(video.shape)
    return video