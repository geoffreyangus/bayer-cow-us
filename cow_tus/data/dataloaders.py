import os

import numpy as np
import torch
from sacred import Ingredient


training_ingredient = Ingredient('dataloaders')


@training_ingredient.config
def config():
    """
    Dataloader configurations by split.
    """
    # modify with dataloaders.batchsize
    batch_size = 1

    train = {
        'class': 'Dataloader',
        'args': {
            'batch_size': batch_size,
            'num_workers': 8,
            'shuffle': False,
            'sampler': 'RandomSampler',
            'num_samples': 2000,
            'replacement': True,
        }
    }

    valid = {
        'class': 'Dataloader',
        'args': {
            'batch_size': batch_size,
            'num_workers': 8,
            'shuffle': True,
        }
    }