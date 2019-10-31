import os

import numpy as np
import torch
from sacred import Ingredient

from cow_tus.data.preprocessing import training_ingredient as preprocessing_ingredient
from cow_tus.data.augmentation import training_ingredient as augmentation_ingredient


training_ingredient = Ingredient('datasets', ingredients=[preprocessing_ingredient,
                                                          augmentation_ingredient])

@training_ingredient.config
def training_config(preprocessing, augmentation):
    """
    Dataset configurations by split. For use with TrainingHarness class.
    """
    # allows use of different reprs in filesystem
    split_names = {
        'train': 'train',
        'valid': 'valid',
    }

    train = {
        'class_name': 'GlobalDataset', # GlobalDataset | InstanceDataset | MixedDataset
        'args': {
            'split_dir': split_names['train'],
            'preprocess_fns': preprocessing['preprocess_fns'],
            'augmentation_fns': augmentation['augmentation_fns']
        }
    }

    valid = {
        'class_name': 'GlobalDataset',
        'args': {
            'split_dir': split_names['valid'],
            'preprocess_fns': preprocess_fns,
            'augmentation_fns': []
        }
    }


testing_ingredient = Ingredient('datasets', ingredients=[preprocessing_ingredient])


@testing_ingredient.config
def testing_config(preprocessing):
    """
    Dataset configurations by split. For use with TestingHarness class.
    """
    # allows use of different reprs in filesystem
    split_names = {
        'test': 'test',
    }

    test = {
        'class_name': 'GlobalDataset',
        'args': {
            'split_dir': split_names['test'],
            'preprocess_fns': preprocessing['preprocess_fns'],
            'augmentation_fns': []
        }
    }