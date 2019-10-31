"""
"""
import os
import os.path as path

import numpy as np
import pandas as pd
from sacred import Experiment
import yaml


EXPERIMENT_NAME = 'splitter'
ex = Experiment(EXPERIMENT_NAME)


@ex.config
def config():
    """
    """
    data_dir = 'data/single-instance-learning/temporal-downsample'

    hypothesis_conditions = ['by-animal-number', 'hold-out-validation']
    group_dir = path.join('data', 'split', *hypothesis_conditions)

    split_counts = {
        'test': 15,
        'valid': 15,
        'train': -1
    }


class DataSplitter:

    def __init__(self):
        """
        """
        pass

    @ex.capture
    def run(self, data_dir, hypothesis_conditions, split_counts):
        """
        """
        attrs_path = path.join(data_dir, 'attrs.csv')
        attrs_df = pd.read_csv(attrs_path, index=False)


@ex.config_hook
def hook(config, command_name, logger):
    if config['group_dir'] == None:
        raise Exception(f'group_dir is {config["group_dir"]}')
    ex.observers.append(FileStorageObserver(config['group_dir']))


@ex.main
def main(_run):
    """
    """
    splitter = DataSplitter()
    results = splitter.run()
    return results


if __name__ == '__main__':
    ex.run_commandline()
