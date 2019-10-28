"""
"""
import os
import sys

from sacred import Experiment

# from cow_tus.models.zoo import *

ex = Experiment()

@ex.config
def my_config():
    experiment_dir = 'experiment/'
    num_epochs = 29

@ex.capture
def run_experiment(experiment_dir, num_epochs):
    print(experiment_dir)
    for i in range(num_epochs):
        print(f'Epoch {i} of {num_epochs}')
    print(ex.observers)

@ex.command
def build_dataset(raw_dir, out_dir):
    print('building dataset')
    print(raw_dir, out_dir)

@ex.command
def split_dataset():
    print('splitting dataset')

@ex.automain
def main():
    run_experiment()