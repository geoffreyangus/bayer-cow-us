"""
"""
import os

import numpy as np
from sacred import Experiment

EXPERIMENT_NAME = 'splitter'
ex = Experiment(EXPERIMENT_NAME)

@ex.config
def config():
    """
    """
    pass


class DataSplitter:

    def __init__(self):
        """
        """
        pass

    @ex.capture
    def run(self):
        """
        """
        pass


@ex.main
def main(_run):
    """
    """
    splitter = DataSplitter()
    results = splitter.run()
    return results


if __name__ == '__main__':
    ex.run_commandline()
