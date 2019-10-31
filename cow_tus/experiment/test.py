import os
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optimizers
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

# import cow_tus.models.zoo as models
# import cow_tus.models.losses as losses
# import cow_tus.data.datasets as datasets
# import cow_tus.data.dataloaders as dataloaders
from cow_tus.experiment.harness import Harness

from cow_tus.data.datasets import training_ingredient as datasets_ingredient
from cow_tus.data.dataloaders import training_ingredient as dataloaders_ingredient
from cow_tus.analysis.metrics.metrics import metrics_ingredient


EXPERIMENT_NAME = 'train'
ex = Experiment(EXPERIMENT_NAME, ingredients=[datasets_ingredient,
                                              dataloaders_ingredient,
                                              metrics_ingredient])


@ex.config
def config(metrics):
    """
    """
    devices = [0, 1]
    if torch.cuda.is_available():
        device = devices[0]
    else:
        device = 'cpu'

    model = {
        'class_name': 'BaseModel',
        'args': {
            'devices': devices,
            'modules': [
                {
                    'class_name': 'ClippedI3D',
                    'args': {
                        'modality': 'flow',
                        'weights_path': 'models/i3d/model_flow.pth',
                    },

                    'srcs': ['_raw.loops'],
                    'name': 'shared',
                    'dsts': ['classification.binary.primary'],
                },
                {
                    'class_name': 'AttentionDecoder',
                    'args': {},

                    'srcs': ['shared'],
                    'name': 'classification.binary.primary',
                    'dsts': ['_loss'],
                }
            ],
            'module_defaults': {
                'ClippedI3D': {},
                'AttentionDecoder': {}
            },
            'load_paths': [],
        }
    }

    tasks = {}
    for module in model['args']['modules']:
        if '_loss' in module['dsts']:
            module_name = module['name']
            module_namespace = module_name.split('.')
            classifier_type, task = module_namespace[:-1], module_namespace[-1]

            tasks[module_name] = {
                'task': task
                'type': classifier_type[0],
                'metric_fns': metrics['type_to_fns'][classifier_type[-1]], # most specific classifier type
            }

class TestingHarness(Harness):

    def __init__(self):
        """
        """
        self.datasets = self._init_datasets()
        self.dataloaders = self._init_dataloaders()
        self.tasks = self._init_tasks()
        self.model = self._init_model()

    @ex.capture
    def _init_datasets(self, datasets):
        return super()._init_datasets(datasets)

    @ex.capture
    def _init_dataloaders(self, dataloaders):
        return super()._init_dataloaders(dataloaders)

    @ex.capture
    def _init_tasks(self, tasks):
        return super()._init_tasks(tasks)

    @ex.capture(prefix="model")
    def _init_model(self, class_name, args):
        return super()._init_model(class_name, **args)

    @ex.capture
    def test():
        """
        """
        test_metrics_df, test_breakdown_df = self.score_for_split('test')
        # TODO: implement saving test metrics
        return test_metrics_df, test_breakdown_df


@ex.main
def main(_run):
    """
    """
    harness = TestingHarness()
    results = harness.test()
    return results

if __name__ == '__main__':
    ex.run_commandline()