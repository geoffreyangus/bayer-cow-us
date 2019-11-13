import uuid
import os
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optimizers
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

import cow_tus.models.losses as losses
from cow_tus.experiment.harness import Harness

from cow_tus.data.datasets import training_ingredient as datasets_ingredient
from cow_tus.data.dataloaders import training_ingredient as dataloaders_ingredient
from cow_tus.analysis.metrics.metrics import metrics_ingredient


EXPERIMENT_NAME = 'train'
ex = Experiment(EXPERIMENT_NAME, ingredients=[datasets_ingredient,
                                              dataloaders_ingredient,
                                              metrics_ingredient])


@ex.config
def config(metrics, datasets):
    """
    """
    devices = [0, 1]
    if torch.cuda.is_available():
        device = devices[0]
    else:
        device = 'cpu'

    num_epochs = 20
    remote_dir = '/data/cow-tus-data/weights'

    dataset_dir = 'data/split/by-animal-number/hold-out-validation'
    labels_path = 'data/labels/globals.csv'

    model = {
        'class_name': 'Model',
        'args': {
            'modules': [
                {
                    'class_name': 'I3DEncoder',
                    'args': {
                        'modality': 'flow',
                        'weights_path': 'i3d/model_flow.pth',
                    },

                    'srcs': ['_raw-loops'],
                    'name': 'shared',
                    'dsts': ['classification-binary-primary'],
                },
                {
                    'class_name': 'AttDecoder',
                    'args': {
                        'num_classes': 2
                    },

                    'srcs': ['shared'],
                    'name': 'classification-binary-primary',
                    'dsts': ['_loss'],
                }
            ],
            'module_defaults': {
                'I3DEncoder': {},
                'AttDecoder': {}
            },
            'load_paths': [],
            'devices': devices,
        }
    }

    task_to_labels = {
        'primary': 'primary'
    }

    tasks = {}
    for module in model['args']['modules']:
        if '_loss' in module['dsts']:
            module_name = module['name']
            module_namespace = module_name.split('-')
            task_type, task = module_namespace[:-1], module_namespace[-1]

            tasks[module_name] = {
                'task': task,
                'type': task_type[0],                                # classification | regression (for model.predict functionality)
                'metric_fns': metrics['type_to_fns'][task_type[-1]], # most specific task group distinction (for metrics / analysis)
                'metric_primary': 'roc_auc',
                'labels_src': task_to_labels[task]
            }

    optimizer = {
        'class_name': 'Adam',
        'args': {
            "lr": 0.0001,
            "weight_decay": 0.0
        }
    }

    loss = {
        'class_name': 'CrossEntropyLoss',
        'args': {}
    }

class TrainingHarness(Harness):

    def __init__(self):
        """
        """
        self.uuid = uuid.uuid1()

        self.datasets = self._init_datasets()
        self.dataloaders = self._init_dataloaders()
        self.tasks = self._init_tasks()
        self.model = self._init_model()

        self.optimizer = self._init_optimizer()
        self.loss_fn = self._init_loss()

        self.task_best_metric = {task['task']: task['metric_primary'] for _, task in self.tasks.items()}
        self.task_bests = {task['task']: 0.0 for _, task in self.tasks.items()}

    @ex.capture
    def _init_datasets(self, dataset_dir, labels_path, datasets, tasks):
        return super()._init_datasets(dataset_dir, labels_path, datasets, tasks)

    @ex.capture
    def _init_dataloaders(self, dataloaders):
        return super()._init_dataloaders(dataloaders)

    @ex.capture
    def _init_tasks(self, tasks):
        return super()._init_tasks(tasks)

    @ex.capture(prefix="model")
    def _init_model(self, class_name, args):
        return super()._init_model(class_name, args)

    @ex.capture(prefix='optimizer')
    def _init_optimizer(self, class_name, args):
        return getattr(optimizers, class_name)(self.model.parameters(), **args)

    @ex.capture(prefix='loss')
    def _init_loss(self, class_name, args):
        return getattr(losses, class_name)(**args)

    @ex.capture
    def train(self, num_epochs):
        """
        """
        valid_metrics_df, valid_breakdown_df = self.score_for_split('valid', breakdown=True)
        for i in range(num_epochs):
            train_metrics_df, train_breakdown_df = self.train_for_epoch()
            valid_metrics_df, valid_breakdown_df = self.score_for_split('valid', breakdown=True)
            for task in self.tasks:
                task_metric = valid_metrics_df.loc[valid_metrics_df['task'] == 'task'][self.task_best_metric[task]]
                if task_metric > self.task_bests[task]:
                    self.task_bests[task] = task_metric
                    self._save(train_metrics_df, train_breakdown_df,
                               valid_metrics_df, valid_breakdown_df, save_type=f'best_{task}')
            self._save(train_metrics_df, train_breakdown_df,
                       valid_metrics_df, valid_breakdown_df, save_type='last')

        return valid_metrics_df, valid_breakdown_df

    @ex.capture
    def train_for_epoch(self):
        """
        """
        all_y_true = defaultdict(list)
        all_probas = defaultdict(list)
        all_info = []
        for i, (X, y_true, info) in enumerate(self.dataloaders['train']):
            output, loss = self._optimize(X, y_true)
            probas = self.model.to_probas(output)
            for module_name in self.task_heads:
                task = self.tasks_heads[module_name]['task']
                all_info[module_name] += info
                all_y_true[module_name] += y_true[task]
                all_probas[module_name] += probas[module_name]
        return self.compute_metrics(all_y_true, all_probas, all_info, breakdown=True)

    @ex.capture
    def _optimize(self, X, y_true):
        """
        """
        output = self.model(X)
        loss = 0.0
        # TODO: backprop here
        return output, loss

    @ex.capture
    def _save(self,
              train_metrics_df, train_breakdown_df,
              valid_metrics_df, valid_breakdown_df,
              group_dir, remote_dir, save_type="last"):
        """
        """
        save_dir = path.join(group_dir, save_type)

        weights_path = path.join(remote_dir, f'{self.uuid}_{save_type}')
        weights_link = path.join(save_dir, 'weights.link')
        self.model.save_weights(weights_path, link_path=link_path)
        ex.add_artifact(link_path)

        train_metrics_path = path.join(save_dir, 'train_metrics.csv')
        train_metrics_df.to_csv(train_metrics_path)
        ex.add_artifact(train_metrics_path)

        train_breakdown_df.to_csv(save_dir)
        train_breakdown_df.to_csv(train_breakdown_path)
        ex.add_artifact(train_breakdown_path)

        valid_metrics_path = path.join(save_dir, 'valid_metrics.csv')
        valid_metrics_df.to_csv(valid_metrics_path)
        ex.add_artifact(valid_metrics_path)

        valid_breakdown_df.to_csv(save_dir)
        valid_breakdown_df.to_csv(valid_breakdown_path)
        ex.add_artifact(valid_breakdown_path)


@ex.capture
def init_observers(group_dir):
    """
    """
    ex.observers.append(FileStorageObserver(group_dir))

@ex.main
def main(_run):
    """
    """
    harness = TrainingHarness()
    results = harness.train()
    return results


if __name__ == '__main__':
    ex.run_commandline()