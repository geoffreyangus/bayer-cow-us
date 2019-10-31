import os
import os.path as path
from collections import defaultdict

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

from cow_tus.data.datasets import training_ingredient as datasets_ingredient
from cow_tus.data.dataloaders import training_ingredient as dataloaders_ingredient
from cow_tus.analysis.metrics.metrics import metrics_ingredient


class Harness:
    def __init__(self):
        """
        """
        raise NotImplementedError('please subclass Harness to use with sacred configs.')

    def init_datasets(self, datasets_config={}):
        """
        """
        def init_dataset(class_name, args):
            return getattr(datasets, class_name)(split_name, **args)

        datasets = {}
        for split, split_config in datasets_config['splits'].items():
            datasets[split] = init_dataset(datasets_config['split_names'][split], **split_config)

        return datasets

    def init_dataloaders(self, dataloaders_config={}):
        """
        """
        def init_dataloader(self, class_name, args, split):
            return getattr(dataloaders, class_name)(self.datasets[split], **args)

        dataloaders = {}
        for split, split_config in dataloaders_config['splits'].items():
            dataloaders[split] = init_dataloader(split, **split_config)

        return dataloaders

    def init_tasks(self, tasks):
        """
        """
        return tasks

    def init_model(self, class_name, args, tasks):
        """
        """
        return getattr(models, class_name)(**args)

    def predict(self, X):
        """
        """
        probas = self.predict_proba(X)

        preds = {}
        for module_name, module_probas in probas.items():
            if self.tasks[module_name]['type'] == 'classification':
                preds[module_name] = torch.argmax(dim=-1)(module_probas)
            else:
                raise NotImplementedError
        return preds

    def predict_proba(self, X):
        """
        """
        output = self.model(X)

        probas = {}
        for module_name, module_output in output.items():
            if self.tasks[module_name]['type'] == 'classification':
                probas[module_name] = nn.Softmax(dim=-1)(module_output['logits'])
            else:
                raise NotImplementedError
        return probas

    def predict_for_split(self, split):
        """
        """
        res = []
        for i, (X, y_true, info) in enumerate(self.dataloaders[split]):
            res.append(self.predict(x))
        return res

    def predict_proba_for_split(self, split):
        """
        """
        res = []
        for i, (X, y_true, info) in enumerate(self.dataloaders[split]):
            res.append(self.predict_proba(x))
        return res

    def score(self, X, y, breakdown=False):
        """
        """
        raise NotImplementedError

    def score_for_split(self, split, breakdown=True)
        """
        """
        all_y_true = defaultdict(list)
        all_probas = defaultdict(list)
        all_info = []
        for i, (X, y_true, info) in enumerate(self.dataloaders[split]):
            probas = self.predict_proba(X)
            for module_name in self.tasks:
                task = self.tasks[module_name]['task']
                all_info[module_name] += info
                all_y_true[module_name] += y_true[task]
                all_probas[module_name] += probas[module_name]
        return compute_metrics(all_y_true, all_probas, breakdown)

    def compute_metrics(self, all_y_true, all_probas, all_info=None, breakdown=True):
        """
        """
        metrics = []
        for module_name in self.tasks:
            task = self.tasks[module_name]['task']
            metric_fns = self.tasks[module_name]['metric_fns']

            y_true = all_y_true[module_name]
            probas = all_probas[module_name]
            for metric_fn in metric_fns:
                fn = metric_fn['fn']
                args = metric_fn['args']

                metrics.append({
                    'module_name': module_name,
                    'task': task,
                    f'{fn}': getattr(metrics, fn)(y_true, probas, **args)
                })
        metrics_df = pd.DataFrame(metrics)

        if breakdown:
            breakdown_data = []
            for module_name in self.tasks:
                m = len(info)
                for i in range(m):
                    breakdown_data.append({
                        'task': self.tasks[module_name]['task'],
                        'type': self.tasks[module_name]['type'],
                        'info':  all_info[module_name]['info'][i]
                        'label': all_y_true[module_name]['y_true'][i],
                        'proba': all_probas[module_name]['probas'][i],
                    })
            breakdown_df = pd.DataFrame(breakdown_data)
            return metrics_df, breakdown_df
        else:
            return metrics_df, None



