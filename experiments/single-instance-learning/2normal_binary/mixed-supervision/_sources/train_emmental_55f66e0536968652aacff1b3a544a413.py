import os
import os.path as path
import logging
from functools import partial

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torchvision import transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from cow_tus.data.transforms import training_ingredient as transforms_ingredient
from cow_tus.util.util import unpickle, ce_loss, output
from cow_tus.models.modules import zoo as modules
import cow_tus.data.datasets as all_datasets

EXPERIMENT_NAME = 'trainer'
ex = Experiment(EXPERIMENT_NAME, ingredients=[transforms_ingredient])
ex.logger = logging.getLogger(__name__)
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config(transforms):
    """
    Configuration for training harness.
    """
    hypothesis_conditions = ['single-instance-learning', '2normal_binary', 'mixed-supervision']
    exp_dir = path.join('experiments', *hypothesis_conditions)

    meta_config = {
        'device': 0
    }

    logging_config = {
        'evaluation_freq': 1,
        'checkpointing': True,
        'checkpointer_config': {
            'checkpoint_metric': {
                '2normal_binary/cow-tus-dataset/valid/accuracy': 'max'
            }
        }
    }

    dataset_class = 'GlobalDataset'
    dataset_args = {
        'dataset_dir': 'data/split/by-animal-number/hold-out-validation',
        'labels_path': 'data/labels/globals.csv'
    }

    transform_fns = {
        'train': transforms['preprocess_fns'] + transforms['augmentation_fns'],
        'valid': transforms['preprocess_fns'],
        'test':  transforms['preprocess_fns']
    }

    dataloader_configs = {
        'train': {
            'batch_size': 1,
            'num_workers': 8,
            'shuffle': False
        },
        'valid': {
            'batch_size': 1,
            'num_workers': 8,
            'shuffle': True
        }
    }

    sampler_configs = {
        'train': {
            'num_samples': 150,
            'replacement': True
        }
    }

    task_to_label_dict = {
        '2normal_binary': '2normal_binary',
    }

    encoder_class = 'I3DEncoder'
    encoder_args = {
        'modality': 'gray',
        'weights_path': 'i3d/model_flow.pth'
    }

    decoder_class = "AttDecoder"
    decoder_args = {}

    learner_config = {
        'n_epochs': 30,
        'valid_split': 'valid',
        'optimizer_config': {'optimizer': 'adam', 'lr': 0.01, 'l2': 0.000},
        'lr_scheduler_config': {
            'warmup_steps': None,
            'warmup_unit': 'batch',
            'lr_scheduler': 'step',
            'step_config': {
                'step_size': 6,
                'gamma': 0.5
            }
        },
    }

class TrainingHarness(object):

    def __init__(self):
        """
        """
        self._init_meta()

        self.datasets = self._init_datasets()
        self.dataloaders = self._init_dataloaders()
        self.model = self._init_model()

    @ex.capture
    def _init_meta(self, _seed, exp_dir, meta_config, learner_config, logging_config):
        emmental.init(path.join(exp_dir, '_emmental_logs'))
        Meta.update_config(
            config={
                'meta_config': {**meta_config, 'seed': _seed},
                'model_config': {'device': meta_config['device']},
                'learner_config': learner_config,
                'logging_config': logging_config
            }
        )

    @ex.capture
    def _init_datasets(self, _log, dataset_class, dataset_args, transform_fns):
        datasets = {}
        for split in ['train', 'valid']:
            datasets[split] = getattr(all_datasets, dataset_class)(
                split_str=split,
                transform_fns=transform_fns[split],
                **dataset_args
            )
            _log.info(f'Loaded {split} split.')
        return datasets

    @ex.capture
    def _init_dataloaders(self, _log, dataloader_configs, sampler_configs, task_to_label_dict):
        dataloaders = []
        for split in ['train', 'valid']:
            dataloader_config = dataloader_configs[split]
            if split == 'train':
                dataloader_config = {
                    'sampler': RandomSampler(data_source=self.datasets[split],
                                             **sampler_configs['train']),
                    **dataloader_config
                }
            dl = EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=self.datasets[split],
                split=split,
                **dataloader_config,
            )
            dataloaders.append(dl)
            _log.info(f'Built dataloader for {split} set.')
        return dataloaders

    @ex.capture
    def _init_model(self, encoder_class, encoder_args, decoder_class, decoder_args, task_to_label_dict):
        encoder_module = getattr(modules, encoder_class)(**encoder_args)
        tasks = [
            EmmentalTask(
                name=task_name,
                module_pool=nn.ModuleDict(
                    {
                        f'encoder_module': encoder_module,
                        f'decoder_module_{task_name}': getattr(modules, decoder_class)(2, **decoder_args),
                    }
                ),
                task_flow=[
                    {
                        'name': 'encoder_module', 'module': 'encoder_module', 'inputs': [('_input_', 'exam')]
                    },
                    {
                        'name':   f'decoder_module_{task_name}',
                        'module': f'decoder_module_{task_name}',
                        'inputs': [('encoder_module', 0)],
                    },
                ],
                loss_func=partial(ce_loss, task_name),
                output_func=partial(output, task_name),
                scorer=Scorer(metrics=['accuracy']),
            )
            for task_name in task_to_label_dict.keys()
        ]
        model = EmmentalModel(name='cow-tus-model', tasks=tasks)
        return model

    def run(self):
        learner = EmmentalLearner()
        learner.learn(self.model, self.dataloaders)


@ex.config_hook
def hook(config, command_name, logger):
    if config['exp_dir'] == None:
        raise Exception(f'exp_dir is {config["exp_dir"]}')
    ex.observers.append(FileStorageObserver(config['exp_dir']))


@ex.main
def main():
    trainer = TrainingHarness()
    res = trainer.run()
    return res


if __name__ == '__main__':
    ex.run_commandline()
