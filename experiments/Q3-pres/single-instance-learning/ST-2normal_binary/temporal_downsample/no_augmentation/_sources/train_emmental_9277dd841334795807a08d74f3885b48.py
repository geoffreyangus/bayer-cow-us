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
import torch.utils.data as torch_data
from torchvision import transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from cow_tus.data.transforms import training_ingredient as transforms_ingredient
from cow_tus.data.dataloaders import get_sample_weights
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
    task_str = None
    assert task_str, f'task {task_str} must have a value'

    tasks = task_str.split('&')
    for task in tasks:
        if task not in {'primary', 'primary_multiclass', '2normal_binary'}:
            raise ValueError(f'task {task} not recognized')

    # tuner parameters
    instance_level = True
    representative_loop_sampling = False


    temporal_downsample = False
    spatial_downsample = False
    data_augmentation = False

    if instance_level:
        assert not representative_loop_sampling, \
            'instance_level and representative_loop_sampling are mutually exclusive'
    assert not (temporal_downsample and spatial_downsample), \
        'temporal_downsample and spatial_downsample are mutually exclusive'

    hypothesis_conditions = ['Q3-pres']

    # whether or not we are working at an exam level or loop level
    if instance_level:
        hypothesis_conditions.append('instance-level-learning')
    else:
        hypothesis_conditions.append('single-instance-learning')

    # labeling schema we are using
    if len(tasks) > 1:
        hypothesis_conditions.append('MT-' + '&'.join(tasks))
    else:
        hypothesis_conditions.append('ST-' + tasks[0])

    # downsampling procedure
    if temporal_downsample:
        hypothesis_conditions.append('temporal_downsample')
    elif spatial_downsample:
        hypothesis_conditions.append('spatial_downsample')
    else:
        hypothesis_conditions.append('full_size')

    # whether or not we are using data augmentation
    if data_augmentation:
        hypothesis_conditions.append('data_augmentation')
    else:
        hypothesis_conditions.append('no_augmentation')


    exp_dir = path.join('experiments', *hypothesis_conditions)

    meta_config = {
        'device': 0
    }

    logging_config = {
        'evaluation_freq': 1,
        'checkpointing': True,
        'checkpointer_config': {
            'checkpoint_runway': 10,
            'checkpoint_metric': {
                "primary/cow-tus-dataset/valid/accuracy": "max"
            }
        }
    }

    metrics = {}
    for task in tasks:
        if task in {'primary', '2normal_binary'}:
            metrics[task] = ['accuracy', 'roc_auc' , 'precision', 'recall', 'f1']
        elif task in {'primary_multiclass'}:
            metrics[task] = ['accuracy']

    dataset_class = 'InstanceDataset' if instance_level else 'GlobalDataset'
    dataset_args = {
        'dataset_dir': 'data/split/by-animal-number/hold-out-validation',
        'labels_path': f"data/labels/{'instances' if instance_level else 'globals'}.csv"
    }

    if temporal_downsample:                     # resize 224x224, normalize, extract_instance, random_offset
        tmp = transforms['preprocess_fns']
        if representative_loop_sampling:
            tmp = tmp + [transforms['rls_transform_fn']]
        preprocess_transforms = tmp + [transforms['temporal_downsample_transform_fn']]

    elif spatial_downsample:                    # resize 120x80, normalize, extract_instance
        tmp = [transforms['spatial_downsample_transform_fn']] + transforms['preprocess_fns'][1:]
        if representative_loop_sampling:
            tmp = tmp + [transforms['rls_transform_fn']]
        preprocess_transforms = tmp
    else:                                       # resize 224x224, normalize
        preprocess_transforms = transforms['preprocess_fns']

    transform_fns = {
        'train': preprocess_transforms,
        'valid': preprocess_transforms,
        'test':  preprocess_transforms
    }

    if data_augmentation:
        transform_fns['train'] = transform_fns['train'] + transforms['augmentation_fns']

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
            'class_name': 'RandomSampler',
            'args': {
                'num_samples': 150,
                'replacement': True,
            }
        }
    }

    task_to_label_dict = {task: task for task in tasks}
    task_to_cardinality = {
        'primary': 2,
        'primary_multiclass': 4,
        '2normal_binary': 2
    }

    encoder_class = 'I3DEncoder'
    encoder_args = {
        'modality': 'gray',
        'weights_path': 'i3d/model_flow.pth'
    }

    decoder_class = "AttDecoder"
    decoder_args = {
        'dropout_p': 0.0
    }

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
                sampler_class = sampler_configs[split]['class_name']
                sampler_args = sampler_configs[split]['args']
                if sampler_class == 'WeightedRandomSampler':
                    weights = get_sample_weights(self.datasets[split], sampler_args['weight_task'], sampler_args['class_probs'])
                    sampler = getattr(torch_data, sampler_class)(
                                      weights=weights, num_samples=sampler_args['num_samples'], replacement=sampler_args['replacement'])
                else:
                    sampler = getattr(torch_data, sampler_class)(
                                      data_source=self.datasets[split], **sampler_args)
                dataloader_config = {
                    'sampler': sampler,
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
    def _init_model(self, encoder_class, encoder_args, decoder_class, decoder_args, task_to_label_dict, task_to_cardinality, metrics):
        encoder_module = getattr(modules, encoder_class)(**encoder_args)
        tasks = [
            EmmentalTask(
                name=task_name,
                module_pool=nn.ModuleDict(
                    {
                        f'encoder_module': encoder_module,
                        f'decoder_module_{task_name}': getattr(modules, decoder_class)(task_to_cardinality[task_name], **decoder_args),
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
                scorer=Scorer(metrics=metrics[task_name]),
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
