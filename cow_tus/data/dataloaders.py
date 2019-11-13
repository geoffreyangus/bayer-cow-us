import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler, WeightedRandomSampler
from sacred import Ingredient

from cow_tus.util.util import concat, stack


training_ingredient = Ingredient('dataloaders')


@training_ingredient.config
def config():
    """
    Dataloader configurations by split.
    """
    # modify with dataloaders.batchsize
    batch_size = 1

    splits = {
        'train': {
            'class_name': 'ExamDataLoader',
            'args': {
                'batch_size': batch_size,
                'num_workers': 8,
                'shuffle': False,
                'sampler': 'RandomSampler',
                'num_samples': 100,
                'replacement': True,
            }
        },
        'valid': {
            'class_name': 'ExamDataLoader',
            'args': {
                'batch_size': batch_size,
                'num_workers': 8,
                'shuffle': True,
            }
        }
    }

testing_ingredient = Ingredient('dataloaders')


@testing_ingredient.config
def config():
    """
    DataLoader configurations by split.
    """
    # modify with dataloaders.batchsize
    batch_size = 1

    splits = {
        'test': {
            'class': 'ExamDataLoader',
            'args': {
                'batch_size': batch_size,
                'num_workers': 8,
                'shuffle': True,
            }
        }
    }


def collate(batch_as_list):
    """
    """
    all_X = defaultdict(lambda: defaultdict(list))
    all_y = defaultdict(list)
    all_info = []

    for X, y, info in batch_as_list:
        all_info.append(info)
        # collates inputs in X
        for x in X:
            i = x['src']
            all_X[i]['logits'].append(x['logits'])
            if x['custom']:
                all_X[i]['custom'].append(x['custom'])
        # collates tasks in y
        for t, y_t in y.items():
            all_y[t].append(y_t)

    all_y = {t: stack(y_t, dim=0) for t, y_t in all_y.items()}
    for i, x in all_X.items():
        all_X[i]['logits'] = stack(x['logits'], dim=0)
        if x['custom']:
            all_X[i]['custom'] = stack(x['custom'], dim=0)

    return all_X, all_y, all_info

class ExamBatchSampler(Sampler):

    def __init__(self, batch_size, num_slices, sampler=None,
                 weights=None, num_samples=None, replacement=None, shuffle=None,
                 drop_last=False):
        """
        Creates batches of exams with same number of slices.

        TODO: Implement `drop_last`.
        """
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_slices = num_slices

        self.sampler = sampler
        if self.sampler is not None:
            self.weights = weights if weights is not None else torch.ones(len(num_slices))
        else:
            self.weights = torch.ones(len(num_slices))

        self.replacement = replacement
        self.shuffle = shuffle

        self.drop_last = drop_last

    def __iter__(self):
        """
        """
        if self.sampler is not None:
            samples = torch.multinomial(self.weights, self.num_samples,
                                        replacement=self.replacement)
        else:
            if self.shuffle:
                samples = torch.multinomial(self.weights, self.num_samples,
                                            replacement=False)
            else:
                samples = torch.tensor(range(self.num_samples))
        samples = sorted(samples, key=lambda idx: self.num_slices[idx])

        curr_iter = 0
        batches = []
        while curr_iter < self.num_samples:
            batch = [samples[curr_iter]]
            batch_slices = self.num_slices[samples[curr_iter]]

            offset = self.batch_size
            for i in range(1, self.batch_size):
                if curr_iter + i < self.num_samples and \
                    batch_slices == self.num_slices[samples[curr_iter + i]]:
                    batch.append(samples[curr_iter + i])
                else:
                    offset = i
                    break
            batches.append(batch)
            curr_iter = curr_iter + offset

        batch_idxs = torch.randperm(len(batches)).tolist()
        for batch_idx in batch_idxs:
            yield batches[batch_idx]

    def __len__(self):
        """
        This is approximate because we cannot know number of batches a priori.
        """
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class ExamDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=6,
                 sampler=None,
                 num_samples=1000,
                 replacement=False,
                 weight_task=None,
                 class_probs=None,
                 pin_memory=False):
        """
        """
        if sampler in {"WeightedRandomSampler", "RandomSampler"}:
            # get example weights so examples are sampled according to class_probs
            if sampler == "WeightedRandomSampler":
                classes = []
                weights = self._get_weights(dataset, weight_task)
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                                replacement=replacement)
            elif sampler == "RandomSampler":
                weights = None
                sampler = RandomSampler(data_source=dataset, num_samples=num_samples,
                                        replacement=replacement)
        elif sampler is not None:
            raise ValueError(f"Sampler {sampler} not supported.")
        else:
            num_samples = len(dataset)
            weights = None

        if batch_size > 1:
            num_slices = dataset.get_num_slices()
            batch_sampler = ExamBatchSampler(batch_size, num_slices,
                                             sampler=sampler,
                                             weights=weights,
                                             num_samples=num_samples,
                                             replacement=replacement,
                                             shuffle=shuffle,
                                             drop_last=False)

            super().__init__(dataset=dataset, num_workers=num_workers,
                             batch_sampler=batch_sampler, pin_memory=pin_memory,
                             collate_fn=collate)
        else:
            super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler, pin_memory=pin_memory,
                             collate_fn=collate)

    def _get_weights(self, dataset, weight_task):
        """
        """
        for y in dataset.get_all_y(tasks=[weight_task], hard=True):
            y_class = y[weight_task]
            classes.append(y_class)

        classes = torch.stack(classes)
        if classes.shape[-1] > 1:
            classes = soft_to_hard(classes, break_ties="random").long()

        classes = torch.LongTensor(classes)
        counts = torch.bincount(classes)

        weights = torch.zeros_like(classes, dtype=torch.float)
        for example_idx, class_idx in enumerate(classes):
            class_prob = class_probs[class_idx] / float(counts[class_idx])
            weights[example_idx] = class_prob
        return weights