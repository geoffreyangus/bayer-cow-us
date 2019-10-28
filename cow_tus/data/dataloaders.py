"""
"""
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from ac.util import array_like_stack


def mt_collate_fn(batch_list):
    """ Collate function for a multi-task dataset.
    Assumes all inputs are the same size.
    Args:
        batch_list (list) list of sequences
    """
    all_inputs = []
    all_targets = defaultdict(list)
    all_info = []

    for inputs, targets, info in batch_list:
        all_inputs.append(inputs)
        all_info.append(info)
        for task, target in targets.items():
            if len(target.shape) < 1:
                target = target.unsqueeze(dim=0)
            all_targets[task].append(target)

    # stack targets and inputs
    all_targets = {task: array_like_stack(targets)
                   for task, targets in all_targets.items()}
    all_inputs = array_like_stack(all_inputs)
    return all_inputs, all_targets, all_info


class MTDataLoader(DataLoader):
    """
    """
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
        if sampler in {"RandomSampler", "WeightedRandomSampler"}:
            if sampler == "RandomSampler":
                weights = None
                sampler = RandomSampler(data_source=dataset, num_samples=num_samples,
                                        replacement=replacement)
            elif sampler == "WeightedRandomSampler":
                # get weights so examples are sampled according to class_probs
                classes = []
                for target in dataset.get_targets(tasks=[weight_task], hard=True):
                    target_class = target[weight_task]
                    classes.append(target_class)
                classes = torch.stack(classes)
                if classes.shape[-1] > 1:
                    classes = soft_to_hard(classes, break_ties="random").long()
                classes = torch.LongTensor(classes)
                counts = torch.bincount(classes)
                weights = torch.zeros_like(classes, dtype=torch.float)
                for example_idx, class_idx in enumerate(classes):
                    class_prob = class_probs[class_idx] / float(counts[class_idx])
                    weights[example_idx] = class_prob
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples,
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
                             collate_fn=mt_exam_collate)
        else:
            super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler, pin_memory=pin_memory,
                             collate_fn=mt_exam_collate)