"""
"""
import os
import os.path as path
from collections import Counter, defaultdict, deque
import random

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver
import yaml

import cow_tus.util.util as util


EXPERIMENT_NAME = 'splitter'
ex = Experiment(EXPERIMENT_NAME)


@ex.config
def config():
    """
    """
    data_dir = 'data/single-instance-learning/temporal-downsample'

    hypothesis_conditions = ['by-animal-number', 'hold-out-validation']
    group_dir = path.join('data', 'split', *hypothesis_conditions)

    strata_key = 'raw.animal_number'
    split_to_count = {
        'train': 0, # will be replaced in the next line
        'valid': 15,
        'test': 15
    }
    split_to_count['train'] = len(pd.read_csv(path.join(data_dir, 'attrs.csv'), index_col=0).index.unique()) - sum(split_to_count.values())
    train_split = 'train'


class DataSplitter:

    def __init__(self):
        """
        """
        self.metadata = {}

    @ex.capture
    def run(self, group_dir, data_dir, hypothesis_conditions, split_to_count, strata_key):
        """
        """
        self.metadata.update({
            'meta.data_dir': data_dir,
            'meta.split_to_count': dict(split_to_count),
            'meta.strata_key': strata_key
        })

        attrs_path = path.join(data_dir, 'attrs.csv')
        attrs_df = pd.read_csv(attrs_path, index_col=0)

        split_to_quota = self._analyze(attrs_df)
        animal_id_to_exams = self._shuffle(attrs_df)
        split_to_exam_ids = self._assign(animal_id_to_exams, split_to_quota)

        split_to_split_df = self._format(attrs_df, split_to_exam_ids)
        for split, split_df in split_to_split_df.items():
            split_df_path  = path.join(group_dir, f'{split}.csv')
            split_df.to_csv(split_df_path)
            ex.add_artifact(split_df_path)

        metadata_path = path.join(group_dir, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            f.write(yaml.dump(self.metadata))
        ex.add_artifact(metadata_path)

    @ex.capture
    def _analyze(self, attrs_df, split_to_count):
        """
        """
        labels = []
        exam_groups = attrs_df.groupby('exdir.exam_id')
        for exam_id, exam_group in exam_groups:
            labels.append(list(set(exam_group['preprocessed.global_label_multiclass']))[0])
        label_freqs = sorted(Counter(labels).items())

        split_to_quota = defaultdict(dict)
        for label, freq in label_freqs:
            for split, count in split_to_count.items():
                # minimum number of each class per split (int(x) truncates x)
                split_to_quota[split][label] = int((count / len(exam_groups)) * freq)
        split_to_quota = dict(split_to_quota)

        self.metadata.update({
            'split.label_counts': dict(Counter(labels)),
            'split.label_split_quotas': split_to_quota
        })

        return split_to_quota

    @ex.capture
    def _shuffle(self, attrs_df, strata_key, _log):
        """
        """
        animal_groups = attrs_df.groupby(strata_key)
        animal_id_to_exams = defaultdict(list)
        for animal_id, animal_group in animal_groups:
            exam_groups = animal_group.groupby('exdir.exam_id')
            for exam_id, exam_group in exam_groups:
                label = exam_group.loc[exam_id]['preprocessed.global_label_multiclass']
                try:
                    label = list(set(label))[0]
                except:
                    _log.warning(f'exam {exam_id} only has one instance')
                animal_id_to_exams[animal_id].append({
                    'exam_id': exam_id,
                    'label': label
                })

        animal_id_to_exams = list(animal_id_to_exams.items())
        return animal_id_to_exams

    @ex.capture
    def _assign(self, animal_id_to_exams, split_to_quota, split_to_count, train_split, _log):
        """
        Caution: splitter relies on randomness to hit quotas and will generate
        new splits until
        """
        while True:
            result = self._get_candidate(animal_id_to_exams, split_to_quota)
            if result == None:
                continue
            split_to_label_freqs, split_to_exam_ids = result
            _log.info(f'verifying candidate with label freqs: \n{split_to_label_freqs}')
            if self._verify_candidate(split_to_label_freqs, split_to_quota):
                _log.info(f'verification success with quota: \n{split_to_quota}')
                break

        return split_to_exam_ids

    @ex.capture
    def _get_candidate(self, animal_id_to_exams, split_to_quota, split_to_count):
        """
        """
        random.shuffle(animal_id_to_exams)
        split_to_exam_ids = defaultdict(list)
        split_to_label_freqs = {split: {class_type: 0
                                        for class_type in split_to_quota[split]}
                                for split in split_to_quota}
        i = 0
        for split, count in split_to_count.items():
            added = 0
            while added < count:
                if i == len(animal_id_to_exams):
                    return None
                animal_id, exams = animal_id_to_exams[i]
                for exam in exams:
                    exam_id = exam['exam_id']
                    label = exam['label']
                    split_to_exam_ids[split].append(exam_id)
                    split_to_label_freqs[split][label] += 1
                added += len(exams)
                i += 1
        return split_to_label_freqs, split_to_exam_ids

    @ex.capture
    def _verify_candidate(self, split_to_label_freqs, split_to_quota, train_split):
        """
        """
        for split, quota in split_to_quota.items():
            if split == train_split:
                continue
            for class_type, quota_count in quota.items():
                if split_to_label_freqs[split][class_type] < quota_count:
                    return False
        return True

    @ex.capture
    def _format(self, attrs_df, split_to_exam_ids):
        """
        """
        split_to_split_df = {}
        split_to_labels = defaultdict(list)
        for split, exam_ids in split_to_exam_ids.items():
            split_df = attrs_df.loc[exam_ids]
            split_to_split_df[split] = split_df

            exam_groups = split_df.groupby('exdir.exam_id')
            for exam_id, exam_group in exam_groups:
                split_to_labels[split].append(list(set(exam_group['preprocessed.global_label_multiclass']))[0])

        split_to_counts = defaultdict(dict)
        for split, labels in split_to_labels.items():
            counts = sorted(Counter(labels).items())
            for label, count in counts:
                split_to_counts[split][label] = count

        self.metadata.update({
            'split.label_to_freq': dict(split_to_counts)
        })
        return split_to_split_df


@ex.config_hook
def hook(config, command_name, logger):
    if config['group_dir'] == None:
        raise Exception(f'group_dir is {config["group_dir"]}')
    else:
        util.require_dir(config['group_dir'])
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
