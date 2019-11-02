"""
"""
import os
import os.path as path
import re

import exdir
import numpy as np
import pandas as pd
from tqdm import tqdm
import skvideo.io
from sacred import Experiment
from sacred.observers import FileStorageObserver
import yaml

from cow_tus.data.transforms.preprocessing import builder_ingredient
import cow_tus.data.transforms.preprocessing as preprocess
import cow_tus.util.util as util

EXPERIMENT_NAME = 'builder'
ex = Experiment(EXPERIMENT_NAME, ingredients=[builder_ingredient])


@ex.config
def config():
    """
    """
    raw_dir = "/data/cow-tus-data/raw"
    raw_labels_filename = "labels_instance_level.csv"
    out_dir = "/data/cow-tus-data/processed"
    # out_dir = 'sample'

    loop_id_substitutions = {
        '4236R44': '4236R4',
        '4042L':   '4042LV',
        '4307L':   '4307L5',
        '4521VRV': '4521RV',
        '44887RV': '4487RV',
        '44787L5': '4487L5',
        '487L6':   '4487L6',
        '487L77':  '4487L7'
    }
    loop_types = ['l5', 'l6', 'l7', 'lv', 'r4', 'r5', 'r6', 'rv']

    hypothesis_conditions = ['single-instance-learning', 'temporal-downsample']
    group_dir = path.join('data', *hypothesis_conditions)


class DataBuilder:

    def __init__(self):
        """
        """
        self.metadata = {}

    @ex.capture
    def _init_exdir(self, raw_dir, out_dir, hypothesis_conditions):
        """
        """
        f = exdir.File(out_dir)
        g = f.require_group(hypothesis_conditions[0])
        for hypothesis_condition in hypothesis_conditions[1:]:
            g = g.require_group(hypothesis_condition)

        self.metadata.update({
            'config.raw_dir': raw_dir,
            'config.out_dir': out_dir,
            'config.hypothesis_conditions': list(hypothesis_conditions),
            'meta.exdir_directory': str(g.directory)
        })
        return g

    @ex.capture
    def run(self, _log, group_dir, raw_dir, raw_labels_filename, out_dir,
            loop_types, loop_id_substitutions, hypothesis_conditions, preprocessing):
        """
        """
        root_group = self._init_exdir()
        raw_attrs_df = pd.read_csv(path.join(raw_dir, raw_labels_filename))

        self.metadata.update({
            'meta.total.global.binary.num_0': 0,
            'meta.total.global.binary.num_1': 0,
            'meta.total.global.multiclass.num_0': 0,
            'meta.total.global.multiclass.num_1': 0,
            'meta.total.global.multiclass.num_2': 0,
            'meta.total.global.multiclass.num_3': 0,
            'meta.total.instance.binary.num_0': 0,
            'meta.total.instance.binary.num_1': 0,
            'meta.total.instance.multiclass.num_0': 0,
            'meta.total.instance.multiclass.num_1': 0,
            'meta.total.instance.multiclass.num_2': 0,
            'meta.total.instance.multiclass.num_3': 0,
            'meta.total.num_loops': 0,
            'meta.total.num_loops_labeled': 0,
            'meta.total.num_loops_inferred_label': 0,
            'meta.total.num_exams': 0,
            'meta.total.num_exams_skipped': 0,
            'meta.total.num_files_skipped': 0,
        })

        out_attrs_data = []
        for i, row in tqdm(raw_attrs_df.iterrows(), total=len(raw_attrs_df)):
            raw_attrs = dict(row)
            attrs = {f'raw.{k}': v for k, v in raw_attrs.items()}

            # find the exam
            exam_id = attrs['raw.id']
            raw_exam_path = os.path.join(raw_dir, 'exams', exam_id)
            if not os.path.isdir(raw_exam_path):
                raw_exam_path = os.path.join(raw_dir, 'exams', exam_id + '-0')
            if not os.path.isdir(raw_exam_path):
                _log.warning(f'{raw_exam_path} not found. Continuing.')
                self.metadata['meta.total.num_exams_skipped'] += 1
                continue
            if len(os.listdir(raw_exam_path)) == 0:
                _log.warning(f'{raw_exam_path} contains no files. Continuing.')
                self.metadata['meta.total.num_exams_skipped'] += 1
                continue
            self.metadata['meta.total.num_exams'] += 1

            # determine its score
            tus_score = attrs['raw.tus_score']
            if tus_score == 'control':
                tus_score = '1'
            tus_score = re.sub(r"[^0-9]", "", tus_score)

            global_binary_label = 0 if int(tus_score) == 1 else 1
            global_multiclass_label = int(tus_score) - 1
            self.metadata[f'meta.total.global.binary.num_{global_binary_label}'] += 1
            self.metadata[f'meta.total.global.multiclass.num_{global_multiclass_label}'] += 1

            # create a group for its instances
            exam_group = root_group.require_group(str(exam_id))
            exam_group.attrs.update(attrs)
            exam_group.attrs.update({
                'raw.exam_path': raw_exam_path,
                'exdir.exam_id': exam_id,
                'exdir.exam_path': str(exam_group.directory),
            })
            exam_group.attrs.update({
                'label.global_binary_label': global_binary_label,
                'label.global_multiclass_label': global_multiclass_label,
            })
            loop_labels = {}
            for loop_type in loop_types:
                if np.isnan(attrs[f'raw.{loop_type}']):
                    if global_binary_label == 0:
                        loop_labels[f'label.{loop_type}'] = 0
                        self.metadata['meta.total.num_loops_inferred_label'] += 1
                    else:
                        loop_labels[f'label.{loop_type}'] = np.nan
                else:
                    loop_labels[f'label.{loop_type}'] = int(attrs[f'raw.{loop_type}']) - 1
            exam_group.attrs.update(loop_labels)

            concat = []
            for loop_filename in os.listdir(raw_exam_path):
                if loop_filename.rfind('.AVI') == -1:
                    _log.warning(f'{loop_filename} is not an AVI file. Continuing.')
                    self.metadata['meta.total.num_files_skipped'] += 1
                    continue
                self.metadata['meta.total.num_loops'] += 1

                loop_id = loop_filename[:loop_filename.rfind('.AVI')].replace(" ", "")
                if loop_id in loop_id_substitutions:
                    loop_id = loop_id_substitutions[loop_id]
                loop_type = loop_id[-2:].lower()

                instance_binary_label = np.nan
                instance_multiclass_label = np.nan
                if loop_type in loop_types:
                    instance_label = exam_group.attrs[f'label.{loop_type}']
                    if not np.isnan(instance_label):
                        instance_binary_label = 0 if int(instance_label) == 0 else 1
                        instance_multiclass_label = int(instance_label)

                        self.metadata[f'meta.total.num_loops_labeled'] += 1
                        self.metadata[f'meta.total.instance.binary.num_{instance_binary_label}'] += 1
                        self.metadata[f'meta.total.instance.multiclass.num_{instance_multiclass_label}'] += 1

                out_attrs_entry = {}
                if loop_id in exam_group:
                    _log.info(f'{loop_id} exists in {str(exam_group.directory)}. Continuing.')
                    out_attrs_entry.update(exam_group.attrs)
                    out_attrs_entry.update(exam_group[loop_id].attrs)
                else:
                    raw_loop_path = path.join(raw_exam_path, loop_filename)
                    _log.info(f'reading {raw_loop_path}...')
                    raw_loop = skvideo.io.vread(raw_loop_path)
                    raw_loop_shape = raw_loop.shape

                    loop = raw_loop
                    for preprocess_fn in preprocessing['preprocess_fns']:
                        fn = preprocess_fn['fn']
                        args = preprocess_fn['args']
                        loop = getattr(preprocess, fn)(loop, **args)
                    loop_shape = loop.shape
                    loop_dataset = exam_group.require_dataset(loop_id, data=loop)

                    loop_dataset.attrs.update({
                        'raw.loop_path': raw_loop_path,
                        'raw.loop_shape': raw_loop_shape,
                        'exdir.loop_id': loop_id,
                        'exdir.loop_type': loop_type if loop_type in loop_types else 'malformed',
                        'exdir.loop_path': str(loop_dataset.directory),
                        'exdir.loop_data_path': path.join(loop_dataset.directory, 'data.npy'),
                        'exdir.loop_shape': loop_shape,
                        'label.instance_binary_label': instance_binary_label,
                        'label.instance_multiclass_label': instance_multiclass_label
                    })
                    out_attrs_entry.update(exam_group.attrs)
                    out_attrs_entry.update(loop_dataset.attrs)
                out_attrs_data.append(out_attrs_entry)

        out_attrs_path = path.join(group_dir, 'attrs.csv')
        out_attrs_df = pd.DataFrame(out_attrs_data)
        out_attrs_df = out_attrs_df.set_index('exdir.exam_id')
        out_attrs_df.to_csv(out_attrs_path)
        ex.add_artifact(out_attrs_path)

        metadata_path = path.join(group_dir, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            f.write(yaml.dump(self.metadata))
        ex.add_artifact(metadata_path)


@ex.config_hook
def hook(config, command_name, logger):
    """
    """
    if config['group_dir'] == None:
        raise Exception(f'group_dir is {config["group_dir"]}')
    else:
        util.require_dir(config['group_dir'])


@ex.main
def main(_run):
    """
    """
    builder = DataBuilder()
    results = builder.run()
    return results


if __name__ == '__main__':
    ex.run_commandline()
