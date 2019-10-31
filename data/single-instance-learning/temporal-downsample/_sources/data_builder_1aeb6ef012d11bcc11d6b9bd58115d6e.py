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

EXPERIMENT_NAME = 'builder'
ex = Experiment(EXPERIMENT_NAME, ingredients=[builder_ingredient])


@ex.config
def config():
    """
    """
    raw_dir = "/data/cow-tus-data/raw"
    out_dir = "/data/cow-tus-data/processed"

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
    def run(self, _log, group_dir, raw_dir, out_dir, hypothesis_conditions, preprocessing):
        """
        """
        root_group = self._init_exdir()
        raw_attrs_df = pd.read_csv(path.join(raw_dir, 'labels.csv'))

        self.metadata.update({
            'binary.num_normals': 0,
            'binary.num_abnormals': 0,
            'multiclass.num_0': 0,
            'multiclass.num_1': 0,
            'multiclass.num_2': 0,
            'multiclass.num_3': 0,
            'meta.num_loops': 0,
            'meta.num_exams': 0,
            'meta.num_exams_skipped': 0,
            'meta.num_files_skipped': 0
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
                self.metadata['meta.num_exams_skipped'] += 1
                continue

            # determine its score
            tus_score = attrs['raw.tus_score']
            if tus_score == 'control':
                tus_score = '1'
            tus_score = re.sub(r"[^0-9]", "", tus_score)

            global_label_binary = 0 if int(tus_score) == 1 else 1
            global_label_multiclass = int(tus_score) - 1
            if global_label_binary == 0:
                self.metadata['binary.num_normals'] += 1
                self.metadata['multiclass.num_0'] += 1
            else:
                self.metadata['binary.num_abnormals'] += 1
                if global_label_multiclass == 1:
                    self.metadata['multiclass.num_1'] += 1
                elif global_label_multiclass == 2:
                    self.metadata['multiclass.num_2'] += 1
                elif global_label_multiclass == 3:
                    self.metadata['multiclass.num_3'] += 1

            # create a group for its instances
            exam_group = root_group.require_group(str(exam_id))
            exam_group.attrs.update(attrs)
            exam_group.attrs.update({
                'raw.exam_path': raw_exam_path,
                'exdir.exam_id': exam_id,
                'exdir.exam_path': str(exam_group.directory),
                'preprocessed.global_label_binary': global_label_binary,
                'preprocessed.global_label_multiclass': global_label_multiclass
            })

            concat = []
            for loop_filename in os.listdir(raw_exam_path):
                if loop_filename.rfind('.AVI') == -1:
                    _log.warning(f'{loop_filename} is not an AVI file. Continuing.')
                    self.metadata['meta.num_files_skipped'] += 1

                raw_loop_path = path.join(raw_exam_path, loop_filename)
                raw_loop = skvideo.io.vread(raw_loop_path)
                raw_loop_shape = raw_loop.shape

                loop = raw_loop
                for preprocess_fn in preprocessing['preprocess_fns']:
                    fn = preprocess_fn['fn']
                    args = preprocess_fn['args']
                    loop = getattr(preprocess, fn)(loop, **args)
                loop_shape = loop.shape

                loop_id = loop_filename[:loop_filename.rfind('.AVI')].replace(" ", "")
                loop_dataset = exam_group.require_dataset(loop_id, data=loop)

                loop_dataset.attrs.update({
                    'raw.loop_path': raw_loop_path,
                    'raw.loop_shape': raw_loop_shape,
                    'exdir.loop_id': loop_id,
                    'exdir.loop_path': str(loop_dataset.directory),
                    'exdir.loop_data_path': path.join(loop_dataset.directory, 'data.npy'),
                    'exdir.loop_shape': loop_shape
                })
                out_attrs_entry = {}
                out_attrs_entry.update(exam_group.attrs)
                out_attrs_entry.update(loop_dataset.attrs)

                out_attrs_data.append(out_attrs_entry)


        out_attrs_path = path.join(group_dir, 'attrs.csv')
        out_attrs_df = pd.DataFrame(out_attrs_data)
        out_attrs_df.to_csv(out_attrs_path, index=False)
        ex.add_artifact(out_attrs_path)

        metadata_path = path.join(group_dir, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            f.write(yaml.dump(self.metadata))
        ex.add_artifact(metadata_path)




@ex.config_hook
def hook(config, command_name, logger):
    if config['group_dir'] == None:
        raise Exception(f'group_dir is {config["group_dir"]}')
    ex.observers.append(FileStorageObserver(config['group_dir']))


@ex.main
def main(_run):
    """
    """
    builder = DataBuilder()
    results = builder.run()
    return results


if __name__ == '__main__':
    ex.run_commandline()
