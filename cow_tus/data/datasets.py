import os
import os.path as path
import random
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sacred import Ingredient
from emmental.data import EmmentalDataset

import cow_tus.data.transforms as transforms


logger = logging.getLogger(__name__)


class GlobalDataset(EmmentalDataset):

    def __init__(self, dataset_dir, split_str, labels_path, transform_fns):
        """
        """
        split_path = path.join(dataset_dir, f'{split_str}.csv')
        self.split_df = pd.read_csv(split_path, index_col=0)
        self.split_str = split_str
        self.labels_df = pd.read_csv(labels_path, index_col=0, header=[0, 1])
        self.exam_ids = list(self.split_df.index.unique())

        self.transform_fns = transform_fns
        self.shuffle_transform = 'shuffle' in [f['fn'] for f in transform_fns]

        self.instance_transform = None
        for f in transform_fns:
            # only extract instances if asked to do so and specified for split
            if 'extract_instance' == f['fn'] and split_str in f['args']['splits']:
                self.instance_transform = f['args']
                logger.info(f"using instance extraction on {f['args']['splits']} splits")
                break
        if self.instance_transform != None and self.instance_transform.get('instance_only', False):
            # only access exam_ids with instance level labels
            exam_ids = []
            for exam_id in self.exam_ids:
                rows = self.split_df.loc[exam_id]
                if isinstance(rows, pd.Series):
                    if not np.isnan(rows['label.lv']):
                        exam_ids.append(exam_id)
                else:
                    if not np.isnan(rows.iloc[0]['label.lv']):
                        exam_ids.append(exam_id)
            logger.info(f'using {len(exam_ids)} of {len(self.exam_ids)} exam_ids')
            self.exam_ids = exam_ids
        else:
            logger.info(f'using {len(self.exam_ids)} exam_ids')

        X_dict = {'exam_ids': []}
        Y_dict = {
            'primary':  [],
            'primary_multiclass': [],
            '2normal_binary': []        # labels: control, 1,2 (normal) | 3,4 (abnormal),
        }

        for idx, exam_id in enumerate(self.exam_ids):
            X_dict['exam_ids'].append(exam_id)

            y_dict = self.get_y(exam_id)
            for t, label in y_dict.items():
                Y_dict[t].append(label)

        Y_dict = {k: torch.from_numpy(np.array(v)) for k, v in Y_dict.items()}
        EmmentalDataset.__init__(self, 'cow-tus-dataset', X_dict=X_dict, Y_dict=Y_dict)

    def __getitem__(self, idx):
        """
        """
        x_dict = {i: inputs[idx] for i, inputs in self.X_dict.items() if i != 'clip'}
        x_dict['clip'] = self.get_x(self.exam_ids[idx])
        y_dict = {t: labels[idx] for t, labels in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        """
        """
        return len(self.exam_ids)

    def get_x(self, exam_id):
        """
        """
        rows = self.split_df.loc[exam_id]

        if isinstance(rows, pd.Series):
            loop_paths = [rows['exdir.loop_data_path']]
        else:
            # adds instance level samples
            if self.instance_transform != None and not np.isnan(rows.iloc[0]['label.lv']):
                loop_paths = []
                loop_added = False
                for exam_id, row in rows.iterrows():
                    loop_type = row['exdir.loop_type']
                    if loop_type == 'malformed':
                        continue

                    # add loop_path if the loop_type label matches the global label
                    loop_path = row['exdir.loop_data_path']
                    if row[f'label.{loop_type}'] == float(row['label.global_multiclass_label']) and not loop_added:
                        loop_paths.append(loop_path)
                        loop_added = True
                    elif row[f'label.{loop_type}'] == float(row['label.global_multiclass_label']) and \
                         random.random() < self.instance_transform['p_add_same_class']:
                        loop_paths.append(loop_path)
                    elif random.random() < self.instance_transform['p_add_diff_class']:
                        loop_paths.append(loop_path)
            else:
                loop_paths = list(rows['exdir.loop_data_path'])

        loops = []
        for loop_path in loop_paths:
            loop = np.load(f'{"/data4" + loop_path[5:]}')
            loops.append(loop)
        if self.shuffle_transform:
            random.shuffle(loops)
        loops = np.concatenate(loops)
        loops = np.expand_dims(loops, axis=3)

        for transform_fn in self.transform_fns:
            fn = transform_fn['fn']
            args = transform_fn['args']
            if fn in {'shuffle', 'extract_instance'}:
                continue
            loops = getattr(transforms, fn)(loops, **args)
        # loops.copy() because of negative striding
        return torch.tensor(loops.copy(), dtype=torch.float)

    def get_y(self, exam_id):
        """
        """
        rows = self.labels_df.loc[exam_id]
        y = {}
        for key in ['primary', 'primary_multiclass']:
            rows_target = rows[key]
            if not isinstance(rows_target, pd.Series):
                soft_target = np.array(rows_target.iloc[0])
                for exam_id, row in rows_target.iterrows():
                    assert np.array_equal(soft_target, np.array(row)), \
                        f'exam_id {exam_id} has conflicting targets'
            else:
                soft_target = np.array(rows_target)
            y[key] = np.argmax(soft_target)
        y['2normal_binary'] = 0 if y['primary_multiclass'] <= 1 else 1
        print(y)
        return y

class InstanceDataset(EmmentalDataset):

    def __init__(self, dataset_dir, split_str, labels_path, transform_fns):
        """
        """
        split_path = path.join(dataset_dir, f'{split_str}.csv')
        self.split_df = pd.read_csv(split_path, index_col=0)
        self.labels_df = pd.read_csv(labels_path, index_col=0, header=[0, 1])
        self.loop_idxs = range(len(self.split_df))

        # use df.iloc because loop_ids are not unique...
        loop_idxs = []
        for loop_idx in self.loop_idxs:
            row = self.split_df.iloc[loop_idx]
            loop_type = row['exdir.loop_type']
            if f'label.{loop_type}' in row.keys() and not np.isnan(row[f'label.{loop_type}']):
                loop_idxs.append(loop_idx)
        logger.info(f'using {len(loop_idxs)} of {len(self.loop_idxs)} loop_idxs')
        self.loop_idxs = loop_idxs

        self.transform_fns = transform_fns

        X_dict = {'loop_idxs': []}
        Y_dict = {
            'primary':  [],
            'primary_multiclass': [],
            '2normal_binary': []        # labels: control, 1,2 (normal) | 3,4 (abnormal),
        }

        for idx, loop_idx in enumerate(self.loop_idxs):
            X_dict['loop_idxs'].append(loop_idx)

            y_dict = self.get_y(loop_idx)
            for t, label in y_dict.items():
                Y_dict[t].append(label)

        Y_dict = {k: torch.from_numpy(np.array(v)) for k, v in Y_dict.items()}
        EmmentalDataset.__init__(self, 'cow-tus-dataset', X_dict=X_dict, Y_dict=Y_dict)

    def __getitem__(self, idx):
        """
        """
        x_dict = {i: inputs[idx] for i, inputs in self.X_dict.items() if i != 'clip'}
        x_dict['clip'] = self.get_x(self.loop_idxs[idx])
        y_dict = {t: labels[idx] for t, labels in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        """
        """
        return len(self.loop_idxs)

    def get_x(self, loop_idx):
        """
        """
        rows = self.split_df.iloc[loop_idx]

        loop_path = rows['exdir.loop_data_path']
        loop = np.load(f'{"/data4" + loop_path[5:]}')
        for transform_fn in self.transform_fns:
            fn = transform_fn['fn']
            args = transform_fn['args']
            loop = getattr(transforms, fn)(loop, **args)
        # loops.copy() because of negative striding
        return torch.tensor(loop.copy(), dtype=torch.float)

    def get_y(self, loop_idx):
        """
        """
        loop_attrs = self.split_df.iloc[loop_idx]
        joint_id = loop_attrs.name + '_' + loop_attrs['exdir.loop_id']

        row = self.labels_df.loc[joint_id]
        y = {}
        for key in ['primary', 'primary_multiclass']:
            row_target = row[key]
            soft_target = np.array(row_target)
            y[key] = np.argmax(soft_target)
        y['2normal_binary'] = 0 if y['primary_multiclass'] <= 1 else 1
        # print(y, joint_id)
        return y