import os
import os.path as path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sacred import Ingredient
from emmental.data import EmmentalDataset

import cow_tus.data.transforms as transforms

class TUSDataset(EmmentalDataset):

    def __init__(self, dataset_dir, split_str, labels_path, transform_fns):
        """
        """
        split_path = path.join(dataset_dir, f'{split_str}.csv')
        self.split_df = pd.read_csv(split_path, index_col=0)
        self.labels_df = pd.read_csv(labels_path, index_col=0, header=[0, 1])
        self.exam_ids = list(self.split_df.index.unique())

        self.transform_fns = transform_fns
        self.shuffle_transform = 'shuffle' in [f['fn'] for f in transform_fns]

        X_dict = {'exam_ids': []}
        Y_dict = {'primary':  []}

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
        x_dict = {i: inputs[idx] for i, inputs in self.X_dict.items() if i != 'exam'}
        x_dict['exam'] = self.get_x(self.exam_ids[idx])
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
            if fn == 'shuffle':
                continue
            loops = getattr(transforms, fn)(loops, **args)
        # loops.copy() because of negative striding
        return torch.tensor(loops.copy(), dtype=torch.float)

    def get_y(self, exam_id):
        """
        TODO: ONLY WORKS WITH BINARY CLASS
        """
        rows = self.labels_df.loc[exam_id]
        y = {}
        rows_target = rows['primary']
        if not isinstance(rows_target, pd.Series):
            soft_target = np.array(rows_target.iloc[0])
            for exam_id, row in rows_target.iterrows():
                assert np.array_equal(soft_target, np.array(row)), \
                    f'exam_id {exam_id} has conflicting targets'
        else:
            soft_target = np.array(rows_target)
        y['primary'] = np.argmax(soft_target)
        return y





