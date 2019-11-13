"""
"""
import os
import os.path as path
import pickle

import torch
import torch.nn.functional as F
import numpy as np


def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def ce_loss(task_name, immediate_output, Y, active):
    """
    CrossEntropyLoss function to be used with Emmental module.
    """
    return F.cross_entropy(
        immediate_output[f"decoder_module_{task_name}"][0], Y.view(-1)
    )


def output(task_name, immediate_output):
    """
    Softmax function to be used with Emmental module.
    """
    return F.softmax(immediate_output[f"decoder_module_{task_name}"][0], dim=1)


def require_dir(dir_str):
    """
    """
    if not(path.exists(dir_str)):
        require_dir(path.dirname(dir_str))
        os.mkdir(dir_str)

def stack(items, dim=0):
    """
    Joins list of items in a new axis.
    """
    if len(items) < 1:
        raise ValueError("items is empty")

    if len(set([type(item) for item in items])) != 1:
        raise TypeError("items are not of the same type")

    if isinstance(items[0], list):
        return items

    elif isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0)

    elif isinstance(items[0], np.ndarray):
        return np.stack(items, axis=0)

    else:
        raise TypeError(f"Unrecognized type f{type(items[0])}")

def concat(items, dim=0):
    """
    Concatenates the items in list items. All elements in items must be of the same type.
    """
    if len(items) < 1:
        raise ValueError("items is empty")

    if len(set([type(item) for item in items])) != 1:
        raise TypeError("items are not of the same type")

    if isinstance(items[0], list):
        return sum(items, [])

    elif isinstance(items[0], torch.Tensor):
        # zero-dimensional tensors cannot be concatenated
        items = [item.expand(1) if not item.shape else item for item in items]
        return torch.cat(items, dim=0)

    else:
        raise TypeError(f"Unrecognized type f{type(items[0])}")

def hard_to_soft(Y_h, k):
    """Converts a 1D tensor of hard labels into a 2D tensor of soft labels
    Source: MeTaL from HazyResearch, https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    Args:
        Y_h: an [n], or [n,1] tensor of hard (int) labels in {1,...,k}
        k: the largest possible label in Y_h
    Returns:
        Y_s: a torch.FloatTensor of shape [n, k] where Y_s[i, j-1] is the soft
            label for item i and label j
    """
    Y_h = Y_h.clone()
    if Y_h.dim() > 1:
        Y_h = Y_h.squeeze()
    assert Y_h.dim() == 1
    assert (Y_h >= 0).all()
    assert (Y_h < k).all()
    n = Y_h.shape[0]
    Y_s = torch.zeros((n, k), dtype=Y_h.dtype, device=Y_h.device)
    for i, j in enumerate(Y_h):
        Y_s[i, int(j)] = 1.0
    return Y_s


def soft_to_hard(Y_s, break_ties="random"):
    """Break ties in each row of a tensor according to the specified policy
    Source: MeTaL from HazyResearch, https://github.com/HazyResearch/metal/
    Modified slightly to accommodate PyTorch tensors.
    Args:
        Y_s: An [n, k] np.ndarray of probabilities
        break_ties: A tie-breaking policy:
            "abstain": return an abstain vote (0)
            "random": randomly choose among the tied options
                NOTE: if break_ties="random", repeated runs may have
                slightly different results due to difference in broken ties
            [int]: ties will be broken by using this label
    """
    n, k = Y_s.shape
    maxes, argmaxes = Y_s.max(dim=1)
    diffs = torch.abs(Y_s - maxes.reshape(-1, 1))

    TOL = 1e-5
    Y_h = torch.zeros(n, dtype=torch.int64)
    for i in range(n):
        max_idxs = torch.where(diffs[i, :] < TOL, Y_s[i], torch.tensor(0.0, dtype=Y_s.dtype))
        max_idxs = torch.nonzero(max_idxs).reshape(-1)
        if len(max_idxs) == 1:
            Y_h[i] = max_idxs[0]
        # Deal with "tie votes" according to the specified policy
        elif break_ties == "random":
            Y_h[i] = torch.as_tensor(np.random.choice(max_idxs))
        elif break_ties == "abstain":
            Y_h[i] = 0
        elif isinstance(break_ties, int):
            Y_h[i] = break_ties
        else:
            ValueError(f"break_ties={break_ties} policy not recognized.")
    return Y_h


def place_on_gpu(data, device=0):
    """
    Recursively places all 'torch.Tensor's in data on gpu and detaches.
    If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i], device) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_gpu(val, device) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def place_on_cpu(data):
    """
    Recursively places all 'torch.Tensor's in data on cpu and detaches from computation
    graph. If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_cpu(data[i]) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_cpu(val) for key,val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data