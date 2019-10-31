"""
"""
from collections import defaultdict

# import torch
# import numpy as np
# import pandas as pd
import sklearn.metrics as skl
from sacred import Ingredient

# from cow_tus.util import place_on_cpu, get_batch_size, array_like_concat


metrics_ingredient = Ingredient('metrics')


@metrics_ingredient.config
def config():
    """
    """
    # maps task types to appropriate metrics
    type_to_fns = {
        'binary': [
            {'fn': 'accuracy'},
            {'fn': 'roc_auc'},
            {'fn': 'precision'},
            {'fn': 'recall'},
            {'fn': 'f1'}
        ]
    }


@metrics_ingredient.config
def config_nva(type_to_fns):
    """
    Metrics configuration for normal vs. abnormal tasks in multiclass settings.
    """
    # use sklearn.metrics.multilabel_confusion_matrix for further analysis
    abnormal_labels = [1, 2, 3]
    type_to_fns['multi_nva'] = [
        {'fn': 'accuracy'},
        {'fn': 'normal_vs_abnormal_accuracy', 'args': {'abnormal_labels': abnormal_labels}},
        {'fn': 'normal_vs_abnormal_roc_auc', 'args': {'abnormal_labels': abnormal_labels}},
        {'fn': 'normal_vs_abnormal_precision', 'args': {'abnormal_labels': abnormal_labels}},
        {'fn': 'normal_vs_abnormal_recall', 'args': {'abnormal_labels': abnormal_labels}},
        {'fn': 'normal_vs_abnormal_f1', 'args': {'abnormal_labels': abnormal_labels}}
    ]

def accuracy(y_true, probas):
    """
    """
    return skl.accuracy_score(y_true, probas)


def roc_auc(y_true, probas):
    """
    """
    return skl.roc_auc_score(y_true, probas)


def precision(y_true, probas):
    """
    """
    return skl.precision_score(y_true, probas)


def recall(y_true, probas):
    """
    """
    return skl.recall_score(y_true, probas)


def f1(y_true, probas):
    """
    """
    return skl.f1_score(y_true, probas)


def normal_vs_abnormal_accuracy(y_true, probas, abnormal_labels=None):
    """
    """
    assert abnormal_labels != None, 'please specify abnormal labels'
    # TODO: bucket abnormal labels for binary classification
    return skl.accuracy_score(y_true, probas)


def normal_vs_abnormal_roc_auc(y_true, probas, abnormal_labels=None):
    """
    """
    assert abnormal_labels != None, 'please specify abnormal labels'
    # TODO: bucket abnormal labels for binary classification
    return skl.roc_auc_score(y_true, probas)


def normal_vs_abnormal_precision(y_true, probas, abnormal_labels=None):
    """
    """
    assert abnormal_labels != None, 'please specify abnormal labels'
    # TODO: bucket abnormal labels for binary classification
    return skl.precision_score(y_true, probas)


def normal_vs_abnormal_recall(y_true, probas, abnormal_labels=None):
    """
    """
    assert abnormal_labels != None, 'please specify abnormal labels'
    # TODO: bucket abnormal labels for binary classification
    return skl.recall_score(y_true, probas)


def normal_vs_abnormal_f1(y_true, probas, abnormal_labels=None):
    """
    """
    assert abnormal_labels != None, 'please specify abnormal labels'
    # TODO: bucket abnormal labels for binary classification
    return skl.f1_score(y_true, probas)






