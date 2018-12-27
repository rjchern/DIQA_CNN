from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from scipy import stats
import os, yaml

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric

class DIQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self.label_pred = []
        self.label      = []

    def update(self, output):
        y_pred, y = output
        self.label.append(y)
        self.label_pred.append(torch.mean(y_pred))

    def compute(self):
        sq_std = np.reshape(np.asarray(self.label), (-1,))
        sq_pred = np.reshape(np.asarray(self.label_pred), (-1,))
        srocc = stats.spearmanr(sq_std, sq_pred)[0]
        plcc = stats.pearsonr(sq_std, sq_pred)[0]
        #plcc = stats.pearsonr(sq, q)[0]
        return srocc, plcc