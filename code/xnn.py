
import time
import numpy as np
import tensorflow as tf

from lr_schedule import _cosine_decay_restarts, _exponential_decay
from metrics import rmse
from nn_module import embed, encode, attend
from nn_module import word_dropout
from nn_module import dense_block, resnet_block
from optimizer import LazyPowerSignOptimizer, LazyAddSignOptimizer, LazyAMSGradOptimizer, LazyNadamOptimizer
from utils import _makedirs


class XNN(object):
    def __init__(self, params, target_scaler, logger):
        self.params = params
        self.target_scaler = target_scaler