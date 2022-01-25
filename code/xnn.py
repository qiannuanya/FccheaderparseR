
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
        self.logger = logger
        _makedirs(self.params["model_dir"], force=True)
        self._init_graph()
        self.gvars_state_list = []

        # 14
        self.bias = 0.01228477
        self.weights = [
            0.00599607, 0.02999416, 0.05985384, 0.20137787, 0.03178938, 0.04612812,
            0.05384821, 0.10121514, 0.05915169, 0.05521121, 0.06448063, 0.0944233,
            0.08306157, 0.11769992
        ]
        self.weights = np.array(self.weights).reshape(-1, 1)

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.params["random_seed"])

            #### input
            self.training = tf.placeholder(tf.bool, shape=[], name="training")
            # seq
            self.seq_name = tf.placeholder(tf.int32, shape=[None, None], name="seq_name")
            self.seq_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_item_desc")
            self.seq_category_name = tf.placeholder(tf.int32, shape=[None, None], name="seq_category_name")
            if self.params["use_bigram"]:
                self.seq_bigram_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_bigram_item_desc")