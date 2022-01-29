
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
            if self.params["use_trigram"]:
                self.seq_trigram_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_trigram_item_desc")
            if self.params["use_subword"]:
                self.seq_subword_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_subword_item_desc")

            # placeholder for length
            self.sequence_length_name = tf.placeholder(tf.int32, shape=[None], name="sequence_length_name")
            self.sequence_length_item_desc = tf.placeholder(tf.int32, shape=[None], name="sequence_length_item_desc")
            self.sequence_length_category_name = tf.placeholder(tf.int32, shape=[None],
                                                                name="sequence_length_category_name")
            self.sequence_length_item_desc_subword = tf.placeholder(tf.int32, shape=[None],
                                                                    name="sequence_length_item_desc_subword")
            self.word_length = tf.placeholder(tf.int32, shape=[None, None], name="word_length")

            # other context
            self.brand_name = tf.placeholder(tf.int32, shape=[None, 1], name="brand_name")
            # self.category_name = tf.placeholder(tf.int32, shape=[None, 1], name="category_name")
            self.category_name1 = tf.placeholder(tf.int32, shape=[None, 1], name="category_name1")
            self.category_name2 = tf.placeholder(tf.int32, shape=[None, 1], name="category_name2")
            self.category_name3 = tf.placeholder(tf.int32, shape=[None, 1], name="category_name3")
            self.item_condition_id = tf.placeholder(tf.int32, shape=[None, 1], name="item_condition_id")
            self.item_condition = tf.placeholder(tf.float32, shape=[None, self.params["MAX_NUM_CONDITIONS"]], name="item_condition")
            self.shipping = tf.placeholder(tf.int32, shape=[None, 1], name="shipping")
            self.num_vars = tf.placeholder(tf.float32, shape=[None, self.params["NUM_VARS_DIM"]], name="num_vars")

            # target
            self.target = tf.placeholder(tf.float32, shape=[None, 1], name="target")

            #### embed
            # embed seq
            # std = np.sqrt(2 / self.params["embedding_dim"])
            std = 0.001
            minval = -std