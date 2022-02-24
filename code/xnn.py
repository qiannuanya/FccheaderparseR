
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
            maxval = std
            emb_word = tf.Variable(
                tf.random_uniform([self.params["MAX_NUM_WORDS"] + 1, self.params["embedding_dim"]], minval, maxval,
                                  seed=self.params["random_seed"],
                                  dtype=tf.float32))
            # emb_word2 = tf.Variable(tf.random_uniform([self.params["MAX_NUM_WORDS"] + 1, self.params["embedding_dim"]], minval, maxval,
            #                                     seed=self.params["random_seed"],
            #                                     dtype=tf.float32))
            emb_seq_name = tf.nn.embedding_lookup(emb_word, self.seq_name)
            if self.params["embedding_dropout"] > 0.:
                emb_seq_name = word_dropout(emb_seq_name, training=self.training,
                                            dropout=self.params["embedding_dropout"],
                                            seed=self.params["random_seed"])
            emb_seq_item_desc = tf.nn.embedding_lookup(emb_word, self.seq_item_desc)
            if self.params["embedding_dropout"] > 0.:
                emb_seq_item_desc = word_dropout(emb_seq_item_desc, training=self.training,
                                                 dropout=self.params["embedding_dropout"],
                                                 seed=self.params["random_seed"])
            # emb_seq_category_name = tf.nn.embedding_lookup(emb_word, self.seq_category_name)
            # if self.params["embedding_dropout"] > 0.:
            #     emb_seq_category_name = word_dropout(emb_seq_category_name, training=self.training,
            #                                      dropout=self.params["embedding_dropout"],
            #                                      seed=self.params["random_seed"])
            if self.params["use_bigram"]:
                emb_seq_bigram_item_desc = embed(self.seq_bigram_item_desc, self.params["MAX_NUM_BIGRAMS"] + 1,
                                                 self.params["embedding_dim"], seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_bigram_item_desc = word_dropout(emb_seq_bigram_item_desc, training=self.training,
                                                            dropout=self.params["embedding_dropout"],
                                                            seed=self.params["random_seed"])
            if self.params["use_trigram"]:
                emb_seq_trigram_item_desc = embed(self.seq_trigram_item_desc, self.params["MAX_NUM_TRIGRAMS"] + 1,
                                                  self.params["embedding_dim"], seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_trigram_item_desc = word_dropout(emb_seq_trigram_item_desc, training=self.training,
                                                             dropout=self.params["embedding_dropout"],
                                                             seed=self.params["random_seed"])
            if self.params["use_subword"]:
                emb_seq_subword_item_desc = embed(self.seq_subword_item_desc, self.params["MAX_NUM_SUBWORDS"] + 1,
                                                  self.params["embedding_dim"], seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_subword_item_desc = word_dropout(emb_seq_subword_item_desc, training=self.training,
                                                             dropout=self.params["embedding_dropout"],
                                                             seed=self.params["random_seed"])

            # embed other context
            std = 0.001
            minval = -std
            maxval = std
            emb_brand = tf.Variable(
                tf.random_uniform([self.params["MAX_NUM_BRANDS"], self.params["embedding_dim"]], minval, maxval,
                                  seed=self.params["random_seed"],
                                  dtype=tf.float32))
            emb_brand_name = tf.nn.embedding_lookup(emb_brand, self.brand_name)
            # emb_brand_name = embed(self.brand_name, self.params["MAX_NUM_BRANDS"], self.params["embedding_dim"],
            #                        flatten=False, seed=self.params["random_seed"])
            # emb_category_name = embed(self.category_name, MAX_NUM_CATEGORIES, self.params["embedding_dim"],
            #                           flatten=False)
            emb_category_name1 = embed(self.category_name1, self.params["MAX_NUM_CATEGORIES_LST"][0], self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_category_name2 = embed(self.category_name2, self.params["MAX_NUM_CATEGORIES_LST"][1], self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_category_name3 = embed(self.category_name3, self.params["MAX_NUM_CATEGORIES_LST"][2], self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_item_condition = embed(self.item_condition_id, self.params["MAX_NUM_CONDITIONS"] + 1, self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_shipping = embed(self.shipping, self.params["MAX_NUM_SHIPPINGS"], self.params["embedding_dim"],
                                 flatten=False, seed=self.params["random_seed"])
