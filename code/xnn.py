
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

            #### encode
            enc_seq_name = encode(emb_seq_name, method=self.params["encode_method"],
                                  params=self.params,
                                  sequence_length=self.sequence_length_name,
                                  mask_zero=self.params["embedding_mask_zero"],
                                  scope="enc_seq_name")
            enc_seq_item_desc = encode(emb_seq_item_desc, method=self.params["encode_method"],
                                       params=self.params, sequence_length=self.sequence_length_item_desc,
                                       mask_zero=self.params["embedding_mask_zero"],
                                       scope="enc_seq_item_desc")
            # enc_seq_category_name = encode(emb_seq_category_name, method=self.params["encode_method"],
            #                                params=self.params, sequence_length=self.sequence_length_category_name,
            #                                mask_zero=self.params["embedding_mask_zero"],
            #                                scope="enc_seq_category_name")
            if self.params["use_bigram"]:
                enc_seq_bigram_item_desc = encode(emb_seq_bigram_item_desc, method="fasttext",
                                                  params=self.params,
                                                  sequence_length=self.sequence_length_item_desc,
                                                  mask_zero=self.params["embedding_mask_zero"],
                                                  scope="enc_seq_bigram_item_desc")
            if self.params["use_trigram"]:
                enc_seq_trigram_item_desc = encode(emb_seq_trigram_item_desc, method="fasttext",
                                                   params=self.params,
                                                   sequence_length=self.sequence_length_item_desc,
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   scope="enc_seq_trigram_item_desc")
            # use fasttext encode method for the following
            if self.params["use_subword"]:
                enc_seq_subword_item_desc = encode(emb_seq_subword_item_desc, method="fasttext",
                                                   params=self.params,
                                                   sequence_length=self.sequence_length_item_desc_subword,
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   scope="enc_seq_subword_item_desc")

            context = tf.concat([
                # att_seq_category_name,
                tf.layers.flatten(emb_brand_name),
                # tf.layers.flatten(emb_category_name),
                tf.layers.flatten(emb_category_name1),
                tf.layers.flatten(emb_category_name2),
                tf.layers.flatten(emb_category_name3),
                self.item_condition,
                tf.cast(self.shipping, tf.float32),
                self.num_vars],
                axis=-1, name="context")
            context_size = self.params["encode_text_dim"] * 0 + \
                           self.params["embedding_dim"] * 4 + \
                           self.params["item_condition_size"] + \
                           self.params["shipping_size"] + \
                           self.params["num_vars_size"]

            feature_dim = context_size + self.params["encode_text_dim"]
            # context = None
            feature_dim = self.params["encode_text_dim"]
            att_seq_name = attend(enc_seq_name, method=self.params["attend_method"],
                                  context=None, feature_dim=feature_dim,
                                  sequence_length=self.sequence_length_name,
                                  maxlen=self.params["max_sequence_length_name"],
                                  mask_zero=self.params["embedding_mask_zero"],
                                  training=self.training,
                                  seed=self.params["random_seed"],
                                  name="att_seq_name_attend")
            att_seq_item_desc = attend(enc_seq_item_desc, method=self.params["attend_method"],
                                       context=None, feature_dim=feature_dim,
                                       sequence_length=self.sequence_length_item_desc,
                                       maxlen=self.params["max_sequence_length_item_desc"],
                                       mask_zero=self.params["embedding_mask_zero"],
                                       training=self.training,
                                       seed=self.params["random_seed"],
                                       name="att_seq_item_desc_attend")
            if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                att_seq_name = tf.layers.Dense(self.params["embedding_dim"])(att_seq_name)
                att_seq_item_desc = tf.layers.Dense(self.params["embedding_dim"])(att_seq_item_desc)
            # since the following use fasttext encode, the `encode_text_dim` is embedding_dim
            feature_dim = context_size + self.params["embedding_dim"]
            feature_dim = self.params["embedding_dim"]
            if self.params["use_bigram"]:
                att_seq_bigram_item_desc = attend(enc_seq_bigram_item_desc, method=self.params["attend_method"],
                                                  context=None, feature_dim=feature_dim,
                                                  sequence_length=self.sequence_length_item_desc,
                                                  maxlen=self.params["max_sequence_length_item_desc"],
                                                  mask_zero=self.params["embedding_mask_zero"],
                                                  training=self.training,
                                                  seed=self.params["random_seed"],
                                                  name="att_seq_bigram_item_desc_attend")
                # reshape
                if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                    att_seq_bigram_item_desc = tf.layers.Dense(self.params["embedding_dim"],
                                                               kernel_initializer=tf.glorot_uniform_initializer(),
                                                               dtype=tf.float32, bias_initializer=tf.zeros_initializer())(att_seq_bigram_item_desc)
            if self.params["use_trigram"]:
                att_seq_trigram_item_desc = attend(enc_seq_trigram_item_desc, method=self.params["attend_method"],
                                                   context=None, feature_dim=feature_dim,
                                                   sequence_length=self.sequence_length_item_desc,
                                                   maxlen=self.params["max_sequence_length_item_desc"],
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   training=self.training,
                                                   seed=self.params["random_seed"],
                                                   name="att_seq_trigram_item_desc_attend")
                # reshape
                if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                    att_seq_trigram_item_desc = tf.layers.Dense(self.params["embedding_dim"],
                                                                kernel_initializer=tf.glorot_uniform_initializer(),
                                                                dtype=tf.float32, bias_initializer=tf.zeros_initializer())(att_seq_trigram_item_desc)
            feature_dim = context_size + self.params["embedding_dim"]
            if self.params["use_subword"]:
                att_seq_subword_item_desc = attend(enc_seq_subword_item_desc, method="ave",
                                                   context=None, feature_dim=feature_dim,
                                                   sequence_length=self.sequence_length_item_desc_subword,
                                                   maxlen=self.params["max_sequence_length_item_desc_subword"],
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   training=self.training,
                                                   seed=self.params["random_seed"],
                                                   name="att_seq_subword_item_desc_attend")
                # reshape
                if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                    att_seq_subword_item_desc = tf.layers.Dense(self.params["embedding_dim"],
                                                                kernel_initializer=tf.glorot_uniform_initializer(),
                                                                dtype=tf.float32, bias_initializer=tf.zeros_initializer())(att_seq_subword_item_desc)

            deep_list = []
            if self.params["enable_deep"]:
                # common
                common_list = [
                    # emb_seq_category_name,
                    emb_brand_name,
                    # emb_category_name,
                    emb_category_name1,
                    emb_category_name2,
                    emb_category_name3,
                    emb_item_condition,
                    emb_shipping

                ]
                tmp_common = tf.concat(common_list, axis=1)

                # word level fm for seq_name and others
                tmp_name = tf.concat([emb_seq_name, tmp_common], axis=1)
                sum_squared_name = tf.square(tf.reduce_sum(tmp_name, axis=1))
                squared_sum_name = tf.reduce_sum(tf.square(tmp_name), axis=1)
                fm_name = 0.5 * (sum_squared_name - squared_sum_name)

                # word level fm for seq_item_desc and others
                tmp_item_desc = tf.concat([emb_seq_item_desc, tmp_common], axis=1)
                sum_squared_item_desc = tf.square(tf.reduce_sum(tmp_item_desc, axis=1))
                squared_sum_item_desc = tf.reduce_sum(tf.square(tmp_item_desc), axis=1)
                fm_item_desc = 0.5 * (sum_squared_item_desc - squared_sum_item_desc)

                #### predict
                # concat
                deep_list += [
                    att_seq_name,
                    att_seq_item_desc,
                    context,
                    fm_name,
                    fm_item_desc,

                ]
                # if self.params["use_bigram"]:
                #     deep_list += [att_seq_bigram_item_desc]
                # # if self.params["use_trigram"]:
                # #     deep_list += [att_seq_trigram_item_desc]
                # if self.params["use_subword"]:
                #     deep_list += [att_seq_subword_item_desc]

            # fm layer
            fm_list = []
            if self.params["enable_fm_first_order"]:
                bias_seq_name = embed(self.seq_name, self.params["MAX_NUM_WORDS"] + 1, 1, reduce_sum=True,
                                      seed=self.params["random_seed"])
                bias_seq_item_desc = embed(self.seq_item_desc, self.params["MAX_NUM_WORDS"] + 1, 1, reduce_sum=True,
                                           seed=self.params["random_seed"])
                # bias_seq_category_name = embed(self.seq_category_name, self.params["MAX_NUM_WORDS"] + 1, 1, reduce_sum=True,
                #                                seed=self.params["random_seed"])
                if self.params["use_bigram"]:
                    bias_seq_bigram_item_desc = embed(self.seq_bigram_item_desc, self.params["MAX_NUM_BIGRAMS"] + 1, 1,
                                                      reduce_sum=True, seed=self.params["random_seed"])
                if self.params["use_trigram"]:
                    bias_seq_trigram_item_desc = embed(self.seq_trigram_item_desc, self.params["MAX_NUM_TRIGRAMS"] + 1, 1,
                                                       reduce_sum=True, seed=self.params["random_seed"])
                if self.params["use_subword"]:
                    bias_seq_subword_item_desc = embed(self.seq_subword_item_desc, self.params["MAX_NUM_SUBWORDS"] + 1, 1,
                                                       reduce_sum=True, seed=self.params["random_seed"])

                bias_brand_name = embed(self.brand_name, self.params["MAX_NUM_BRANDS"], 1, flatten=True,
                                        seed=self.params["random_seed"])
                # bias_category_name = embed(self.category_name, MAX_NUM_CATEGORIES, 1, flatten=True)
                bias_category_name1 = embed(self.category_name1, self.params["MAX_NUM_CATEGORIES_LST"][0], 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_category_name2 = embed(self.category_name2, self.params["MAX_NUM_CATEGORIES_LST"][1], 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_category_name3 = embed(self.category_name3, self.params["MAX_NUM_CATEGORIES_LST"][2], 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_item_condition = embed(self.item_condition_id, self.params["MAX_NUM_CONDITIONS"] + 1, 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_shipping = embed(self.shipping, self.params["MAX_NUM_SHIPPINGS"], 1, flatten=True,
                                      seed=self.params["random_seed"])

                fm_first_order_list = [
                    bias_seq_name,
                    bias_seq_item_desc,
                    # bias_seq_category_name,
                    bias_brand_name,
                    # bias_category_name,
                    bias_category_name1,
                    bias_category_name2,
                    bias_category_name3,
                    bias_item_condition,
                    bias_shipping,
                ]
                if self.params["use_bigram"]:
                    fm_first_order_list += [bias_seq_bigram_item_desc]
                if self.params["use_trigram"]:
                    fm_first_order_list += [bias_seq_trigram_item_desc]
                # if self.params["use_subword"]:
                #     fm_first_order_list += [bias_seq_subword_item_desc]
                tmp_first_order = tf.concat(fm_first_order_list, axis=1)
                fm_list.append(tmp_first_order)

            if self.params["enable_fm_second_order"]:
                # second order
                emb_list = [
                    tf.expand_dims(att_seq_name, axis=1),
                    tf.expand_dims(att_seq_item_desc, axis=1),
                    # tf.expand_dims(att_seq_category_name, axis=1),

                    emb_brand_name,
                    # emb_category_name,
                    emb_category_name1,
                    emb_category_name2,
                    emb_category_name3,
                    emb_item_condition,
                    emb_shipping,

                ]
                if self.params["use_bigram"]:
                    emb_list += [tf.expand_dims(att_seq_bigram_item_desc, axis=1)]
                # if self.params["use_trigram"]:
                #     emb_list += [tf.expand_dims(att_seq_trigram_item_desc, axis=1)]
                if self.params["use_subword"]:
                    emb_list += [tf.expand_dims(att_seq_subword_item_desc, axis=1)]
                emb_concat = tf.concat(emb_list, axis=1)
                emb_sum_squared = tf.square(tf.reduce_sum(emb_concat, axis=1))
                emb_squared_sum = tf.reduce_sum(tf.square(emb_concat), axis=1)

                fm_second_order = 0.5 * (emb_sum_squared - emb_squared_sum)
                fm_list.extend([emb_sum_squared, emb_squared_sum])

            if self.params["enable_fm_second_order"] and self.params["enable_fm_higher_order"]:
                fm_higher_order = dense_block(fm_second_order, hidden_units=[self.params["embedding_dim"]] * 2,
                                              dropouts=[0.] * 2, densenet=False, training=self.training, seed=self.params["random_seed"])
                fm_list.append(fm_higher_order)

            if self.params["enable_deep"]:
                deep_list.extend(fm_list)