
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
                deep_in = tf.concat(deep_list, axis=-1, name="concat")
                # dense
                hidden_units = [self.params["fc_dim"]*4, self.params["fc_dim"]*2, self.params["fc_dim"]]
                dropouts = [self.params["fc_dropout"]] * len(hidden_units)
                if self.params["fc_type"] == "fc":
                    deep_out = dense_block(deep_in, hidden_units=hidden_units, dropouts=dropouts, densenet=False,
                                           training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "resnet":
                    deep_out = resnet_block(deep_in, hidden_units=hidden_units, dropouts=dropouts, cardinality=1,
                                            dense_shortcut=True, training=self.training,
                                            seed=self.params["random_seed"])
                elif self.params["fc_type"] == "densenet":
                    deep_out = dense_block(deep_in, hidden_units=hidden_units, dropouts=dropouts, densenet=True,
                                           training=self.training, seed=self.params["random_seed"])
                fm_list.append(deep_out)


            fm_list.append(self.num_vars)
            fm_list.append(self.item_condition)
            fm_list.append(tf.cast(self.shipping, tf.float32))
            out = tf.concat(fm_list, axis=-1)


            self.pred = tf.layers.Dense(1, kernel_initializer=tf.glorot_uniform_initializer(self.params["random_seed"]),
                                        dtype=tf.float32, bias_initializer=tf.zeros_initializer())(out)

            # intermediate meta
            self.meta = out

            #### loss
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.target, self.pred))
            # target is normalized, so std is 1
            # we apply 3 sigma principle
            std = 1.
            self.loss = tf.losses.huber_loss(self.target, self.pred, delta=1. * std)
            # self.loss = self.rmse

            #### optimizer
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            if self.params["optimizer_type"] == "nadam":
                self.optimizer = LazyNadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                beta2=self.params["beta2"], epsilon=1e-8,
                                                schedule_decay=self.params["schedule_decay"])
            elif self.params["optimizer_type"] == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                        beta2=self.params["beta2"], epsilon=1e-8)
            elif self.params["optimizer_type"] == "lazyadam":
                self.optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate,
                                                                  beta1=self.params["beta1"],
                                                                  beta2=self.params["beta2"], epsilon=1e-8)
            elif self.params["optimizer_type"] == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-7)
            elif self.params["optimizer_type"] == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.params["optimizer_type"] == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
            elif self.params["optimizer_type"] == "rmsprop":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9, momentum=0.9,
                                                           epsilon=1e-8)
            elif self.params["optimizer_type"] == "lazypowersign":
                self.optimizer = LazyPowerSignOptimizer(learning_rate=self.learning_rate)
            elif self.params["optimizer_type"] == "lazyaddsign":
                self.optimizer = LazyAddSignOptimizer(learning_rate=self.learning_rate)
            elif self.params["optimizer_type"] == "lazyamsgrad":
                self.optimizer = LazyAMSGradOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                  beta2=self.params["beta2"], epsilon=1e-8)

            #### training op
            """
            https://stackoverflow.com/questions/35803425/update-only-part-of-the-word-embedding-matrix-in-tensorflow
            TL;DR: The default implementation of opt.minimize(loss), TensorFlow will generate a sparse update for 
            word_emb that modifies only the rows of word_emb that participated in the forward pass.

            The gradient of the tf.gather(word_emb, indices) op with respect to word_emb is a tf.IndexedSlices object
             (see the implementation for more details). This object represents a sparse tensor that is zero everywhere, 
             except for the rows selected by indices. A call to opt.minimize(loss) calls 
             AdamOptimizer._apply_sparse(word_emb_grad, word_emb), which makes a call to tf.scatter_sub(word_emb, ...)* 