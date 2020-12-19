
"""
60 mins. (4 cores / 16 GB RAM / 60 minutes run-time / 1 GB scratch and output disk space)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import datetime
import os
import gc
import glob
import mmh3
import nltk
import numpy as np
import pandas as pd
import pickle as pkl
import shutil
import spacy
import time
import tensorflow as tf
import re
import string
import sys
from collections import Counter, defaultdict
from hashlib import md5

from fastcache import clru_cache as lru_cache

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import ToktokTokenizer

from multiprocessing import Pool

from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.sequence import pad_sequences

from metrics import rmse
from topk import top_k_selector
from xnn import XNN
from utils import _get_logger, _makedirs, _timestamp


##############################################################################
_makedirs("./log")
logger = _get_logger("./log", "hyperopt-%s.log" % _timestamp())

##############################################################################

RUNNING_MODE = "validation"
# RUNNING_MODE = "submission"
DEBUG = False
DUMP_DATA = True
USE_PREPROCESSED_DATA = True

USE_MULTITHREAD = False
if RUNNING_MODE == "submission":
    N_JOBS = 4
else:
    N_JOBS = 4
NUM_PARTITIONS = 32

DEBUG_SAMPLE_NUM = 200000
LRU_MAXSIZE = 2 ** 16

#######################################
# File
MISSING_VALUE_STRING = "MISSINGVALUE"
DROP_ZERO_PRICE = True
#######################################


# Preprocessing
USE_SPACY_TOKENIZER = False
USE_NLTK_TOKENIZER = False
USE_KAGGLE_TOKENIZER = False
# default: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
KERAS_TOKENIZER_FILTERS = '\'!"#%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
KERAS_TOKENIZER_FILTERS = ""
KERAS_SPLIT = " "

USE_LEMMATIZER = False
USE_STEMMER = False
USE_CLEAN = True

WORDREPLACER_DICT = {
    "bnwt": "brand new with tags",
    "nwt": "new with tags",
    "bnwot": "brand new without tags",
    "nwot": "new without tags",
    "bnip": "brand new in packet",
    "nip": "new in packet",
    "bnib": "brand new in box",
    "nib": "new in box",
    "mib": "mint in box",
    "mwob": "mint without box",
    "mip": "mint in packet",
    "mwop": "mint without packet"
}

BRAND_NAME_PATTERN_LIST = [
    ("nike", "nike"),
    ("pink", "pink"),
    ("apple", "iphone|ipod|ipad|iwatch|apple|mac"),
    ("victoria's secret", "victoria"),
    ("lularoe", "lularoe"),
    ("nintendo", "nintendo"),
    ("lululemon", "lululemon"),
    ("forever 21", "forever\s+21|forever\s+twenty\s+one"),
    ("michael kors", "michael\s+kors"),
    ("american eagle", "american\s+eagle"),
    ("rae dunn", "rae dunn"),
]

# word count |   #word
#    >= 1    |  195523
#    >= 2    |   93637
#    >= 3    |   67498
#    >= 4    |   56265
#    >= 5    |   49356
MAX_NUM_WORDS = 80000
MAX_NUM_BIGRAMS = 50000
MAX_NUM_TRIGRAMS = 50000
MAX_NUM_SUBWORDS = 20000

NUM_TOP_WORDS_NAME = 50
NUM_TOP_WORDS_ITEM_DESC = 50

MAX_CATEGORY_NAME_LEN = 3

EXTRACTED_BIGRAM = True
EXTRACTED_TRIGRAM = True
EXTRACTED_SUBWORD = False
VOCAB_HASHING_TRICK = False

######################

####################################################################
HYPEROPT_MAX_EVALS = 1

param_space_com = {
    "RUNNING_MODE": RUNNING_MODE,
    # size for the attention block
    "MAX_NUM_WORDS": MAX_NUM_WORDS,
    "MAX_NUM_BIGRAMS": MAX_NUM_BIGRAMS,
    "MAX_NUM_TRIGRAMS": MAX_NUM_TRIGRAMS,
    "MAX_NUM_SUBWORDS": MAX_NUM_SUBWORDS,

    "model_dir": "./weights",

    "item_condition_size": 5,
    "shipping_size": 1,
    "num_vars_size": 3,
    # pad_sequences
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",
    # optimization
    "optimizer_clipnorm": 1.,
    "batch_size_train": 512,
    "batch_size_inference": 512*2,
    "shuffle_with_replacement": False,
    # CyclicLR
    "t_mul": 1,
    "snapshot_every_num_cycle": 128,
    "max_snapshot_num": 14,
    "snapshot_every_epoch": 4,  # for t_mult != 1
    "eval_every_num_update": 1000,
    # static param
    "random_seed": 2018,
    "n_folds": 1,
    "validation_ratio": 0.4,
}

param_space_best = {

    #### params for input
    # bigram/trigram/subword
    "use_bigram": True,
    "use_trigram": True,
    "use_subword": False,

    # seq len
    "max_sequence_length_name": 10,
    "max_sequence_length_item_desc": 50,
    "max_sequence_length_category_name": 10,
    "max_sequence_length_item_desc_subword": 45,

    #### params for embed
    "embedding_dim": 250,
    "embedding_dropout": 0.,
    "embedding_mask_zero": False,
    "embedding_mask_zero_subword": False,

    #### params for encode
    "encode_method": "fasttext",
    # cnn
    "cnn_num_filters": 16,
    "cnn_filter_sizes": [2, 3],
    "cnn_timedistributed": False,
    # rnn
    "rnn_num_units": 16,
    "rnn_cell_type": "gru",
    #### params for attend
    "attend_method": "ave",

    #### params for predict
    # deep
    "enable_deep": True,
    # fm
    "enable_fm_first_order": True,
    "enable_fm_second_order": True,
    "enable_fm_higher_order": False,
    # fc block
    "fc_type": "fc",
    "fc_dim": 64,
    "fc_dropout": 0.,

    #### params for optimization
    "optimizer_type": "nadam",  # "nadam",  # ""lazyadam", "nadam"
    "max_lr_exp": 0.005,
    "lr_decay_each_epoch_exp": 0.9,
    "lr_jump_exp": True,
    "max_lr_cosine": 0.005,
    "base_lr": 0.00001,  # minimum lr
    "lr_decay_each_epoch_cosine": 0.5,
    "lr_jump_rate": 1.,
    "snapshot_before_restarts": 4,
    "beta1": 0.975,
    "beta2": 0.999,
    "schedule_decay": 0.004,
    # "lr_schedule": "exponential_decay",
    "lr_schedule": "cosine_decay_restarts",
    "epoch": 4,
    # CyclicLR
    "num_cycle_each_epoch": 8,

    #### params ensemble
    "enable_snapshot_ensemble": True,
    "n_runs": 2,

}
param_space_best.update(param_space_com)
if RUNNING_MODE == "submission":
    EXTRACTED_BIGRAM = param_space_best["use_bigram"]
    EXTRACTED_SUBWORD = param_space_best["use_subword"]

param_space_hyperopt = param_space_best

int_params = [
    "max_sequence_length_name",
    "max_sequence_length_item_desc",
    "max_sequence_length_item_desc_subword",
    "max_sequence_length_category_name",
    "embedding_dim", "embedding_dim",
    "cnn_num_filters", "rnn_num_units", "fc_dim",
    "epoch", "n_runs",
    "num_cycle_each_epoch", "t_mul", "snapshot_every_num_cycle",
]
int_params = set(int_params)

if DEBUG:
    param_space_hyperopt["num_cycle_each_epoch"] = param_space_best["num_cycle_each_epoch"] = 2
    param_space_hyperopt["snapshot_every_num_cycle"] = param_space_best["snapshot_every_num_cycle"] = 1
    param_space_hyperopt["batch_size_train"] = param_space_best["batch_size_train"] = 512
    param_space_hyperopt["batch_size_inference"] = param_space_best["batch_size_inference"] = 512


####################################################################################################
########################################### NLP ####################################################
####################################################################################################
def mmh3_hash_function(x):
    return mmh3.hash(x, 42, signed=True)


def md5_hash_function(x):
    return int(md5(x.encode()).hexdigest(), 16)


@lru_cache(LRU_MAXSIZE)
def hashing_trick(string, n, hash_function="mmh3"):
    if hash_function == "mmh3":
        hash_function = mmh3_hash_function
    elif hash_function == "md5":
        hash_function = md5_hash_function
    i = (hash_function(string) % n) + 1
    return i


# 5.67 µs ± 78.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word_all(word, n1=3, n2=6):
    z = []
    z_append = z.append
    word = "*" + word + "*"
    l = len(word)
    z_append(word)
    for k in range(n1, n2 + 1):
        for i in range(l - k + 1):
            z_append(word[i:i + k])
    return z


@lru_cache(LRU_MAXSIZE)
def get_subword_for_word_all0(word, n1=3, n2=6):
    z = []
    z_append = z.append
    word = "*" + word + "*"
    l = len(word)
    z_append(word)
    if l > n1:
        n2 = min(n2, l - 1)
        for i in range(l - n1 + 1):
            for k in range(n1, n2 + 1):
                if 2 * i + n2 < l:
                    z_append(word[i:(i + k)])
                    if i == 0:
                        z_append(word[-(i + k + 1):])
                    else:
                        z_append(word[-(i + k + 1):-i])
                else:
                    if 2 * i + k < l:
                        z_append(word[i:(i + k)])
                        z_append(word[-(i + k + 1):-i])
                    elif 2 * (i - 1) + n2 < l:
                        z_append(word[i:(i + k)])
    return z


# 3.44 µs ± 101 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word0(word, n1=4, n2=5, include_self=False):
    """only extract the prefix and suffix"""
    l = len(word)
    n1 = min(n1, l)
    n2 = min(n2, l)
    z1 = [word[:k] for k in range(n1, n2 + 1)]
    z2 = [word[-k:] for k in range(n1, n2 + 1)]
    z = z1 + z2
    if include_self:
        z.append(word)
    return z


# 2.49 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word(word, n1=3, n2=6, include_self=False):
    """only extract the prefix and suffix"""
    z = []
    if len(word) >= n1:
        word = "*" + word + "*"
        l = len(word)
        n1 = min(n1, l)
        n2 = min(n2, l)
        # bind method outside of loop to reduce overhead
        # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L144
        z_append = z.append
        if include_self:
            z_append(word)
        for k in range(n1, n2 + 1):
            z_append(word[:k])
            z_append(word[-k:])
    return z


# 564 µs ± 14.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
def get_subword_for_list0(input_list, n1=4, n2=5):
    subword_lst = [get_subword_for_word(w, n1, n2) for w in input_list]
    subword_lst = [w for ws in subword_lst for w in ws]
    return subword_lst


# 505 µs ± 15 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
def get_subword_for_list(input_list, n1=4, n2=5):
    subwords = []
    subwords_extend = subwords.extend
    for w in input_list:
        subwords_extend(get_subword_for_word(w, n1, n2))
    return subwords


@lru_cache(LRU_MAXSIZE)
def get_subword_for_text(text, n1=4, n2=5):
    return get_subword_for_list(text.split(" "), n1, n2)


stopwords = [
    "and",
    "the",
    "for",
    "a",
    "in",
    "to",
    "is",
    # "s",
    "of",
    "i",
    "on",
    "it",
    "you",
    "your",
    "are",
    "this",
    "my",
]
stopwords = set(stopwords)


# spacy model
class SpacyTokenizer(object):
    def __init__(self):
        self.nlp = spacy.load("en", disable=["parser", "tagger", "ner"])

    def tokenize(self, text):
        tokens = [tok.lower_ for tok in self.nlp(text)]
        # tokens = get_valid_words(tokens)
        return tokens


LEMMATIZER = nltk.stem.wordnet.WordNetLemmatizer()
STEMMER = nltk.stem.snowball.EnglishStemmer()
TOKTOKTOKENIZER = ToktokTokenizer()


# SPACYTOKENIZER = SpacyTokenizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_valid_words(sentence):
    res = [w.strip() for w in sentence]
    return [w for w in res if w]


@lru_cache(LRU_MAXSIZE)
def stem_word(word):
    return STEMMER.stem(word)


@lru_cache(LRU_MAXSIZE)
def lemmatize_word(word, pos=wordnet.NOUN):
    return LEMMATIZER.lemmatize(word, pos)


def stem_sentence(sentence):
    return [stem_word(w) for w in get_valid_words(sentence)]


def lemmatize_sentence(sentence):
    res = []
    sentence_ = get_valid_words(sentence)
    for word, pos in pos_tag(sentence_):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatize_word(word, pos=wordnet_pos))
    return res


def stem_lemmatize_sentence(sentence):
    return [stem_word(word) for word in lemmatize_sentence(sentence)]


TRANSLATE_MAP = maketrans(KERAS_TOKENIZER_FILTERS, KERAS_SPLIT * len(KERAS_TOKENIZER_FILTERS))


def get_tokenizer():
    if USE_LEMMATIZER and USE_STEMMER:
        return stem_lemmatize_sentence
    elif USE_LEMMATIZER:
        return lemmatize_sentence
    elif USE_STEMMER:
        return stem_sentence
    else:
        return get_valid_words


tokenizer = get_tokenizer()


#
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
# @lru_cache(LRU_MAXSIZE)
# 40.1 µs ± 918 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams0(words, ngram_value):
    # # return list
    ngrams = [" ".join(ngram) for ngram in zip(*[words[i:] for i in range(ngram_value)])]
    # return generator (10x faster)
    # ngrams = (" ".join(ngram) for ngram in zip(*[words[i:] for i in range(ngram_value)]))
    return ngrams


# 36.2 µs ± 1.04 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams(words, ngram_value):
    tokens = []
    tokens_append = tokens.append
    for i in range(ngram_value):
        tokens_append(words[i:])
    ngrams = []
    ngrams_append = ngrams.append
    space_join = " ".join
    for ngram in zip(*tokens):
        ngrams_append(space_join(ngram))
    return ngrams


def get_bigrams(words):
    return get_ngrams(words, 2)


def get_trigrams(words):
    return get_ngrams(words, 3)


@lru_cache(LRU_MAXSIZE)
# 68.8 µs ± 1.86 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams_range(text, ngram_range):
    unigrams = text.split(" ")
    ngrams = []
    ngrams_extend = ngrams.extend
    for i in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_extend(get_ngrams(unigrams, i))
    return ngrams


# 69.6 µs ± 1.45 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams_range0(text, ngram_range):
    unigrams = text.split(" ")
    res = []
    for i in ngram_range:
        res += get_ngrams(unigrams, i)
    res += unigrams
    return res


@lru_cache(LRU_MAXSIZE)
def stem(s):
    return STEMMER.stem(s)


tags = re.compile(r'<.+?>')
whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')


@lru_cache(LRU_MAXSIZE)
def clean_text(text):
    # text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        # if len(t) <= 2 and not t.isdigit():
        #     continue
        if t in stopwords:
            continue