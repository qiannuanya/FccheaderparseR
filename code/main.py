
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