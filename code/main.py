
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