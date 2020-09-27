

import numpy as np

"""
https://github.com/tensorflow/tensorflow/blob/3989529e6041be9b16009dd8b5b3889427b47952/tensorflow/python/training/learning_rate_decay.py
"""

def _cosine_decay_restarts(learning_rate, global_step, first_decay_steps,
                           t_mul=2.0, m_mul=1.0, alpha=0.0,
                           initial_variance=0.00, variance_decay=0.55):
    initial_variance = min(learning_rate, initial_variance / 2.)
    # noisy cosine decay with restarts
    completed_fraction = global_step / first_decay_steps

    def compute_step(completed_fraction, geometric=False):