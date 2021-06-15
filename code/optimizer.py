

"""
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwih7-6VlejYAhWGS98KHWeLCWQQFgg3MAE&url=https%3A%2F%2Fwww.bigdatarepublic.nl%2Fcustom-optimizer-in-tensorflow%2F&usg=AOvVaw3jmxRDqr2pkGRLvX6rNJrl
"""

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import random_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class LazyPowerSignOptimizer(optimizer.Optimizer):
    """Implementation of PowerSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="PowerSign"):
        super(LazyPowerSignOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t * grad * tf.exp(
            tf.log(alpha_t) * tf.sign(grad) * tf.sign(m_t)))  # Update 'ref' by subtracting 'value
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_slice = tf.gather(m, grad.indices)
        m_t = state_ops.scatter_update(m, grad.indices,
                                       tf.maximum(beta_t * m_slice + eps, tf.abs(grad.values)))
        m_t_slice = tf.gather(m_t, grad.indices)

        var_update = state_ops.scatter_sub(var, grad.indices, lr_t * grad.values * tf.exp(
            tf.log(alpha_t) * tf.sign(grad.values) * tf.sign(m_t_slice)))  # Update 'ref' by subtracting 'value
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])


class LazyAddSignOptimizer(optimizer.Optimizer):
    """Implementation of AddSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=1.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign"):
        super(LazyAddSignOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):