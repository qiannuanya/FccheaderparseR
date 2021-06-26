

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
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t * grad * (1.0 + alpha_t * tf.sign(grad) * tf.sign(m_t)))
        # Create an op that groups multiple operations
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_slice = tf.gather(m, grad.indices)
        m_t = state_ops.scatter_update(m, grad.indices,
                                       tf.maximum(beta_t * m_slice + eps, tf.abs(grad.values)))
        m_t_slice = tf.gather(m_t, grad.indices)

        var_update = state_ops.scatter_sub(var, grad.indices,
                                           lr_t * grad.values * (
                                                   1.0 + alpha_t * tf.sign(grad.values) * tf.sign(m_t_slice)))

        # Create an op that groups multiple operations
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])


class LazyAMSGradOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="AMSGrad"):
        super(LazyAMSGradOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "v_prime", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1. - beta1_t) * grad, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, beta2_t * v + (1. - beta2_t) * tf.square(grad), use_locking=self._use_locking)
        v_prime = self.get_slot(var, "v_prime")
        v_t_prime = state_ops.assign(v_prime, tf.maximum(v_prime, v_t))

        var_update = state_ops.assign_sub(var,
                                          lr_t * m_t / (tf.sqrt(v_t_prime) + epsilon_t),
                                          use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, v_t_prime])

    # keras Nadam update rule
    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.scatter_update(m, grad.indices,
                                       beta1_t * array_ops.gather(m, grad.indices) +
                                       (1. - beta1_t) * grad.values,
                                       use_locking=self._use_locking)
        m_t_slice = tf.gather(m_t, grad.indices)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.scatter_update(v, grad.indices,
                                       beta2_t * array_ops.gather(v, grad.indices) +
                                       (1. - beta2_t) * tf.square(grad.values),
                                       use_locking=self._use_locking)
        v_prime = self.get_slot(var, "v_prime")
        v_t_slice = tf.gather(v_t, grad.indices)
        v_prime_slice = tf.gather(v_prime, grad.indices)
        v_t_prime = state_ops.scatter_update(v_prime, grad.indices, tf.maximum(v_prime_slice, v_t_slice))

        v_t_prime_slice = array_ops.gather(v_t_prime, grad.indices)
        var_update = state_ops.scatter_sub(var, grad.indices,
                                           lr_t * m_t_slice / (math_ops.sqrt(v_t_prime_slice) + epsilon_t),
                                           use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, v_t_prime])


class LazyNadamOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, use_locking=False, name="Nadam"):
        super(LazyNadamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._schedule_decay = schedule_decay
        # momentum cache decay
        self._momentum_cache_decay = tf.cast(0.96, tf.float32)
        self._momentum_cache_const = tf.pow(self._momentum_cache_decay, 1. * schedule_decay)

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._schedule_decay_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None
        self._iterations = None
        self._m_schedule = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")