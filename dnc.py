from collections import namedtuple

import numpy as np
import tensorflow as tf
from numpy.random import permutation
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.math_ops import sigmoid, tanh

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_EPSILON = 1e-6

DNCStateTuple = namedtuple('DNCStateTuple',
                           ('controller_state', 'memory_state', 'memory_usage',
                           'read_values', 'read_weights', 'write_weights',
                           'precedence', 'link_matrix'))

def oneplus(x, name = 'oneplus'):
  with tf.name_scope(name):
    return 1. + tf.nn.softplus(x)


class _LayerRNNCell(rnn_cell.RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: `VariableScope` for the created subgraph; if not provided,
        defaults to standard `tf.layers.Layer` behavior.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(self, inputs, state, scope=scope)

class DNCCell(_LayerRNNCell):

  """Neural Turing Machine Recurrent Unit cell

  (cf. https://arxiv.org/abs/1410.5401).

  """

  def __init__(self, N, W, read_heads = 1, write_heads = 1,
               controller_size = 19, controller_depth = 3, num_proj = None,
               initializer=None, reuse = None, name = None):
    #TODO:
    # Number of projection
    """Initialize the parameters for an DNC cell.

    Args:
      N: int, The number of memory locations.
      W: int, The size of each memory location.
      read_heads: int, The number of read heads to be used (in R^N).
      write_heads: int, The number of write heads to be used (in R^N).

      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      activation: Activation function of the inner states.
    """
    super(DNCCell, self).__init__(_reuse=reuse, name=name)

    self._N = N
    self._W = W
    self._read_heads = read_heads
    self._write_heads = write_heads
    if self._write_heads > 1:
      raise ValueError("DNC currently supports a single write vector")
    self._controller_size = controller_size
    self._controller_depth = controller_depth
    self._initializer = initializer

    R = self._read_heads
    with tf.name_scope('controller'):
      self._controller = rnn_cell.MultiRNNCell(
          [rnn_cell.LSTMCell(self._controller_size,
                             initializer = self._initializer)
                                for _ in range(self._controller_depth)])
    del R, W

    self._num_proj = num_proj
    if num_proj:
      self._state_size = (rnn_cell.LSTMStateTuple(self._W, num_proj))
      self._output_size = num_proj
    else:
      self._state_size = (rnn_cell.LSTMStateTuple(self._W, self._W))
      self._output_size = self._W

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _read(self, read_heads, memories):
    '''
    Reads from memory using read_heads

    Args:
    read_heads : 3-d tensor of dimensions : bs x R x N
    memories : 3-d tensor of dimensions : bs x N x M

    Returns:
    output : 3-d tensor of dimensions : bs x R x M
    '''
    with tf.name_scope('ntm_read'):
      output = tf.stack([ tf.matmul(heads, memory) for heads, memory
                  in zip(tf.unstack(read_heads), tf.unstack(memories))])
      return output

  #NOTE: This only works for one write head
  def _write(self, memories, write_heads, erase, write_vector):
    '''
    Erases content from memory based on the erase vector e and the write heads.
    Then writes to memory the vector a using write_heads.

    Args:
    memories : 3-d tensor of dimensions : bs x N x W
    write_heads : 3-d tensor of dimensions : bs x N
    e : 2-d erase vector of dimensions :  bs x W
    write_vector : 2-d content vector of dimensions :  bs x W

    '''
    write_heads = tf.expand_dims(write_heads, 2)
    with tf.name_scope('dnc_erase'):
      memories = tf.multiply(memories,
                             tf.ones(memories.shape)
                             - tf.matmul(write_heads,
                                         tf.expand_dims(erase, 1)))

    with tf.name_scope('dnc_write'):
      memories = memories + tf.matmul(write_heads,
                                      tf.expand_dims(write_vector, 1))

    return memories

  def _content_based_addressing(self, heads, memories, strength, name = ''):
    '''
    Compute the content based addressing.

    Args:
    heads : 2-d tensor of dimensions : bs x (#heads * W)
    memories : 3-d tensor of dimensions : bs x N x W
    strength : 2-d tensor of dimensions : bs x #heads

    Returns:
    output : 3-d tensor of dimensions : bs x #heads x N
    '''
    # k = number of read or write heads
    k  = int(heads.shape[1].value / self._W)

    with tf.name_scope(name + 'content_based_addressing'):
      with tf.name_scope('cosine_similarity'):
        # bs x k x W
        heads = tf.reshape(heads, [-1, k, self._W])

        heads_norm = tf.nn.l2_normalize(heads, 2)
        memory_norm = tf.nn.l2_normalize(memories, 2)

        # bs x k x 1
        strength = tf.expand_dims(strength, -1)

        # bs x R x N
        similarity = tf.stack([tf.matmul(heads, tf.transpose(memory)) \
            for heads, memory in
                zip(tf.unstack(heads_norm), tf.unstack(memory_norm))])

      # softmax
      # bs x R x N
      content_addressing = tf.nn.softmax(tf.multiply(strength, similarity))

      return content_addressing

  def _dynamic_memory_allocation_addressing(self, free_gates, prev_read_weights,
                                            prev_write_weights, prev_memory_usage,
                                            name = ''):
    '''
    Compute the dynamic memory allocation based addressing.

    Args:
    free_gates : 2-d tensor of dimensions : bs x R
    prev_read_weights : 3-d tensor of dimensions : bs x R x N
    prev_write_weights :2-d tensor of dimensions : bs x N
    memory_usage : 2-d tensor of dimensions : bs x N

    Returns:
    a : 2-d tensor of dimensions : bs x N

    '''
    with tf.variable_scope(name + 'dynamic_memory_allocation'):

      # bs x R x 1
      free_gates = tf.expand_dims(free_gates, -1)

      # memory_retention
      # bs x N
      memory_retention = tf.reduce_prod(1. - free_gates * prev_read_weights,
                                        axis = 1)
      # memory_usage
      # bs x N
      memory_usage = tf.multiply((prev_memory_usage
                                + prev_write_weights
                                - tf.multiply(prev_memory_usage,
                                              prev_write_weights)),
                                memory_retention)


      memory_usage = _EPSILON + (1 - _EPSILON) * memory_usage

      # values and phi in descending order
      # bs x N
      values, phi = tf.nn.top_k(memory_usage, k=self._N, sorted=True)

      # values and phi in ascending order
      # bs x N
      values = tf.reverse(values, axis=[-1])
      phi = tf.reverse(phi, axis=[-1])

      # cumulative product
      prod = tf.cumprod(values, axis=-1, exclusive=True)
      prod = tf.stack([tf.gather(p, i) for i, p in
                          zip(tf.unstack(phi), tf.unstack(prod))])

      a = tf.multiply((1 - memory_usage), prod)

    return a, memory_usage

  def _temporal_memory_linkage_addressing(self, write_weights, prev_precedence,
                                          prev_read_weights, prev_link_matrix,
                                          name = ''):
    '''
    Compute the temporal memory linkage addressing.

    Args:
    write_weights : 2-d tensor of shape : bs x N
    prev_precedence : 2-d tensor of shape : bs x N
    prev_read_weights : 3-d tensor of shape : bs x R x N
    prev_link_matrix : 3-d tensor of shape : bs x N x N

    Returns:
    backward_weighting : 3-d tensor of shape : bs x R x N
    forward_weighting : 3-d tensor of shape : bs x R x N
    precedence : 2-d tensor of shape : bs x N
    link_matrix : 3-d tensor of shape : bs x N x N
    '''
    with tf.name_scope(name + 'temporal_memory_linkage_addressing'):
      # precedence weights
      # bs x N
      precedence = (1. - tf.reduce_sum(write_weights, axis=1, keep_dims=True)) \
                    * prev_precedence \
                    + write_weights

      # link_matrix update
      # bs x N x N
      link_matrix = tf.multiply(1 \
                                - tf.expand_dims(write_weights, 2)
                                - tf.expand_dims(write_weights, 1),
                                prev_link_matrix) \
                    + tf.matmul(tf.expand_dims(write_weights, 2),
                                tf.expand_dims(prev_precedence, 1))

      # backward_weighting
      # bs x R x N
      backward_weighting = tf.matmul(prev_read_weights, link_matrix)

      # forward_weighting
      # bs x R x N
      forward_weighting = tf.matmul(prev_read_weights,
                                    tf.transpose(link_matrix, perm = [0,2,1]))

    return backward_weighting, forward_weighting, precedence, link_matrix

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    # num_proj = self._cell_size if self._num_proj is None else self._num_proj

    # Weights:
    R = self._read_heads
    W = self._W

    # kernel size:
    # input dim:
    # hidden_states (controller_size x controller_depth)
    # output dim:
    # dim(v_t) = W
    # dim(eta_t) = (W * R) + 3 * W + 5 * R + 3
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[(self._controller_size * self._controller_depth),
          self._output_size + (W * R) + 3 * W + 5 * R + 3],
          initializer=self._initializer)

    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._output_size + (W * R) + 3 * W + 5 * R + 3],
        initializer=tf.zeros_initializer(dtype=self.dtype))

    # projection size : (R * W) x output_size
    self._read_projection_w = self.add_variable(
        'read_projection_weights',
        shape=[R * W, self._output_size],
        initializer = self._initializer)

    self._read_projection_bias = self.add_variable(
        'read_projection_bias',
        shape=[self._output_size],
        initializer=tf.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state, scope=None):
    """Run one step of Associative LSTM.

    Args:
      inputs: input Tensor, 2D, batch x input_size.
      state: a tuple of state Tensors, both `2-D`, with column sizes `c_state`
          and `m_state`.
      scope: VariableScope for the created subgraph; defaults to
          "AssociativeLSTMCell".

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           cell_size otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    # num_proj = self._cell_size if self._num_proj is None else self._num_proj

    # (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]


    # preparing input
    with tf.name_scope('preparing_input'):

      # A tuple of contoller_depth items of shape:
      # bs x (controller_size x controller_depth)
      prev_controller_state = state.controller_state

      # bs x N x W
      prev_memories = state.memory_state

      # bs x N
      prev_memory_usage = state.memory_usage

      # bs x (R * W)
      prev_read_values = state.read_values

      # bs x read_heads
      prev_read_weights = state.read_weights

      # bs x write_heads
      prev_write_weights = state.write_weights

      # bs x N
      prev_precedence = state.precedence

      # bs x N x N
      prev_link_matrix = state.link_matrix

    # bs x (input_size + (R * W))
    controller_inputs = tf.concat([inputs, prev_read_values],
                                  axis=1,
                                  name='controller_inputs')

    # bs x (input_size + (R x W))
    _, controller_state = self._controller(controller_inputs,
                                           prev_controller_state)

    # bs x (controller_depth * controller_size)
    controller_outputs = tf.concat([h for _, h in controller_state],
                                   axis=1, name='controller_outputs')

    # computing v_t and eta_t
    # bs x (W + (W * R) + 3 * W + 5 * R + 3)
    dnc_matrix = tf.matmul(controller_outputs, self._kernel)
    dnc_matrix = tf.nn.bias_add(dnc_matrix, self._bias)

    with tf.name_scope('controller_split_outputs'):
      # v_t shape: bs x num_proj
      # read_keys shape: bs x (R x W)
      # read_strength shape: bs x R
      # write_key shape: bs x W
      # write_strength shape: bs x 1
      # erase shape: bs x W
      # write_vector shape: bs x W
      # free_gates shape: bs x R
      # allocation_gate shape: bs x 1
      # write_gate shape: bs x 1
      # read_modes shape: bs x (3 * R)
      v_t, read_keys, read_strength, write_key, write_strength, \
          erase, write_vector, free_gates, allocation_gate, write_gate, \
          read_modes = tf.split(dnc_matrix, axis = 1,
              num_or_size_splits = [self._output_size,
                                    self._W * self._read_heads,
                                    self._read_heads,
                                    self._W,
                                    1,
                                    self._W,
                                    self._W,
                                    self._read_heads,
                                    1,
                                    1,
                                    3 * self._read_heads])

      read_strength = oneplus(read_strength)
      write_strength = oneplus(write_strength)
      erase = sigmoid(erase)
      free_gates = sigmoid(free_gates)
      allocation_gate = sigmoid(allocation_gate)
      write_gate = sigmoid(write_gate)

      # pi1, pi2, pi3 shapes: bs x R x 1 each
      read_modes = tf.nn.softmax(read_modes)
      read_modes = tf.reshape(read_modes, [-1, self._read_heads, 3])
      pi1, pi2, pi3 = tf.split(read_modes, num_or_size_splits=3, axis=2)

    # content based read weights
    # bs x R x N
    c_r_w = self._content_based_addressing(read_keys, prev_memories,
                                           read_strength)

    # content based write weights
    # bs x 1 x N
    c_w_w = self._content_based_addressing(write_key, prev_memories,
                                           write_strength)

    # NOTE: bs x N (because we have only one write weight)
    c_w_w = tf.squeeze(c_w_w, axis=1)


    # allocation write weight : bs x N
    # memory_usage : bs x N
    d_w_w, memory_usage  = self._dynamic_memory_allocation_addressing(
                                free_gates, prev_read_weights,
                                prev_write_weights, prev_memory_usage)

    # write_weights
    # bs x N
    write_weights = write_gate \
                    * (allocation_gate * d_w_w + (1 - allocation_gate) * c_w_w)

    # backward_weighting : bs x R x N
    # forward_weighting :  bs x R x N
    backward_weighting, forward_weighting, precedence, link_matrix = \
        self._temporal_memory_linkage_addressing(write_weights, prev_precedence,
                                                 prev_read_weights,
                                                 prev_link_matrix)

    # read_weights
    # bs x R x N
    read_weights = pi1 * backward_weighting \
                   + pi2 * forward_weighting \
                   + pi3 * c_r_w

    # updated memories
    # bs x N x W
    memories = self._write(prev_memories, write_weights, erase, write_vector)

    # read_values
    # bs x (R * M)
    read_values = self._read(read_weights, memories)
    read_values = tf.reshape(read_values, [-1, self._W * self._read_heads])

    # next state
    state = DNCStateTuple(controller_state, memories, memory_usage,
                          read_values, read_weights, write_weights,
                          precedence, link_matrix)


    # current output
    output = tf.matmul(read_values, self._read_projection_w)
    output = tf.nn.bias_add(output, self._read_projection_bias)
    output = output + v_t

    return output, state

  def zero_state(self, batch_size, dtype):
    controller_state = self._controller.zero_state(batch_size, tf.float32)
    memory_state = tf.zeros([batch_size, self._N, self._W])
    memory_usage = tf.zeros([batch_size, self._N])

    read_values = tf.zeros([batch_size, self._read_heads * self._W])
    read_weights = tf.zeros([batch_size, self._read_heads, self._N])
    write_weights = tf.zeros([batch_size, self._N])

    precedence = tf.zeros([batch_size, self._N])
    link_matrix = tf.zeros([batch_size, self._N, self._N])

    state = DNCStateTuple(controller_state, memory_state, memory_usage,
                          read_values, read_weights, write_weights,
                          precedence, link_matrix)
    return state

