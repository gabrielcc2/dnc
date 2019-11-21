# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A one hop  task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sonnet as snt
import tensorflow as tf
import numpy as np
import random
import copy

from tensorflow import Tensor

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations',
                                                           'target', 'mask'))


def masked_sigmoid_cross_entropy(logits,
                                 target,
                                 mask,
                                 time_average=False,
                                 log_prob_in_bits=False):
  """Adds ops to graph which compute the (scalar) NLL of the target sequence.

  The logits parametrize independent bernoulli distributions per time-step and
  per batch element, and irrelevant time/batch elements are masked out by the
  mask tensor.

  Args:
    logits: `Tensor` of activations for which sigmoid(`logits`) gives the
        bernoulli parameter.
    target: time-major `Tensor` of target.
    mask: time-major `Tensor` to be multiplied elementwise with cost T x B cost
        masking out irrelevant time-steps.
    time_average: optionally average over the time dimension (sum by default).
    log_prob_in_bits: iff True express log-probabilities in bits (default nats).

  Returns:
    A `Tensor` representing the log-probability of the target.
  """
  xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
  loss_time_batch = tf.reduce_sum(xent, axis=2)
  loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)

  batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

  if time_average:
    mask_count = tf.reduce_sum(mask, axis=0)
    loss_batch /= (mask_count + np.finfo(np.float32).eps)

  loss = tf.reduce_sum(loss_batch) / batch_size
  if log_prob_in_bits:
    loss /= tf.log(2.)

  return loss


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.

  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch
  """

  def _readable(datum):
    return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

  obs_batch = data.observations
  targ_batch = data.target

  iterate_over = range(batch_size) if whole_batch else range(1)

  batch_strings = []
  for batch_index in iterate_over:
    obs = obs_batch[:, batch_index, :]
    targ = targ_batch[:, batch_index, :]

    obs_channels = range(obs.shape[1])
    targ_channels = range(targ.shape[1])
    obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
    targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

    readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
    readable_targ = 'Targets:\n' + '\n'.join(targ_channel_strings)
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output[:, batch_index, :]
      output_strings = [_readable(output[:, i]) for i in targ_channels]
      strings.append('Model Output:\n' + '\n'.join(output_strings))

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)


class OneHop(snt.AbstractModule):
  """Sequence data generator for the task of finding the one hop neighbors of an item in
  a graph.

  This is a small adaptation over the code for repeat copy from the original authors (https://github.com/deepmind/dnc)

  When called, an instance of this class will return a tuple of tensorflow ops
  (obs, targ, mask), representing an input sequence, target sequence, and
  binary mask. Each of these ops produces tensors whose first two dimensions
  represent sequence position and batch index respectively. The value in
  mask[t, b] is equal to 1 iff a prediction about targ[t, b, :] should be
  penalized and 0 otherwise.

  For each realisation from this generator, the observation sequence is
  comprised of some binary vectors (and some flags).

  This observation sequence encodes the structure of some graph, providing
  first 2 ones in the positions that correspond to the vertexes being connected,
  followed by a 1 in the position of the start node, and 1 in the position of
  the end node.

  After giving the structure, it encodes some questions for the paired nodes.

  The target sequence is comprised of the answers (the matching vertex).
  Here's an example:

  ```none
  Note: blank space represents 0.

  time ------------------------------------------>

                +-------------------------------+
  mask:         |0000000000111111111111111111111|
                +-------------------------------+

                +-------------------------------+
  target:       |               1              |
                |                              |
                |                              |
                |             1                | 'start-marker' channel
                |                1             | 'end-marker' channel.
                +-------------------------------+

                +-------------------------------+
  observation:  | 110110                       |
                | 101000                       |
                | 000101    1                  |
                |1        1                    | 'start-marker' channel
                |       1    1                 | 'end-marker' channel.
                +-------------------------------+
  ```

  As in the diagram, the target sequence is offset to begin directly after the
  observation sequence; both sequences are padded with zeros to accomplish this,
  resulting in their lengths being equal. Additional padding is done at the end
  so that all sequences in a minibatch represent tensors with the same shape.
  """

  def __init__(
      self,
      batch_size=32,
      word_length=20,
      max_items=32,
      max_edges=3,
      max_questions=3,
      norm_max=10,
      log_prob_in_bits=False,
      time_average_cost=False,
      name='one_hop',):
    """Creates an instance of OneHop task.

    Args:
      name: A name for the generator instance (for name scope purposes).
      batch_size: Minibatch size per realization.
      max_items: Size of sequence.
      max_length: Upper limit on number of random binary vectors in the
          observation pattern.
      max_edges: Limit on number of edges
      max_questions: Upper limit on number of questions
      norm_max: Upper limit on uniform distribution w.r.t which the encoding
          of the number of repetitions presented in the observation sequence
          is normalised.
      log_prob_in_bits: By default, log probabilities are expressed in units of
          nats. If true, express log probabilities in bits.
      time_average_cost: If true, the cost at each time step will be
          divided by the `true`, sequence length, the number of non-masked time
          steps, in each sequence before any subsequent reduction over the time
          and batch dimensions.
    """
    super(OneHop, self).__init__(name=name)

    self._batch_size = batch_size
    self._word_length = word_length
    self._max_items = max_items
    self._max_edges = max_edges
    self._max_questions = max_questions
    self._norm_max = norm_max
    self._log_prob_in_bits = log_prob_in_bits
    self._time_average_cost = time_average_cost

  def _normalise(self, val):
    return val / self._norm_max

  def _unnormalise(self, val):
    return val * self._norm_max

  @property
  def time_average_cost(self):
    return self._time_average_cost

  @property
  def log_prob_in_bits(self):
    return self._log_prob_in_bits

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def word_length(self):
    return self._word_length

  def _input_generator(self):
    for w in range(1000):
      obsarr = [[0 if x != self._word_length - 2 else 1 for x in range(self._word_length)]]
      targarr = [[0 for x in range(self._word_length)]]
      items_counter = 1

      #Edge insertion....
      edge_count = random.choice(range(1, self._max_edges+1))
      edge_tracker = dict()
      for i in range(0, edge_count):
        #We pick 2 random vertices
        v1 = random.choice(range(1, self._word_length-2))
        v2 = random.choice(range(1, self._word_length-2))
        while (v2 == v1): #No self-loops
          v2 = random.choice(range(1, self._word_length-2))
        #We track the existing edges, for question-answering purposes
        if v1 in edge_tracker:
          a = edge_tracker[v1]
          a.add(v2)
          edge_tracker[v1] = a
        else:
          a = set()
          a.add(v2)
          edge_tracker[v1] = a

        #Next we insert a sequence of 3 items: edge, source vertex, target vertex...
        insert = [0 for x in range(self._word_length)]
        insert[v1 - 1] = 1
        insert[v2 - 1] = 1
        items_counter += 1
        obsarr.append(insert)

        insert2 = [0 for x in range(self._word_length)]
        insert2[v1 - 1] = 1
        obsarr.append(insert2)
        items_counter += 1

        insert3 = [0 for x in range(self._word_length)]
        insert3[v2 - 1] = 1
        obsarr.append(insert3)
        items_counter += 1

      #Questions and answers...
      obsarr.append([0 if x != self._word_length - 1 else 1 for x in range(self._word_length)])
      items_counter += 1
      obsarr.append([0 if x != self._word_length - 2 else 1 for x in range(self._word_length)])
      items_counter += 1
      q_count = random.choice(range(1, self._max_questions+1))
      for i in range(1, items_counter + q_count):
        targarr.append([0 for x in range(self._word_length)])
      targarr.append([0 if x != self._word_length - 2 else 1 for x in range(self._word_length)])
      items_counter2 = items_counter + q_count
      items_counter2 += 1

      #Once we know the number of questions, we can also define the mask...
      maskarr = np.zeros((self._max_items,))
      for i in range(items_counter + q_count + 1, self._max_items):
        maskarr[i] = 1

      #Questions and answers...
      for i in range(0, q_count):
        insert = [0 for x in range(self._word_length)]
        pick = random.choice(list(edge_tracker.keys()))
        insert[pick - 1] = 1
        items_counter += 1
        obsarr.append(insert)
        for tgt in list(edge_tracker[pick]):
          insert = [0 for x in range(self._word_length)]
          insert[tgt - 1] = 1
          items_counter2 += 1
          targarr.append(insert)
      targarr.append([0 if x != self._word_length - 1 else 1 for x in range(self._word_length)])
      obsarr.append([0 if x != self._word_length - 1 else 1 for x in range(self._word_length)])
      items_counter += 1
      items_counter2 += 1

      while (items_counter < self._max_items):
        obsarr.append([0 for x in range(self._word_length)])
        items_counter += 1
      while (items_counter2 < self._max_items):
        targarr.append([0 for x in range(self._word_length)])
        items_counter2 += 1

      obsarr = np.array(obsarr)
      targarr = np.array(targarr)
      yield obsarr, targarr, maskarr

  def _build(self):
    """Implements build method which adds ops to graph."""
    obs_batch_shape = [self._batch_size, self._max_items, self._word_length]
    targ_batch_shape = [self._batch_size, self._max_items, self._word_length]
    mask_batch_trans_shape = [self._batch_size, self._max_items]
    obs_tensors = []
    targ_tensors = []
    mask_tensors = []

    # Generates patterns for each element independently.
    dataset = tf.data.Dataset.from_generator(self._input_generator, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([self._max_items,self._word_length]), tf.TensorShape([self._max_items,self._word_length]),tf.TensorShape([self._max_items])))
    dataset = dataset.repeat(count=-1)#An infinite repeat of the generator...
    for batch_index in range(self._batch_size):
      iterator = dataset.make_one_shot_iterator()
      x, y, z = iterator.get_next()
      obs_tensors.append(x)
      targ_tensors.append(y)
      mask_tensors.append(z)
      # Concatenate each batch element into a single tensor.

    obs = tf.reshape(tf.concat(obs_tensors, 1), obs_batch_shape)
    targ = tf.reshape(tf.concat(targ_tensors, 1), targ_batch_shape)
    mask = tf.transpose(
        tf.reshape(tf.concat(mask_tensors, 0), mask_batch_trans_shape))
    return DatasetTensors(obs, targ, mask)

  def cost(self, logits, targ, mask):
    return masked_sigmoid_cross_entropy(
        logits,
        targ,
        mask,
        time_average=self.time_average_cost,
        log_prob_in_bits=self.log_prob_in_bits)

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    obs = data.observations
    data = data._replace(observations=obs)
    return bitstring_readable(data, self.batch_size, model_output, whole_batch)
