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
"""A repeat copy task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sonnet as snt
import tensorflow as tf
import numpy as np
import random

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
  """Sequence data generator for the task of repeating a random binary pattern.

  When called, an instance of this class will return a tuple of tensorflow ops
  (obs, targ, mask), representing an input sequence, target sequence, and
  binary mask. Each of these ops produces tensors whose first two dimensions
  represent sequence position and batch index respectively. The value in
  mask[t, b] is equal to 1 iff a prediction about targ[t, b, :] should be
  penalized and 0 otherwise.

  For each realisation from this generator, the observation sequence is
  comprised of I.I.D. uniform-random binary vectors (and some flags).

  The target sequence is comprised of this binary pattern repeated
  some number of times (and some flags). Before explaining in more detail,
  let's examine the setup pictorially for a single batch element:

  ```none
  Note: blank space represents 0.

  time ------------------------------------------>

                +-------------------------------+
  mask:         |0000000001111111111111111111111|
                +-------------------------------+

                +-------------------------------+
  target:       |                              1| 'end-marker' channel.
                |         101100110110011011001 |
                |         010101001010100101010 |
                +-------------------------------+

                +-------------------------------+
  observation:  | 1011001                       |
                | 0101010                       |
                |1                              | 'start-marker' channel
                |        3                      | 'num-repeats' channel.
                +-------------------------------+
  ```

  The length of the random pattern and the number of times it is repeated
  in the target are both discrete random variables distributed according to
  uniform distributions whose parameters are configured at construction time.

  The obs sequence has two extra channels (components in the trailing dimension)
  which are used for flags. One channel is marked with a 1 at the first time
  step and is otherwise equal to 0. The other extra channel is zero until the
  binary pattern to be repeated ends. At this point, it contains an encoding of
  the number of times the observation pattern should be repeated. Rather than
  simply providing this integer number directly, it is normalised so that
  a neural network may have an easier time representing the number of
  repetitions internally. To allow a network to be readily evaluated on
  instances of this task with greater numbers of repetitions, the range with
  respect to which this encoding is normalised is also configurable by the user.

  As in the diagram, the target sequence is offset to begin directly after the
  observation sequence; both sequences are padded with zeros to accomplish this,
  resulting in their lengths being equal. Additional padding is done at the end
  so that all sequences in a minibatch represent tensors with the same shape.
  """

  def __init__(
      self,
      num_bits=6,
      batch_size=1,
      min_length=1,
      max_length=1,
      min_repeats=1,
      max_repeats=2,
      norm_max=10,
      log_prob_in_bits=False,
      time_average_cost=False,
      name='repeat_copy',):
    """Creates an instance of RepeatCopy task.

    Args:
      name: A name for the generator instance (for name scope purposes).
      num_bits: The dimensionality of each random binary vector.
      batch_size: Minibatch size per realization.
      min_length: Lower limit on number of random binary vectors in the
          observation pattern.
      max_length: Upper limit on number of random binary vectors in the
          observation pattern.
      min_repeats: Lower limit on number of times the obervation pattern
          is repeated in targ.
      max_repeats: Upper limit on number of times the observation pattern
          is repeated in targ.
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
    self._num_bits = num_bits
    self._min_length = min_length
    self._max_length = max_length
    self._min_repeats = min_repeats
    self._max_repeats = max_repeats
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
  def num_bits(self):
    """The dimensionality of each random binary vector in a pattern."""
    return self._num_bits

  @property
  def target_size(self):
    """The dimensionality of the target tensor."""
    return 20#self._num_bits + 1

  @property
  def batch_size(self):
    return self._batch_size

  def _build(self):
    """Implements build method which adds ops to graph."""
    print("Hi")
    obs_batch_shape = [24, 24, 20]
    targ_batch_shape = [24, 24, 20]
    mask_batch_trans_shape = [24, 24]
    batch_size=24
    obs_tensors = []
    targ_tensors = []
    mask_tensors = []
    num_bits=8
    # Generates patterns for each batch element independently.
    for batch_index in range(batch_size):
      sub_seq_len = 6

      # The observation pattern is a sequence of random binary vectors.
      obsarr=[
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
      ]

      """,#1
              [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 2
              [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 3
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 4
              [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 5
              [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 6
              [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 7
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  # 8
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # 9
              [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 10
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  # 11
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 12
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 13
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 14
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 15
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
              """

      targarr = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      ]

      """,  # 1
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 2
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 3
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 6
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 7
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 8
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 9
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 10
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 11
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,1,0],  # 12
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 13
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,1],  # 14
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],  # 15
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]"""

      max_items=24 #(<=)
      items_counter=1
      edge_count=random.choice(range(1,4))
      edge_tracker=dict()
      for i in range(0,edge_count):
        v1=random.choice(range(1,17))
        v2 = random.choice(range(1, 17))
        while (v2==v1):
          v2 = random.choice(range(1, 17))
        if v1 in edge_tracker:
          a=edge_tracker[v1]
          a.add(v2)
          edge_tracker[v1]=a
        else:
          a=set()
          a.add(v2)
          edge_tracker[v1] = a
        insert= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        insert[v1-1]=1
        insert[v2-1]=1
        items_counter+=1
        obsarr.append(insert)
        insert2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        insert2[v1 - 1] = 1
        obsarr.append(insert2)
        items_counter += 1
        insert3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        insert3[v2 - 1] = 1
        obsarr.append(insert3)
        items_counter += 1
      obsarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
      items_counter+=1
      obsarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
      items_counter += 1

      q_count = random.choice(range(1, 4))
      for i in range(1, items_counter+q_count+1):
        targarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      targarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
      items_counter2 = items_counter + q_count
      items_counter2 += 1

      maskarr=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      for i in range(items_counter+q_count+1,max_items):
        maskarr[i]=1
      maskarr = np.array(maskarr)

      for i in range(0,q_count):
        insert = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pick=random.choice(list(edge_tracker.keys()))
        insert[pick-1]=1
        items_counter+=1
        obsarr.append(insert)
        for tgt in list(edge_tracker[pick]):
          insert = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          insert[tgt-1] = 1
          items_counter2 += 1
          targarr.append(insert)
      targarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
      obsarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
      items_counter += 1
      items_counter2 += 1




      while (items_counter<max_items):
        obsarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        items_counter+=1
      while (items_counter2<max_items-1):
        targarr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        items_counter2+=1

      obsarr=np.array(obsarr)
      obs = tf.cast(tf.convert_to_tensor(obsarr),tf.float32)

      targarr=np.array(targarr)
      targ = tf.cast(tf.convert_to_tensor(targarr),tf.float32)

      mask = tf.cast(tf.convert_to_tensor(maskarr)
        ,
        tf.float32)

      #print(obsarr)
      #print(targarr)
      #print(maskarr)

      obs_tensors.append(obs)
      targ_tensors.append(targ)
      mask_tensors.append(mask)

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
    unnormalised_num_reps_flag = self._unnormalise(obs[:,:,-1:]).round()
    obs = np.concatenate([obs[:,:,:-1], unnormalised_num_reps_flag], axis=2)
    data = data._replace(observations=obs)
    return bitstring_readable(data, self.batch_size, model_output, whole_batch)
