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

import pandas as pd
import networkx as nx

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
  This is how vertices are encoded:
           1:         00001
           2:         00010
           3:         00011
           4:         00100
           5:         00101
           6:         00110
           7:         00111
           8:         01000
           9:         01001
          10:         01010
  At the moment only 10 vertices are supported.
  The observation space contains on the first rows edges, where each
  contains of one source vertex and one source edge.
  e.g. 0000100010
  After the question marker, which indicates the number of questions,
  the observation space includes a series of source vertices as questions..
  As answers we will have the target vertices.
  The target sequence is comprised of the answers (the matching vertices),in
  the top right section.
  Here's an example:
  ```none
  Note: blank space represents 0.
  time ------------------------------------------>
                +----------------------------------+
  mask:         |0000000000011111111111111111111111|
                +----------------------------------+
                +----------------------------------+
  target:       |           00010                  | 2 (1->2)
                |           00011                  | 3 (1->3)  When a single vertex will match to multiple
                |           00001                  | 1 (2->1)  the order of insertion is preserved
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                +----------------------------------+
                +----------------------------------+
  observation:  | 0000100010                       | 1->2
                | 0000100011                       | 1->3
                | 0101001001                       | 10->9
                | 0001000001                       | 2->1
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  |
                |                                  | (up to max edges: 10)
                |1                                 | 'start-marker' channel
                |           2                      | 'number of questions' channel.
                | 00001                            |
                | 00010                            |
                |                                  |
                |                                  |
                |                                  |
                |                                  |  (up to max questions: 6)
                +----------------------------------+
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
            name='one_hop', ):
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

    def my_func(arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        # return tf.matmul(arg, arg) + arg
        return arg

    def _input_generator(self):

        df1 = np.zeros((self._max_items, self.word_length))
        answers1 = np.zeros((self._max_items, self.word_length))
        questions1 = np.zeros((self._max_items, 1))

        for w in range(1):

            data = 'FacebookDegreeDistribution'
            number_of_questions = 1

            df = pd.read_csv(data + '.csv', delimiter='|')
            df.drop('creationDate', axis=1, inplace=True)

            df = df.drop(df.index[2000:180623])

            distinc_id_1 = df.groupby('Person.id')
            ids_1 = list(distinc_id_1.groups)

            new_ids_1 = {}
            for i in range(len(ids_1)):
                new_ids_1[ids_1[i]] = i

            distinc_id_2 = df.groupby('Person.id.1')
            ids_2 = list(distinc_id_2.groups)

            new_ids_2 = {}
            for i in range(len(ids_2)):
                new_ids_2[ids_2[i]] = i

            df['Person.id'] = df['Person.id'].apply(lambda x: '{:014b}'.format(new_ids_1[x]))
            df['Person.id.1'] = df['Person.id.1'].apply(lambda x: '{:014b}'.format(new_ids_2[x]))

            # ---------------------------------------------------------------------

            G = nx.Graph()

            all_edges = list(zip(df['Person.id'], df['Person.id.1']))
            for i in all_edges:
                G.add_edge(*i)

            # ---------------------------------------------------------------------
            questions = []

            for i in range(number_of_questions):
                sample_1 = df.sample()
                sample_2 = df.sample()

                source = sample_2['Person.id'].values[0]
                destination = sample_1['Person.id.1'].values[0]
                questions.append([source, destination])

            # ---------------------------------------------------------------------
            answers = []
            for item in questions:
                shortest_path = nx.shortest_path(G, item[0], item[1])
                answers.append(shortest_path)

            # obs_pattern_shape = [sub_seq_len, num_bits]
            # df = tf.reshape(tf.concat(df, 1), obs_batch_shape)

            output = {
                'network': df,
                'questions': questions,
                'answers': answers
            }

            dim1 = len(questions)
            dim2 = len(questions[0])
            dimensions = {'rows': dim1,
                          'columns': dim2
                          }
            x = []
            v = []
            l = []
            # a=np.zeros((dim2),dtype=np.float32)
            # print(questions)
            for i in range(dim1):
                for j in range(dim2):
                    v = questions[i][j].split(',')
                    y = [float(v[0])]
                    l.append(y)
                x.append(l)
                l = []
                # l.clear()

            #    y= questions(map(float, string.split(',')))
            # questions = map(float, questions)
            # valu = strconv.ParseFloat(question[0], 32)
            # float = float32(valu)
            # a=np.zeros((dim1,dim2),dtype=np.float32)
            # for i in questions:
            #       a[i,:]=map(np.float32,questions[i])
            #       a[i,:]=line.split()

            # output3 = {'que': questions,
            #       'flque': x
            # }

            df.to_csv(r'MoeZipfDistribution_binary.csv', index=None, header=True)
            answers = np.array(answers)
            questions = np.array(questions)
            df = np.array(df)

            df1[:df.shape[0], :df.shape[1]] = df[:df1.shape[0], :df1.shape[1]]
            questions1[:questions.shape[0], :questions.shape[1]] = questions[:questions1.shape[0], :questions1.shape[1]]
            # print("inside _input_generator - answers.shape = " + str(answers.shape) + "answers1.shape = " + str(answers1.shape))
            try:
                # print("inside _input_generator try - answers.shape = " + str(answers.shape) + "answers1.shape = " + str(answers1.shape))
                # print("answers type = " + str(answers.dtype) + "  answers1.type = " + str(answers1.dtype))
                answers1[:answers.shape[0], :answers.shape[1]] = answers[:answers1.shape[0], :answers1.shape[1]]
            except:
                answers = answers.reshape((answers.shape[0], 1))
                answers.fill(0)
                # print("inside _input_generator except - answers.shape = " + str(answers.shape) + "answers1.shape = " + str(answers1.shape))
                # print("answers type = " + str(answers.dtype) + "  answers1.type = " + str(answers1.dtype))
                answers1[:answers.shape[0], :answers.shape[1]] = answers[:answers1.shape[0], :answers1.shape[1]]

            questions1 = questions1.reshape((questions1.shape[0],))
            answers1 = np.array(answers1, dtype=np.float32)
            questions1 = np.array(questions1, dtype=np.float32)
            df1 = np.array(df1, dtype=np.float32)
            # output2 = my_func(df)

            # return output2

            yield df1, answers1, questions1

    def _build(self):
        """Implements build method which adds ops to graph."""
        obs_batch_shape = [self._max_items, self._batch_size, self._word_length]
        targ_batch_shape = [self._max_items, self._batch_size, self._word_length]
        mask_batch_trans_shape = [self._batch_size, self._max_items]
        obs_tensors = []
        targ_tensors = []
        mask_tensors = []

        # Generates patterns for each element independently.
        dataset = tf.data.Dataset.from_generator(self._input_generator, (tf.float32, tf.float32, tf.float32), (
        tf.TensorShape([self._max_items, self._word_length]), tf.TensorShape([self._max_items, self._word_length]),
        tf.TensorShape([self._max_items])))
        dataset = dataset.repeat(count=-1)  # An infinite repeat of the generator...
        print("dataset_inside_build point 1= " + str(dataset))
        for batch_index in range(self._batch_size):
            iterator = dataset.make_one_shot_iterator()
            x, y, z = iterator.get_next()
            obs_tensors.append(x)
            targ_tensors.append(y)
            mask_tensors.append(z)

        print("dataset_inside_build point 2= " + str(obs_tensors))
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

    def to_human_readable_test(self, data, model_output=None, whole_batch=False):
        obs = data.observations
        data = data._replace(observations=obs)
        return bitstring_readable(data, self.batch_size, model_output, whole_batch)

    def to_human_readable(self, data, model_output=None, whole_batch=False):
        obs = data.observations
        data = data._replace(observations=obs)
        return bitstring_readable(data, self.batch_size, model_output, whole_batch)
