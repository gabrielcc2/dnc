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
"""Example script to train the DNC on the one hop task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

from dnc import dnc
from dnc import one_hop_task
from tensorboard import Tensorboard

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 30, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 20, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 2, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
#tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.") #Unused
#tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
#                      "Epsilon used for RMSProp optimizer.") #Unused

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("word_length", 20, "Overall size of observation (# possible vertices + start and end markers).")
tf.flags.DEFINE_integer("max_items", 32, "Overall length of sequence.")
tf.flags.DEFINE_integer("max_edges", 10, "Max number of edges in input sequence (up to 10 are currently supported).")
tf.flags.DEFINE_integer("max_questions", 6, "Max number of questions in input sequence (up to 6 currently supported).")
tf.flags.DEFINE_bool("curriculum_learning", True, "Whether to use a curriculum learning or not.")
tf.flags.DEFINE_string("curriculum_strategy", "interleaved", "Kind of strategy, either regular or interleaved")
tf.flags.DEFINE_float("curriculum_loss_threshold", 2.0, "Lower bound on the loss, that signals that we can increase the difficulty of the task")


# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 1000000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 5000,
                        "Checkpointing step interval.")

def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence


def train(num_training_iterations, report_interval):
  """Trains the DNC and periodically reports the loss."""

  tensorboard = Tensorboard(FLAGS.checkpoint_dir)
  dataset = []
  if not FLAGS.curriculum_learning:
    dataset=one_hop_task.OneHop(FLAGS.batch_size,
                                   FLAGS.word_length, FLAGS.max_items,
                                   FLAGS.max_edges, FLAGS.max_questions)
  else:
    dataset = one_hop_task.OneHop(FLAGS.batch_size,
                                    FLAGS.word_length, FLAGS.max_items,
                                    1, 1)

  dataset_tensors = dataset()
  output_logits = run_model(dataset_tensors.observations, dataset.word_length)
  output = tf.round(
      tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))

  train_loss = dataset.cost(output_logits, dataset_tensors.target,
                            dataset_tensors.mask)

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.compat.v1.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])


  optimizer = tf.compat.v1.train.AdamOptimizer()#Note we commented out the use of RMSProp... RMSPropOptimizer(#FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

  saver = tf.compat.v1.train.Saver()

  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.compat.v1.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  # Train.
  with tf.compat.v1.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
    update=0
    start_iteration = sess.run(global_step)
    total_loss = 0
    update_edges=True

    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss

      if (train_iteration + 1) % report_interval == 0:
        dataset_tensors_np, output_np = sess.run([dataset_tensors, output])
        dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                   output_np)
        tf.compat.v1.logging.info("%d: Avg training loss %f. Max Questions: %f.  Max Edges: %f.\n%s",
                        train_iteration, total_loss / report_interval, dataset._max_questions, dataset._max_edges,
                        dataset_string)
        tensorboard.log_summary(
            total_loss / report_interval, dataset._max_edges, dataset._max_questions, train_iteration)

        if FLAGS.curriculum_learning:
          if FLAGS.curriculum_strategy == "regular":
            if dataset._max_edges >= FLAGS.max_edges:
                if (total_loss / report_interval) < FLAGS.curriculum_loss_threshold and (
                        train_iteration - update) > 500 and dataset._max_questions < FLAGS.max_questions:
                    dataset._max_questions += 1
                    update = train_iteration
            elif (total_loss / report_interval) < FLAGS.curriculum_loss_threshold and (
                    train_iteration - update) > 500 and dataset._max_edges < FLAGS.max_edges:
                dataset._max_edges += 1
                update = train_iteration
          elif FLAGS.curriculum_strategy=="interleaved":
            if update_edges and (total_loss / report_interval) < FLAGS.curriculum_loss_threshold and (train_iteration - update) > 500 and dataset._max_edges < FLAGS.max_edges:
              dataset._max_edges += 1
              update = train_iteration
              update_edges=False
              if dataset._max_questions>=FLAGS.max_questions:
                update_edges=True
            elif (total_loss/report_interval) < FLAGS.curriculum_loss_threshold and (train_iteration - update) > 500 and dataset._max_questions<FLAGS.max_questions and not update_edges:
              update_edges=True
              dataset._max_questions += 1
              update = train_iteration
        total_loss = 0

def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.compat.v1.app.run()
