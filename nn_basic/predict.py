from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import csv
import math
import pandas as pd

import numpy as np
import tensorflow as tf

import input as input
import model_basic as model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'tmp/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 130,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples',523842 ,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

def eval_once(saver, summary_writer, values, features, summary_op):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    error_mean: mean distance between the input and autoencoder
    error_stdev: standard deviation of the distance
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.1
  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/autoencoder_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      total_sample_count = num_iter * 1
      step = 0
      predictions = np.zeros((FLAGS.num_examples,2))
      while step < num_iter and not coord.should_stop():
        pred_tmp, feat = sess.run([values, features])
        predictions[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size] = pred_tmp
        step += 1
        print("iteration ", step)

      np.savetxt('pred_test.txt', predictions)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels
    eval_data = FLAGS.eval_data == 'test'
    dataset = model.inputs_predict()
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    values = model.inference(features, True)

    max = tf.constant([8.326264e+01, 1.198684e+02])
    mean = tf.constant([4.887722e+01, 2.350651e+00])

    values = tf.math.multiply(values,max) + mean


    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, values, features, summary_op)
      if FLAGS.run_once:
        break
      with tf.device('/cpu:0'):
        time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
