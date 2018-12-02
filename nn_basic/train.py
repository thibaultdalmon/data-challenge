from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import time
import tensorflow as tf
import model_basic as model
import input as input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train for a number of steps."""
  with tf.Graph().as_default():
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    global_step_init = -1
    global_step = tf.train.get_or_create_global_step()
    if ckpt and ckpt.model_checkpoint_path:
      # This is only for the logger (e.g., it is not responsible for saving).
      global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    dataset = model.inputs(False)
    iterator = dataset.make_one_shot_iterator()
    labels, features = iterator.get_next()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    values = model.inference(features, False)

    # Calculate loss.
    loss = model.loss(values, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = global_step_init
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time
          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction=0.5
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=120,
        config=config) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
    #tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
