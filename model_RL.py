# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import urllib

import os
import re
import sys
import tarfile
import tensorflow as tf
import input as input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../Data/Train/Tensorflow',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing data set.
NUM_CLASSES = input.NUM_CLASSES
NUM_LABELS_PER_EXAMPLE = input.NUM_LABELS_PER_EXAMPLE


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
GROWTH_RATE = 2

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  with tf.name_scope('') as scope:
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x, name=tensor_name+'/zero_fraction'))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  #with tf.device('/cpu:0'):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inputs(eval_data, bases_info, bases_idx=[], bases_id=[]):
  """Construct input
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    bases_info: indicating if one should load info on basetations.
    bases_idx: indicating which basetations should be loaded.
  Returns:
    dataset: the dataset processed for evaluation
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  
  if bases_info:
    dataset = input.info_data(data_dir=FLAGS.data_dir,
                              batch_size=FLAGS.batch_size
                              bases_idx=bases_idx
                              bases_idx=bases_id)
  else:
    dataset = input.init_data(eval_data=eval_data,
                              data_dir=FLAGS.data_dir,
                              batch_size=FLAGS.batch_size)
  return dataset


def inference_reward(images, eval):
  """Build the scoring model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    values: input prediction.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  
  #if not eval:
  #  images = tf.contrib.nn.alpha_dropout(images, 0.8)

  with tf.name_scope("init") as scope:
    with tf.name_scope("dense_1") as scope:
      in_fully = tf.layers.batch_normalization(images)
      acti_fully = tf.nn.selu(in_fully, 'selu')
      _activation_summary(acti_fully)
      out_fully = tf.layers.dense(
            acti_fully,
            units=1024,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2))
      out_fully = tf.reshape(out_fully, [FLAGS.batch_size, 1024])
    with tf.name_scope("dense_2") as scope:
      images = tf.layers.batch_normalization(out_fully, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[1024,512],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [512], tf.constant_initializer(0.0))
      out_dense_2 = tf.add(tf.matmul(images,kernel), biases)
    with tf.name_scope("dense_3") as scope:
      images = tf.layers.batch_normalization(out_dense_2, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[512,256],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [256], tf.constant_initializer(0.0))
      out_dense_3 = tf.add(tf.matmul(images,kernel), biases)
    with tf.name_scope("dense_4") as scope:
      images = tf.layers.batch_normalization(out_dense_3, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[256,512],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [512], tf.constant_initializer(0.0))
      out_dense_4 = tf.add(tf.matmul(images,kernel), biases)
    with tf.name_scope("dense_5") as scope:
      images = tf.layers.batch_normalization(out_dense_4, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[512,1024],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [1024], tf.constant_initializer(0.0))
      out_dense_5 = tf.add(tf.matmul(images,kernel), biases)

  with tf.name_scope('predict') as scope:

    pre_activation = tf.layers.batch_normalization(out_dense_5, name=scope + 'batch_normalization')
    pre_activation = tf.nn.selu(pre_activation, 'selu')
    _activation_summary(pre_activation)
    
    #if not eval:
    #  pre_activation = tf.contrib.nn.alpha_dropout(pre_activation, 0.8)
      
    with tf.name_scope('logits') as scope:
      weights = _variable_with_weight_decay(scope + 'weights', shape=[1024,907],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu(scope + 'biases', [907], tf.constant_initializer(0.1))
      values = tf.matmul(pre_activation, weights) + biases
      _activation_summary(values)
    
    
  return values
  
def inferenceRL(images, eval):
  """Build the RL model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    values: input prediction.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  
  #if not eval:
  #  images = tf.contrib.nn.alpha_dropout(images, 0.8)

  with tf.name_scope("init") as scope:
    with tf.name_scope("dense_1") as scope:
      in_fully = tf.layers.batch_normalization(images)
      acti_fully = tf.nn.selu(in_fully, 'selu')
      _activation_summary(acti_fully)
      out_fully = tf.layers.dense(
            acti_fully,
            units=1024,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2))
      out_fully = tf.reshape(out_fully, [FLAGS.batch_size, 1024])
    with tf.name_scope("dense_2") as scope:
      images = tf.layers.batch_normalization(out_fully, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[1024,512],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [512], tf.constant_initializer(0.0))
      out_dense_2 = tf.add(tf.matmul(images,kernel), biases)
    with tf.name_scope("dense_3") as scope:
      images = tf.layers.batch_normalization(out_dense_2, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[512,256],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [256], tf.constant_initializer(0.0))
      out_dense_3 = tf.add(tf.matmul(images,kernel), biases)
    with tf.name_scope("dense_4") as scope:
      images = tf.layers.batch_normalization(out_dense_3, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[256,512],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [512], tf.constant_initializer(0.0))
      out_dense_4 = tf.add(tf.matmul(images,kernel), biases)
    with tf.name_scope("dense_5") as scope:
      images = tf.layers.batch_normalization(out_dense_4, name=scope + 'batch_normalization')
      images = tf.nn.selu(images, 'selu')
      _activation_summary(images)
      kernel = _variable_with_weight_decay(scope + 'weights',
                                           shape=[512,1024],
                                           stddev=5e-2,
                                           wd=None)
      biases = _variable_on_cpu(scope + 'biases', [1024], tf.constant_initializer(0.0))
      out_dense_5 = tf.add(tf.matmul(images,kernel), biases)

  with tf.name_scope('predict') as scope:

    pre_activation = tf.layers.batch_normalization(out_dense_5, name=scope + 'batch_normalization')
    pre_activation = tf.nn.selu(pre_activation, 'selu')
    _activation_summary(pre_activation)
    
    #if not eval:
    #  pre_activation = tf.contrib.nn.alpha_dropout(pre_activation, 0.8)
      
    with tf.name_scope('logits') as scope:
      weights = _variable_with_weight_decay(scope + 'weights', shape=[1024,907],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu(scope + 'biases', [907], tf.constant_initializer(0.1))
      bases_idx = tf.matmul(pre_activation, weights) + biases
      _activation_summary(values)
    
    
  return bases_idx, bases_id

def loss(values, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "losses"
  Args:
    values: Values from inference().
    labels: Input from inference().
  Returns:
    Loss tensor of type float.
  """

  ## Calculate the average cross entropy loss across the batch.
  l2 = tf.nn.l2_loss(tf.subtract(labels, values))
  l2_mean = tf.divide(l2, tf.to_float(FLAGS.batch_size), name='l2')

  tf.add_to_collection('losses', l2_mean)

  # The total loss is defined as the l2 loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  with tf.name_scope("hide_all_conds"):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step):
  """Train autoencoder model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(learning_rate=0.001,)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op