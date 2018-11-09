"""Routine for decoding the CSV file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Global constants describing the CSV data set.
NUM_CLASSES = 1
NUM_LABELS_PER_EXAMPLE = 2

# Speed Optimizations
NUM_PARALLEL_CALLS = 12
NUM_PARALLEL_READERS = 12
PREFETCH_BUFFER_SIZE = 8
SHUFFLE_BUFFER_SIZE = 256


def init_data(eval_data, data_dir, batch_size):
  """Reads and parses examples from CSV data files.
  Args:
    eval_data: A boolean to select the dataset
    data_dir: A string with the directory to read from
    batch_size: An int with the size of a batch
  Returns:
    dataset: A Dataset containing the inputs
  """
  with tf.device('/cpu:0'):
  
    with tf.name_scope('input') as scope:
    
      if not eval_data:
        filenames = [os.path.join(data_dir, 'Train_dataset_%d.csv' % i)
          for i in xrange(0, 11)]
        perform_shuffle = True
      else:
        filenames = [os.path.join(data_dir, 'Train_dataset_%d.csv' % i)
          for i in xrange(11, 12)]
        perform_shuffle = False

      for f in filenames:
        if not tf.gfile.Exists(f):
          raise ValueError('Failed to find file: ' + f)

      with tf.name_scope('input'):

        # Getting filenames from the filenames list, then reads
        # them in parallel with NUM_PARALLEL_READERS
        filenames = tf.data.Dataset.list_files(filenames)

        # Dimensions of the images in the CSV dataset.
        label_bytes = NUM_LABELS_PER_EXAMPLE
        height = 909
        width = 1
        depth = 1
        channel = 1
        image_bytes = height * width * depth * channel - NUM_LABELS_PER_EXAMPLE
        
        # One header CSV format, so we set skip to 1
        dataset = filenames.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TextLineDataset(filename).skip(1),
                cycle_length=NUM_PARALLEL_READERS))

        # Decodes a csv line and return the tensors corresponsing
        # to the training data and the label
        def decode_csv(line):

          # Every record consists of a label followed by an example
          record = ([[0.]]*(label_bytes+27) + 
                    [[0]] * (image_bytes-27))
          record = tf.decode_csv(line, record)
          for k in range(len(record)):
            record[k] = tf.cast(record[k], dtype=tf.float32, name=None)
          
          # The first bytes represent the label,
          values = tf.reshape(
            tf.strided_slice(record, [label_bytes],
                               [image_bytes+label_bytes]),
            [image_bytes])

          labels = tf.reshape(
            tf.strided_slice(record, [0],
                               [label_bytes]),
            [NUM_LABELS_PER_EXAMPLE])

          return values, values

        # Then maps CSV to tensors with NUM_PARALLEL_CALLS threads
        # Finally caches the obtained dataset into memory
        dataset = dataset.map(map_func=decode_csv,
                        num_parallel_calls=NUM_PARALLEL_CALLS)

        if perform_shuffle:
           # Randomizes input using a window of 256 elements (read into memory)
           dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat() # Repeats dataset
        # Ensures that all batches have the same size
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        # Prepares the next input even if previous are still not used
        dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

  return dataset

def info_data(data_dir, batch_size, bases_idx=bases_idx, bases_id=bases_id):
  """Reads and parses examples from CSV data files.
  Args:
    eval_data: A boolean to select the dataset
    data_dir: A string with the directory to read from
    batch_size: An int with the size of a batch
  Returns:
    dataset: A Dataset containing the inputs
  """
  with tf.device('/cpu:0'):
  
    advanced_dataset = Dataset()
  
    with tf.name_scope('input') as scope:
    
      for id in range(len(bases_id)):
    
        filenames = [os.path.join(data_dir, 'Bases_dataset_%d.csv' % bases_id[id])]

        for f in filenames:
          if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

        with tf.name_scope('input'):

          # Getting filenames from the filenames list, then reads
          # them in parallel with NUM_PARALLEL_READERS
          filenames = tf.data.Dataset.list_files(filenames)

          # Dimensions of the images in the CSV dataset.
          size = 120
          
          # One header CSV format, so we set skip to 1
          dataset = filenames.apply(
              tf.contrib.data.parallel_interleave(
                  lambda filename: tf.data.TextLineDataset(filename).skip(1).range(bases_idx[id],bases_idx[id+1]),
                  cycle_length=NUM_PARALLEL_READERS))

          # Decodes a csv line and return the tensors corresponsing
          # to the training data and the label
          def decode_csv(line):

            # Every record consists of a label followed by an example
            record = ([[0.]]*(label_bytes+27) + 
                      [[0]] * (image_bytes-27))
            record = tf.decode_csv(line, record)
            for k in range(len(record)):
              record[k] = tf.cast(record[k], dtype=tf.float32, name=None)
            
            # The first bytes represent the label,
            values = tf.reshape(
              tf.strided_slice(record, [label_bytes],
                                 [image_bytes+label_bytes]),
              [image_bytes])

            labels = tf.reshape(
              tf.strided_slice(record, [0],
                                 [label_bytes]),
              [NUM_LABELS_PER_EXAMPLE])

            return values, values

          # Then maps CSV to tensors with NUM_PARALLEL_CALLS threads
          # Finally caches the obtained dataset into memory
          dataset = dataset.map(map_func=decode_csv,
                          num_parallel_calls=NUM_PARALLEL_CALLS)
                          
          # Add the lines generated to generate a batch
          advanced_dataset.concatenate(dataset)


  return advanced_dataset