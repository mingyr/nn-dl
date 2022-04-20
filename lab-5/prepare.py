import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

save_path = 'data/MNIST'

print("Loading MNIST dataset")
data = input_data.read_data_sets("data/MNIST/", one_hot=False)

print('beginning prepare MNIST tfrecords for training')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'mnist-train.tfr'))

num_train_records = 0

for image, label in zip(data.train.images, data.train.labels):
    feature = \
    {
        'image': float_feature(image),
        'label': int64_feature(label)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    writer.write(example.SerializeToString())

    num_train_records = num_train_records + 1
    
writer.close()
print('end of tfrecords preparation for training')

print('beginning prepare MNIST tfrecords for testing')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'mnist-test.tfr'))

num_test_records = 0

for image, label in zip(data.test.images, data.test.labels):
    feature = {
        'image': float_feature(image),
        'label': int64_feature(label)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    writer.write(example.SerializeToString())

    num_test_records = num_test_records + 1
    
writer.close()

print('end of tfrecords preparation for testing')

print('#tfrecords for training: {}'.format(num_train_records))
print('#tfrecords for testing: {}'.format(num_test_records))

