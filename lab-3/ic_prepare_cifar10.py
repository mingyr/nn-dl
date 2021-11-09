import os
import numpy as np
import tensorflow as tf
import cifar10

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

save_path = 'data/cifar-10'

print("Loading CIFAR10 dataset")
cifar10.extract()

print('beginning prepare CIFAR-10 tfrecords for training')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'cifar-10-train.tfr'))

num_train_records = 0
images, cls = cifar10.load_training_data()

print("images shape {}".format(images.shape))
print("cls shape {}".format(cls.shape))

for image, label in zip(images, cls):
    feature = \
    {
        'image': float_feature(np.reshape(image, [-1])),
        'label': int64_feature(label)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    writer.write(example.SerializeToString())

    num_train_records = num_train_records + 1
    
writer.close()
print('end of tfrecords preparation for training')

print('beginning prepare CIFAR-10 tfrecords for testing')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'cifar-10-test.tfr'))

num_test_records = 0

images, cls = cifar10.load_test_data()
for image, label in zip(images, cls):
    feature = {
        'image': float_feature(np.reshape(image, [-1])),
        'label': int64_feature(label)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    writer.write(example.SerializeToString())

    num_test_records = num_test_records + 1
    
writer.close()

print('end of tfrecords preparation for testing')

print('#tfrecords for training: {}'.format(num_train_records))
print('#tfrecords for testing: {}'.format(num_test_records))

