import os
import numpy as np
import tensorflow as tf

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

save_path = r'data\mnist'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Loading MNIST dataset")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print('beginning prepare MNIST tfrecords for training')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'mnist-train.tfr'))

for i, (image, label) in enumerate(zip(x_train, y_train)):
    print(f"Processing the {i}-th sample ...")
    feature = \
    {
        'image': float_feature(image.flatten()),
        'label': int64_feature(tf.squeeze(label))
    }
    example = tf.train.Example(features = tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

num_train_records = i + 1    
writer.close()
print('end of tfrecords preparation for training')

print('beginning prepare MNIST tfrecords for testing')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'mnist-test.tfr'))

for i, (image, label) in enumerate(zip(x_test, y_test)):
    print(f"Processing the {i}-th sample ...")
    feature = {
        'image': float_feature(image.flatten()),
        'label': int64_feature(tf.squeeze(label))
    }
    example = tf.train.Example(features = tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

num_test_records = i + 1    
writer.close()
print('end of tfrecords preparation for testing')

print('#tfrecords for training: {}'.format(num_train_records))
print('#tfrecords for testing: {}'.format(num_test_records))

