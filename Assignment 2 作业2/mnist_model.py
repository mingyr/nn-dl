import os
import numpy as np
import tensorflow as tf
import sonnet as snt
from functools import partial

class Model(snt.Module):
    def __init__(self, num_classes, filter_size=5, name="model"):
        super(Model, self).__init__(name=name)
       
        self._conv1 = snt.Conv2D(32, filter_size, name="conv1")
        self._relu1 = partial(tf.nn.relu, name="relu1")
        self._pool1 = partial(tf.nn.max_pool2d, ksize=2, strides=2, 
                              padding="SAME", name="pool1")

        self._conv2 = snt.Conv2D(64, filter_size, name="conv2")
        self._relu2 = partial(tf.nn.relu, name="relu2")
        self._pool2 = partial(tf.nn.max_pool2d, ksize=2, strides=2, 
                              padding="SAME", name="pool2")

        self._lin = snt.Linear(256, name="lin")
        self._relu3 = partial(tf.nn.relu, name="relu3")
        self._output = snt.Linear(num_classes, name="output")
            
    def __call__(self, x):        
        y = self._conv1(x)
        y = self._relu1(y)
        y = self._pool1(y)
        
        y = self._conv2(y)
        y = self._relu2(y)
        y = self._pool2(y)
        
        y = snt.Flatten(name="flatten")(y)
        
        y = self._lin(y)
        y = self._relu3(y)
        
        return self._output(y)

def test():
    log_dir = "output-model"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    input_spec = [tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)]
    model = tf.function(Model(10), input_signature=input_spec)

    x = tf.random.normal([32, 28, 28, 1])        
    y = model(x)
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        tf.summary.graph(model.get_concrete_function().graph)
    
        
if __name__ == "__main__":
    test()        
        
