import numpy as np
import tensorflow as tf
import sonnet as snt
from functools import partial


class LRModel(snt.Module):
    def __init__(self, name = "lr_model"):
        super(LRModel, self).__init__(name = name)    
        self._h1 = snt.Linear(16, name = "hidden_layer_1")
        self._h1_relu = partial(tf.nn.relu, name="hidden_layer_1_relu")
        self._h2 = snt.Linear(8, name = "hidden_layer_2")
        self._h2_relu = partial(tf.nn.relu, name = "hidden_layer_2_relu")
        self._out = snt.Linear(1, name = "output_layer")
            
    def __call__(self, x):
        y = self._h1(x)
        y = self._h1_relu(y)

        y = self._h2(y)
        y = self._h2_relu(y)

        y = self._out(y)
        
        return y

        
if __name__ == "__main__":
    import os
    
    log_dir = "output-model"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    input_spec = [tf.TensorSpec(shape=[None, 1], dtype=tf.float32)]
    model = tf.function(LRModel(), input_signature=input_spec)
    x = tf.random.uniform((32, 1), dtype=tf.float32) * 4
    y = model(x)
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        tf.summary.graph(model.get_concrete_function().graph)