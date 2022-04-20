import os
import numpy as np
import tensorflow as tf
import sonnet as snt
from utils import Pooling


class MaxPool(snt.AbstractModule):
    def __init__(self, k = 2, padding = 'SAME', name = "max_pool"):
        super(MaxPool, self).__init__(name = name)
        self._k = k
        self._padding = padding

    def _build(self, inputs):
        return tf.nn.max_pool2d(inputs, self._k, self._k, self._padding)
        
        
class Encoder(snt.AbstractModule):
    def __init__(self, filter_size=5, hidden_size_multiplier=4, name="encoder"):
        super(Encoder, self).__init__(name=name)
        with self._enter_variable_scope():
            
            self._conv1 = snt.Conv2D(32, filter_size, stride=1, name="first_conv_layer")            
            self._conv2 = snt.Conv2D(64, filter_size, stride=1, name="second_conv_layer")
            self._lin = snt.Linear(49*hidden_size_multiplier, name = "fully_conn_layer")
            
    def _build(self, x):
            y = tf.nn.relu(self._conv1(x))
            y = MaxPool()(y)
            
            y = tf.nn.relu(self._conv2(y))
            y = MaxPool()(y)
            
            y = snt.BatchFlatten()(y)
            y = tf.math.sigmoid(self._lin(y))

            return y

class Decoder(snt.AbstractModule):
    def __init__(self, filter_size=5, name="decoder"):
        super(Decoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._conv1trans = snt.Conv2DTranspose(64, kernel_shape=filter_size, stride=2, name="first_conv_trans_layer")            
            self._conv2trans = snt.Conv2DTranspose(32, kernel_shape=filter_size, stride=2, name="second_conv_trans_layer")
            self._conv3 = snt.Conv2D(1, 1, name="third_conv_layer")
            
    def _build(self, x):    
            y = snt.BatchReshape((7, 7, -1))(x)
            y = tf.nn.relu(self._conv1trans(y))
            y = tf.nn.relu(self._conv2trans(y))
            y = tf.math.sigmoid(self._conv3(y))

            return y

def test():
    x = tf.random_normal([32, 28, 28, 1], name="x")
    
    encoder = Encoder()
    y = encoder(x)
    
    decoder = Decoder()
    r = decoder(y)
    
    if not os.path.exists("output-model"):
        os.makedirs("output-model")
        
    writer = tf.summary.FileWriter("output-model")
    
    with tf.Session() as sess:
        writer.add_graph(sess.graph)
    
        sess.run([tf.global_variables_initializer(), 
                  tf.local_variables_initializer()])
                  
        sess.run(y)
        
        writer.close()
        
if __name__ == "__main__":
    test()        
        
        