import os
import numpy as np
import tensorflow as tf
import sonnet as snt

class Pooling(snt.AbstractModule):
    def __init__(self, pool = None, k = 2, padding = 'SAME', name = "pooling"):
        super(Pooling, self).__init__(name = name)
        self._pool = pool        
        self._k = k
        self._padding = padding

    def _build(self, x):
        if self._pool == 'max':
            return tf.nn.max_pool2d(x, self._k, self._k, self._padding)
        elif pool == 'avg':
            return tf.nn.avg_pool2d(x, self._k, self._k, self._padding)
        else:
            return lambda x: x

class Model(snt.AbstractModule):
    def __init__(self, num_classes, filter_size = 5, name = "model"):
        super(Model, self).__init__(name = name)
        
        with self._enter_variable_scope():
            self._conv1 = snt.Conv2D(32, filter_size, name = "first_conv_layer")
            self._pool1 = Pooling('max', name = "first_max_pool_layer")
            
            self._conv2 = snt.Conv2D(64, filter_size, name = "second_conv_layer")
            self._pool2 = Pooling('max', name = "second_pool_layer")
            
            self._lin = snt.Linear(256, name = "fully_conn_layer")
            self._output = snt.Linear(num_classes, name = "output_layer")
            
    def _build(self, x):
            y = tf.nn.relu(self._conv1(x))
            y = self._pool1(y)
            
            y = tf.nn.relu(self._conv2(y))
            y = self._pool2(y)
            
            y = snt.BatchFlatten()(y)
            
            y = tf.nn.relu(self._lin(y))
            
            return self._output(y)

def test():
    x = tf.random_normal([32, 28, 28, 1], name="x")
    
    model = Model(10)
    y = model(x)
    
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
        
        