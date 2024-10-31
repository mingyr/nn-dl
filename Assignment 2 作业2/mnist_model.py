import os
import numpy as np
import tensorflow as tf
import sonnet as snt

class Pooling(snt.Module):
    def __init__(self, pool=None, k=2, padding='SAME', name="pooling"):
        super(Pooling, self).__init__(name=name)
        self._pool = pool        
        self._k = k
        self._padding = padding

    def __call__(self, x):
        if self._pool == 'max':
            return tf.nn.max_pool2d(x, self._k, self._k, self._padding)
        elif pool == 'avg':
            return tf.nn.avg_pool2d(x, self._k, self._k, self._padding)
        else:
            return lambda x: x

class Model(snt.Module):
    def __init__(self, num_classes, filter_size=5, name="model"):
        super(Model, self).__init__(name=name)
        self._num_classes = num_classes
        self._filter_size = filter_size
    
    @snt.once
    def _initialize(self):    
        self._conv1 = snt.Conv2D(32, self._filter_size, name="first_conv_layer")
        self._pool1 = Pooling('max', name="first_max_pool_layer")
        
        self._conv2 = snt.Conv2D(64, self._filter_size, name="second_conv_layer")
        self._pool2 = Pooling('max', name="second_pool_layer")            


        self._lin = snt.Linear(256, name="fully_conn_layer")
        self._output = snt.Linear(self._num_classes, name="output_layer")
            
    def __call__(self, x):
        self._initialize()
        
        y = tf.nn.relu(self._conv1(x))
        y = self._pool1(y)
        
        y = tf.nn.relu(self._conv2(y))
        y = self._pool2(y)
        

        y = snt.Flatten()(y)
        
        y = tf.nn.relu(self._lin(y))
        
        return self._output(y)

def test():
    log_dir = "output-model"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    x = tf.random.normal([32, 28, 28, 1])        
    model = tf.function(Model(10))

    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=False)
    y = model(x)
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)
    tf.summary.trace_off()
        
if __name__ == "__main__":
    test()        
        
