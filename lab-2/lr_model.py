import numpy as np
import tensorflow as tf
import sonnet as snt

class LRModel(snt.AbstractModule):
    def __init__(self, name = "lr_model"):
        super(LRModel, self).__init__(name = name)
        
        with self._enter_variable_scope():
            self._h1 = snt.Linear(16, name = "hidden_layer_1")
            self._h2 = snt.Linear(8, name = "hidden_layer_2")
            self._out = snt.Linear(1, name = "output_layer")
            
    def _build(self, x):
        y = tf.nn.relu(self._h1(x))
        y = tf.nn.relu(self._h2(y))
        y = tf.nn.relu(self._out(y))
        
        return y
        
if __name__ == "__main__":
    import os
    
    if not os.path.exists("output-model"):
        os.makedirs("output-model")
    
    x = tf.random.uniform((32, 1)) * 4
    lr_model = LRModel()
    y = lr_model(x)
    
    writer = tf.summary.FileWriter('output-model', tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_val, y_val = sess.run([x, y])
        print("x -> {}".format(x_val))
        print("y -> {}".format(y_val))
        
    writer.close()

        