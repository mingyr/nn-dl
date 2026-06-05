import numpy as np
import tensorflow as tf
import sonnet as snt

class LRModel(snt.Module):
    def __init__(self, name = "lr_model"):
        super(LRModel, self).__init__(name = name)
    
    @snt.once
    def _initialize(self):
        self._h1 = snt.Linear(16, name = "hidden_layer_1")
        self._h2 = snt.Linear(8, name = "hidden_layer_2")

        self._out = snt.Linear(1, name = "output_layer")
            
    def __call__(self, x):
        self._initialize()
        y = tf.nn.relu(self._h1(x))
        y = tf.nn.relu(self._h2(y))

        y = self._out(y)
        
        return y

        
if __name__ == "__main__":
    import os
    
    log_dir = "output-model"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    model = tf.function(LRModel())
    x = tf.random.uniform((32, 1)) * 4
    summary_writer = tf.summary.create_file_writer(log_dir)

    tf.summary.trace_on(graph=True, profiler=False)
    y = model(x)
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)
    tf.summary.trace_off()
