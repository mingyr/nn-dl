import os
import numpy as np
import tensorflow as tf
import tfmpl

if not os.path.exists("output-data"):
    os.makedirs("output-data")

x = tf.random.uniform((100, 1)) * 4
y_ = 1.6 * x + 0.4
y = y_ + tf.random.normal(tf.shape(x), stddev = 0.25)

pts = tf.concat([x, y_, y], axis = -1)

@tfmpl.figure_tensor
def draw_scatter(scaled, colors): 
    '''Draw scatter plots. One for each color.'''  
    fig = tfmpl.create_figures(1, figsize = (4, 4))[0]
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[0])
    ax.scatter(scaled[:, 0], scaled[:, 2], c=colors[1])
    fig.tight_layout()
    return fig

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image_tensor = draw_scatter(pts, ['r', 'b'])
    image_summary = tf.summary.image('scatter', image_tensor)      
    all_summaries = tf.summary.merge_all() 
    
    writer = tf.summary.FileWriter('output-data', sess.graph)
    summary = sess.run(all_summaries)
    writer.add_summary(summary, global_step = 0)
