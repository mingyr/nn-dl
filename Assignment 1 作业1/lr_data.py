import os
import numpy as np
import tensorflow as tf
import tfmpl

log_dir = "output-data"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 生成区间[0, 4]之内的数据
# y_d表示真实模型的数据，通常我们并不知道；
# y表示带噪声或实际问题中的数据，我们经过采集或采样得到；
x = tf.random.uniform((100, 1)) * 4 
y_d = 1.6 * x + 0.4
y = y_d + tf.random.normal(tf.shape(x), stddev=0.1)

pts = tf.concat([x, y], axis = -1)

@tfmpl.figure_tensor
def draw_scatter(scaled, colors): 
    '''Draw scatter plots. One for each color.'''  
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    for i,c in enumerate(colors):
        ax.scatter(scaled[:, 0], scaled[:, i+1], c=c)
    fig.tight_layout()
    return fig

summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default():
    image_tensor = draw_scatter(pts, ['r'])
    image_summary = tf.summary.image("images", image_tensor, step=0)    
summary_writer.close()
