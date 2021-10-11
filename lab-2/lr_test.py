import os
import numpy as np
import tensorflow as tf
import tfmpl

from lr_model import LRModel 

if not os.path.exists("output-test"):
    os.makedirs("output-test")
    
if not os.path.exists("output-train"):
    raise ValueError("Non-existing output-train folder")

# 训练好的模型的保存路径
checkpoint_dir = "output-train"

# 生成测试数据
x = tf.random.uniform((100, 1), dtype=tf.float64) * 4 + 2.0
y_ = 1.6 * x + 0.4


# 由输入得到模型的输出
net = LRModel()
y = net(x)

pts = tf.concat([x, y_, y], axis = -1)

@tfmpl.figure_tensor
def draw_scatter(scaled, colors): 
    '''Draw scatter plots. One for each color.'''  
    fig = tfmpl.create_figures(1, figsize = (8, 4))[0]
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(scaled[:, 0], scaled[:, 1], c=colors[0])
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(scaled[:, 0], scaled[:, 2], c=colors[1])
    fig.tight_layout()
    return fig

saver = tf.train.Saver(tf.trainable_variables())

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint.
        saver.restore(sess, ckpt.model_checkpoint_path)

        print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)

    else:
        print('No checkpoint file found')
        exit

    image_tensor = draw_scatter(pts, ['r', 'b'])
    image_summary = tf.summary.image('scatter', image_tensor)      
    all_summaries = tf.summary.merge_all() 
    
    writer = tf.summary.FileWriter('output-test', sess.graph)
    summary = sess.run(all_summaries)
    writer.add_summary(summary, global_step = 0)
